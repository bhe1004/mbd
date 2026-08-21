"""Closed-loop trial and its record.

One trial drives the plant from home to a goal under one backend and one
schedule, and returns the quantities the paper reports. Every condition runs
through this function, so success, settling and latency are measured the same
way everywhere.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, Optional

import numpy as np

from .config import Config
from .planner import MBDPlanner, Schedule
from .plant import FrankaPlant

Array = np.ndarray


@dataclass
class TrialRecord:
    condition: str
    schedule: str
    model_seed: int
    target_idx: int
    lam: float
    final_err: float
    min_err: float
    steps: int              # first entry below the strict tolerance, else the cap
    reached: bool
    reached_strict: bool
    ms_per_step: float
    worst_ms: float
    deadline_misses: int    # control steps whose planning exceeded the period
    err_curve: Optional[list] = None    # per-step tracking error, for plateau analysis
    plateau_step: Optional[int] = None  # first step the error stops improving
    plan_end_pred: Optional[float] = None   # model's terminal error for the settled plan
    plan_end_true: Optional[float] = None   # true terminal error for the same plan
    # Update statistics, medians over the terminal stage of every control step.
    # n_eff is the effective sample size the precision floor divides by, so it
    # is the manipulation check for any study that moves the temperature.
    n_eff: Optional[float] = None
    n_eff_late: Optional[float] = None      # same, over the last third of the run
    dU: Optional[float] = None              # displacement per control step
    clip_frac: Optional[float] = None       # candidate mass on the input bound
    tcp_disp: Optional[float] = None        # metres of tool motion the step commands
    task_disp: Optional[float] = None       # part of the step the TCP responds to
    null_disp: Optional[float] = None       # part the arm's redundancy absorbs

    def to_json(self) -> Dict[str, Any]:
        return asdict(self)


def _plateau_step(curve: list, tol: float, patience: int = 15) -> Optional[int]:
    """First step after which the error never improves by more than ``tol``.

    Returns the step at which the run effectively stops making progress, or None
    if it keeps improving to the end. This separates a run that stalls early
    (plateau well before the budget) from one still descending at the cap.
    """
    best = float("inf")
    best_at = 0
    for i, e in enumerate(curve):
        if e < best - tol:
            best = e
            best_at = i
    # if the last meaningful improvement was long before the end, it plateaued
    if best_at <= len(curve) - patience:
        return best_at
    return None


def run_trial(cfg: Config, plant: FrankaPlant, backend, schedule: Schedule,
              goal: Array, *, condition: str, model_seed: int, target_idx: int,
              lam: float = 0.0, rng_seed: int = 0,
              record_curve: bool = False,
              probe_plan_end: bool = False,
              record_trace: bool = False,
              path_out: Optional[list] = None) -> TrialRecord:
    """Drive one reach and record it.

    ``record_curve`` stores the per-step tracking error for plateau analysis.
    ``probe_plan_end`` re-plans once at the settled state and rolls that plan
    through the model and the plant, so a stalled run can be labelled as sitting
    on a surrogate minimum (model says reached, plant says not) rather than
    merely running out of budget.
    ``record_trace`` opens up the update itself: the effective sample size, the
    displacement, and how that displacement splits into the part the TCP moves
    along and the part the arm absorbs in its null space.
    ``path_out``, if given, collects the executed tool center point at the start
    and after every control step, so a trial can be drawn as a path.
    """
    rng = np.random.default_rng(rng_seed)
    planner = MBDPlanner(cfg.planner.horizon, plant.act_dim, plant.limit,
                         schedule, cfg.planner.alpha, cfg.planner.eta,
                         cfg.planner.alpha_adaptive, cfg.planner.alpha_scale)

    state = plant.reset()
    if path_out is not None:
        path_out.append(plant.tcp(plant.observe(state)).copy())
    U = np.zeros((cfg.planner.horizon, plant.act_dim))
    min_err = float("inf")
    strict_at = None
    times: list[float] = []
    curve: list[float] = []
    period_ms = cfg.plant.control_dt * 1e3

    # Run the full budget without early stopping, so the final error is
    # comparable across conditions; the settling step is read off separately.
    stage_stats: list[dict] = []
    for step in range(cfg.task.steps):
        cost_fn = backend.cost_fn(state, goal)
        trace: list[dict] | None = [] if record_trace else None
        t0 = time.perf_counter()
        U = planner.plan(U, cost_fn, rng, trace=trace)
        times.append((time.perf_counter() - t0) * 1e3)

        if trace:
            # The terminal stage sets the executed command, so it is the one the
            # precision floor speaks about. Split its joint-space displacement
            # with the orthogonal projector J^+J: the arm is 7-DoF against a
            # 3-D task, so four directions move no tool center point at all and
            # any dispersion living there is invisible in the tracking error.
            last = dict(trace[-1])
            d0 = last.pop("dU0")
            j = plant.position_jacobian(
                np.asarray(state, dtype=np.float64)[: plant.num_joints])
            task_cmd = np.linalg.pinv(j) @ (j @ d0)
            last["tcp_disp"] = float(np.linalg.norm(j @ d0))
            last["task_disp"] = float(np.linalg.norm(task_cmd))
            last["null_disp"] = float(np.linalg.norm(d0 - task_cmd))
            stage_stats.append(last)

        state = plant.step(state, plant.clip(U[0]))
        tcp = plant.tcp(plant.observe(state))
        if path_out is not None:
            path_out.append(tcp.copy())
        err = float(np.linalg.norm(tcp - goal))
        curve.append(err)
        min_err = min(min_err, err)
        if strict_at is None and err <= cfg.task.strict:
            strict_at = step + 1
        U = MBDPlanner.warm_start(U)

    final_err = float(np.linalg.norm(plant.tcp(plant.observe(state)) - goal))
    times_arr = np.asarray(times)
    plateau = _plateau_step(curve, tol=cfg.task.strict)

    stats: Dict[str, Optional[float]] = dict.fromkeys(
        ("n_eff", "n_eff_late", "dU", "clip_frac",
         "tcp_disp", "task_disp", "null_disp"))
    if stage_stats:
        med = lambda k, rows: float(np.median([r[k] for r in rows]))  # noqa: E731
        late = stage_stats[len(stage_stats) * 2 // 3:] or stage_stats
        for k in ("n_eff", "dU", "clip_frac", "tcp_disp", "task_disp", "null_disp"):
            stats[k] = med(k, stage_stats)
        stats["n_eff_late"] = med("n_eff", late)

    pred_end = true_end = None
    if probe_plan_end:
        plan = planner.plan(U, backend.cost_fn(state, goal), rng)
        batch = plan[None]
        true_obs = plant.rollout_true(state, batch)
        true_end = float(np.linalg.norm(true_obs[0, -1, plant.num_joints:] - goal))
        if hasattr(backend, "model"):
            import torch
            b0 = torch.as_tensor(plant.observe(state), dtype=torch.float32)
            with torch.no_grad():
                model_obs = backend.model.rollout(
                    b0, torch.as_tensor(batch, dtype=torch.float32)).numpy()
            pred_end = float(np.linalg.norm(model_obs[0, -1, plant.num_joints:] - goal))
        else:
            pred_end = true_end

    return TrialRecord(
        condition=condition,
        schedule=schedule.name,
        model_seed=model_seed,
        target_idx=target_idx,
        lam=lam,
        final_err=final_err,
        min_err=min_err,
        steps=strict_at or cfg.task.steps,
        reached=min_err <= cfg.task.reach,
        reached_strict=min_err <= cfg.task.strict,
        ms_per_step=float(np.median(times_arr)),
        worst_ms=float(np.max(times_arr)),
        deadline_misses=int(np.sum(times_arr > period_ms)),
        err_curve=[round(e, 4) for e in curve] if record_curve else None,
        plateau_step=plateau,
        plan_end_pred=pred_end,
        plan_end_true=true_end,
        **stats,
    )


def plan_end_gap(cfg: Config, plant: FrankaPlant, backend, schedule: Schedule,
                 goal: Array, rng_seed: int = 0) -> tuple[float, float]:
    """Terminal error of one converged plan, as the model sees it and as it is.

    The same control sequence is rolled twice, once through the planner's model
    and once through the plant, so any difference is the landscape the optimizer
    settled on rather than the one it was aiming at.
    """
    rng = np.random.default_rng(rng_seed)
    planner = MBDPlanner(cfg.planner.horizon, plant.act_dim, plant.limit,
                         schedule, cfg.planner.alpha, cfg.planner.eta,
                         cfg.planner.alpha_adaptive, cfg.planner.alpha_scale)

    state = plant.reset()
    U = np.zeros((cfg.planner.horizon, plant.act_dim))
    for _ in range(cfg.task.steps):
        U = planner.plan(U, backend.cost_fn(state, goal), rng)
        state = plant.step(state, plant.clip(U[0]))
        U = MBDPlanner.warm_start(U)

    # At the settled state, optimize one plan and roll exactly that plan from the
    # same state through the model and through the plant. Any gap is the distance
    # between the landscape the optimizer settled on and the true one.
    plan = planner.plan(U, backend.cost_fn(state, goal), rng)
    batch = plan[None]

    true_obs = plant.rollout_true(state, batch)
    true_end = float(np.linalg.norm(true_obs[0, -1, plant.num_joints:] - goal))
    if not hasattr(backend, "model"):
        return true_end, true_end

    import torch
    b0 = torch.as_tensor(plant.observe(state), dtype=torch.float32)
    with torch.no_grad():
        model_obs = backend.model.rollout(
            b0, torch.as_tensor(batch, dtype=torch.float32)).numpy()
    pred_end = float(np.linalg.norm(model_obs[0, -1, plant.num_joints:] - goal))
    return pred_end, true_end
