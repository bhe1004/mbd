"""Stage 3 -- execute: plan against the wall clock while the arm moves.

Two execution modes, both driving the same planner through the same plant:

``async`` (deployment-like)
    The plant advances in wall-clock real time at the control period no matter
    what the planner is doing. A background thread replans from the newest
    state snapshot as fast as it can; every control boundary applies the newest
    finished plan *indexed by its age*, ``u = U[k]`` with
    ``k = floor((t_now - t_snapshot) / dt)``. Slow planning therefore shows up
    as stale actions (k >= 1), or -- if planning is far too slow -- as plans
    that expire before they are used (k >= horizon -> zero velocity). Nothing
    waits for the planner, which is the only honest way to ask whether it is
    fast enough.

``lockstep`` (benchmark)
    plan -> apply -> step, serialized. The motion still plays back at 1x
    between plans, so the reported reach times are in simulated time. Use it to
    measure the planner's serial replanning rate.

The referee that counts collisions is deliberately independent of the planner's
own penalty and margin-free: it tests the executed pose, not the predicted one.
"""

from __future__ import annotations

import threading
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch

from config import Config
from mbd.planner import BKMBDPlanner
from pipeline import viewer as vz
from pipeline.build import Environment, build_environment, build_planner, build_plant, \
    describe_setup
from pipeline import goals as goal_sources


# --------------------------------------------------------------- plan exchange
@dataclass
class Plan:
    controls: np.ndarray          # (T, action_dim)
    t_state: float                # clock time of the snapshot it started from
    prediction: np.ndarray        # predicted tool path, for the overlay
    latency: float                # plan wall time [s]


@dataclass
class Snapshot:
    features: np.ndarray
    ee: np.ndarray
    goal: np.ndarray          # where the target is now (reporting, noise schedule)
    goal_horizon: np.ndarray  # where it will be at each planned step, (T, 3)
    t: float


class PlanExchange:
    """The only shared state between the control loop and the planner thread."""

    def __init__(self, snapshot: Snapshot) -> None:
        self.lock = threading.Lock()
        self.snapshot = snapshot
        self.plan: Optional[Plan] = None
        self.stop = False
        self.latencies: List[float] = []

    def publish_state(self, snapshot: Snapshot) -> Optional[Plan]:
        with self.lock:
            self.snapshot = snapshot
            return self.plan

    def take_state(self):
        with self.lock:
            return (None if self.stop else self.snapshot)

    def publish_plan(self, plan: Plan) -> None:
        with self.lock:
            self.plan = plan
            self.latencies.append(plan.latency)

    def snapshot_latencies(self) -> List[float]:
        with self.lock:
            return list(self.latencies)

    def request_stop(self) -> None:
        with self.lock:
            self.stop = True


# ------------------------------------------------------------------ segments
@dataclass
class Segment:
    """One target the arm is asked to reach."""

    target_id: int
    goal: np.ndarray
    t_start: float
    t_reach: Optional[float] = None
    min_error: float = float("inf")


@dataclass
class RunLog:
    segments: List[Segment] = field(default_factory=list)
    errors: List[float] = field(default_factory=list)
    applied_k: List[int] = field(default_factory=list)
    plan_age_ms: List[float] = field(default_factory=list)
    staleness: Counter = field(default_factory=Counter)
    expired: int = 0
    boundaries_without_plan: int = 0
    violations: List[int] = field(default_factory=list)
    qpos: List[np.ndarray] = field(default_factory=list)
    qvel: List[np.ndarray] = field(default_factory=list)
    controls: List[np.ndarray] = field(default_factory=list)
    ee: List[np.ndarray] = field(default_factory=list)
    goals: List[np.ndarray] = field(default_factory=list)


class Runner:
    """Drive one planner through a target sequence, in real time."""

    def __init__(self, cfg: Config, env: Environment, planner: BKMBDPlanner) -> None:
        self.cfg = cfg
        self.env = env
        self.task = env.task
        self.planner = planner
        self.period = cfg.task.control_dt
        self.horizon = cfg.task.horizon
        self.action_dim = env.task.action_dim
        self.strict = cfg.task.strict_threshold
        self.min_plan_s = cfg.run.min_plan_ms / 1000.0
        self.rng = np.random.default_rng(cfg.mbd.seed)
        self.log = RunLog()
        self.source = goal_sources.build(cfg, env.task)
        self.tracking: List[float] = []      # dynamic goals: |tool - target| per step
        self.plant = None
        self._lockstep_latencies: List[float] = []

    # ------------------------------------------------------- planner thread
    def _planner_loop(self, exchange: PlanExchange) -> None:
        """Replan continuously from the newest snapshot (async mode).

        The warm start is shifted by however many control periods elapsed
        between the previous snapshot and this one -- the receding-horizon
        roll-by, but measured in real time instead of assumed to be one step.
        """

        U = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        previous_t = None
        while True:
            snapshot = exchange.take_state()
            if snapshot is None:
                return

            if previous_t is not None:
                shift = int(np.clip(round((snapshot.t - previous_t) / self.period),
                                    0, self.horizon))
                if shift >= self.horizon:
                    U = np.zeros_like(U)          # the whole plan is stale
                elif shift > 0:
                    U = np.roll(U, -shift, axis=0)
                    U[self.horizon - shift:] = U[self.horizon - shift - 1]

            err = self.task.error(snapshot.ee, snapshot.goal)
            t0 = time.perf_counter()
            result = self.planner.plan(snapshot.features, snapshot.goal_horizon,
                                       U, err, self.rng)
            latency = time.perf_counter() - t0
            if latency < self.min_plan_s:         # emulate a slower planner
                time.sleep(self.min_plan_s - latency)
                latency = time.perf_counter() - t0

            U = result.controls
            exchange.publish_plan(Plan(
                controls=U.copy(), t_state=snapshot.t,
                prediction=self.task.tool_of(torch.as_tensor(
                    result.predicted_features)).numpy(),
                latency=latency))
            previous_t = snapshot.t

    # ------------------------------------------------------------------- run
    def run(self) -> str:
        cfg = self.cfg
        dynamic = self.source.dynamic

        key_callback = (self.source.on_key
                        if cfg.goal.mode == "keyboard" else None)
        self.plant = build_plant(cfg, self.env, key_callback=key_callback)
        plant = self.plant
        obs = plant.reset(self.env.start_q)
        features = self.task.features(obs.q, obs.ee)
        # The source is placed relative to where the tool actually starts, so a
        # path target begins at the tool instead of stepping to it.
        self.source.start(obs.ee, 0.0)
        goal = self.source.goal(0.0)

        # ---- warm-up: throwaway plans, excluded from every statistic --------
        if cfg.run.warmup_plans > 0:
            t_warm = time.perf_counter()
            self.planner.warmup(features, goal, cfg.run.warmup_plans)
            print(f"warm-up: {cfg.run.warmup_plans} throwaway plans in "
                  f"{time.perf_counter() - t_warm:.2f} s")

        scn = plant.user_scene()
        vz.draw_start_pose(scn, goal=goal, threshold=self.strict,
                           obstacles=self.env.obstacles,
                           robot=self.env.robot if cfg.viewer.collision_spheres else None,
                           q=obs.q)
        plant.sync()
        if dynamic:
            print(f"\ntarget: {self.source.label()}\n")
        if cfg.run.wait_for_start:
            try:
                input("ready at the start pose - press Enter to start motion... ")
            except EOFError:
                pass
        # Only now: a raw-mode key reader would have eaten the prompt above.
        self.source.activate()

        horizon_goal = self._horizon_goal(0.0, goal)
        exchange = PlanExchange(
            Snapshot(features, obs.ee, goal, horizon_goal, time.perf_counter()))
        thread = None
        if cfg.run.mode == "async":
            thread = threading.Thread(target=self._planner_loop, args=(exchange,),
                                      daemon=True)

        try:
            wall = self._control_loop(exchange, thread, goal)
        finally:
            exchange.request_stop()
            if thread is not None and thread.is_alive():
                # Let the planner finish its current plan before torch/MuJoCo are
                # torn down, so its teardown does not race interpreter shutdown.
                thread.join(timeout=10.0)
            # Hand the terminal back before anything else reads stdin (the save
            # prompt below) and unconditionally, so a crash mid-run cannot leave
            # the operator with a terminal stuck in raw mode.
            self.source.close()

        latencies = (exchange.snapshot_latencies() if cfg.run.mode == "async"
                     else self._lockstep_latencies)
        report = self.summarize(wall, latencies)
        replay = self._maybe_save_replay()
        if replay is not None:
            report += f"\nreplay: {replay}"
        plant.close()
        return report

    def _horizon_goal(self, t: float, goal: np.ndarray) -> np.ndarray:
        """The target for each planned step, or the present one held throughout."""

        if not self.cfg.goal.predict:
            return goal
        return self.source.horizon(t, self.horizon, self.period)

    def _control_loop(self, exchange: PlanExchange, thread, goal: np.ndarray) -> float:
        """The boundary loop: measure, apply the newest plan, advance one period."""

        cfg = self.cfg
        plant = self.plant
        log = self.log
        dynamic = self.source.dynamic
        scn = plant.user_scene()
        trail: List[np.ndarray] = []
        ghosts: List = []
        last_ghost = -1e9
        last_status = 0.0
        self._lockstep_latencies: List[float] = []

        segment = Segment(cfg.run.target_ids[0], goal.copy(), 0.0)
        seg_index = 0
        U_lock = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        finished = False

        obs = plant.observe()
        log.qpos.append(obs.q.copy())
        log.qvel.append(obs.qd.copy())
        log.ee.append(obs.ee.copy())
        log.goals.append(goal.copy())

        if thread is not None:
            thread.start()
        plant.start_clock()
        t0 = time.perf_counter()
        boundary = 0

        try:
            while plant.is_running() and not finished:
                now = time.perf_counter()
                wall = now - t0
                # In async, wall time IS simulated time (the plant is paced hard).
                # In lockstep the plant is decoupled from the wall clock, so
                # segment timing is reported in simulated time.
                t_seg = wall if cfg.run.mode == "async" else boundary * self.period
                if wall >= cfg.run.max_time:
                    break

                if dynamic:
                    goal = self.source.goal(t_seg)
                    segment.goal = goal

                # ---- measure -------------------------------------------------
                obs = plant.observe()
                features = self.task.features(obs.q, obs.ee)
                err = self.task.error(obs.ee, goal)
                segment.min_error = min(segment.min_error, err)
                log.errors.append(err)
                trail.append(obs.ee.copy())
                if dynamic and len(trail) > 300:      # keep a recent tail only
                    del trail[0]

                # ---- the referee: margin-free, on the EXECUTED pose ----------
                if self.env.obstacles is not None:
                    hits = self.env.obstacles.violations(
                        self.env.robot.spheres_np(obs.q), self.env.robot.sphere_radii_np)
                    log.violations.append(hits)
                    if hits and (len(log.violations) < 2 or not log.violations[-2]):
                        print(f"[{wall:6.2f}s] COLLISION: {hits} arm sphere(s) inside "
                              "an obstacle")

                # ---- reach / settle / advance -------------------------------
                # A moving target never "completes": there is nothing to settle
                # on and no next segment, so the run ends on time and the metric
                # is tracking error, collected here.
                if dynamic:
                    self.tracking.append(err)
                elif segment.t_reach is None and err < self.strict:
                    segment.t_reach = t_seg - segment.t_start
                    print(f"[{t_seg:6.2f}s] target {segment.target_id} reached "
                          f"({err * 1000:.1f} mm) in {segment.t_reach:.2f} s")
                if (not dynamic and segment.t_reach is not None
                        and t_seg - segment.t_start - segment.t_reach >= cfg.run.settle_time):
                    log.segments.append(segment)
                    seg_index += 1
                    if seg_index >= len(cfg.run.target_ids):
                        if not cfg.run.cycle:
                            finished = True
                            break
                        seg_index = 0
                    target_id = cfg.run.target_ids[seg_index]
                    goal = self.env.goal_for(target_id, cfg)
                    self.source.set(goal)
                    segment = Segment(target_id, goal.copy(), t_seg)
                    print(f"[{t_seg:6.2f}s] next target -> {target_id} "
                          f"{np.round(goal, 3)}")

                # ---- decide the command --------------------------------------
                prediction = None
                if cfg.run.mode == "async":
                    plan = exchange.publish_state(Snapshot(
                        features, obs.ee, goal, self._horizon_goal(t_seg, goal), now))
                    if plan is None:
                        u = np.zeros(self.action_dim)
                        log.boundaries_without_plan += 1
                    else:
                        age = now - plan.t_state
                        k = int(age / self.period)
                        log.plan_age_ms.append(age * 1000.0)
                        if k >= self.horizon:
                            u = np.zeros(self.action_dim)   # the plan expired
                            log.expired += 1
                            log.applied_k.append(-1)
                        else:
                            u = plan.controls[k]
                            log.staleness[k] += 1
                            log.applied_k.append(k)
                        prediction = plan.prediction
                else:
                    t_plan = time.perf_counter()
                    result = self.planner.plan(features, self._horizon_goal(t_seg, goal),
                                               U_lock, err, self.rng)
                    elapsed = time.perf_counter() - t_plan
                    if elapsed < self.min_plan_s:
                        time.sleep(self.min_plan_s - elapsed)
                        elapsed = time.perf_counter() - t_plan
                    self._lockstep_latencies.append(elapsed)
                    U_lock = result.controls
                    u = U_lock[0]
                    prediction = self.task.tool_of(
                        torch.as_tensor(result.predicted_features)).numpy()
                    U_lock = np.roll(U_lock, -1, axis=0)
                    U_lock[-1] = U_lock[-2]

                u = plant.clip(u)
                plant.send(u)
                log.controls.append(u.copy())

                # ---- show what the planner is aiming at ----------------------
                # The plant decides how: its own window, or published markers.
                plant.publish_debug(goal=goal, prediction=prediction)
                if scn is not None:
                    if cfg.viewer.ghosts.enabled and t_seg - last_ghost >= cfg.viewer.ghosts.interval_s:
                        ghosts.append(vz.build_ghost(plant.model, obs.q.copy()))
                        if len(ghosts) > cfg.viewer.ghosts.max_ghosts:
                            ghosts.pop(0)
                        last_ghost = t_seg
                    extra = ([lambda s: vz.draw_ghosts(s, ghosts, plant.model,
                                                       cfg.viewer.ghosts.alpha,
                                                       cfg.viewer.ghosts.fade)]
                             if ghosts else ())
                    vz.draw_overlay(scn, goal=goal, threshold=self.strict, trail=trail,
                                    prediction=prediction, obstacles=self.env.obstacles,
                                    show_trail=cfg.viewer.trail,
                                    show_prediction=cfg.viewer.prediction, extra=extra)

                # ---- let one control period of wall time pass ----------------
                plant.advance(resync=(cfg.run.mode == "lockstep"))
                boundary += 1

                after = plant.observe()
                log.qpos.append(after.q.copy())
                log.qvel.append(after.qd.copy())
                log.ee.append(after.ee.copy())
                log.goals.append(goal.copy())

                if wall - last_status >= 1.0:
                    last_status = wall
                    lat = (exchange.snapshot_latencies() if cfg.run.mode == "async"
                           else self._lockstep_latencies)
                    recent = 1000 * np.mean(lat[-20:]) if lat else float("nan")
                    where = (f"target {np.round(goal, 3)}" if dynamic
                             else f"target {segment.target_id}")
                    print(f"[{wall:6.2f}s] {where} err={err:.3f} m | "
                          f"{len(lat)} plans, recent {recent:.0f} ms/plan "
                          f"({len(lat) / max(wall, 1e-9):.1f} Hz)")
        except KeyboardInterrupt:
            print("\ninterrupted")

        if not finished and segment.min_error < np.inf:
            log.segments.append(segment)
        return time.perf_counter() - t0

    # ---------------------------------------------------------------- output
    def summarize(self, wall: float, latencies: List[float]) -> str:
        cfg = self.cfg
        log = self.log
        lines = ["", "=" * 72,
                 f"REAL-TIME VERDICT  ({self.planner.name}, mode={cfg.run.mode})",
                 "=" * 72]

        lat = np.asarray(latencies) * 1000.0
        if lat.size:
            lines.append(
                f"plan latency: mean {lat.mean():.1f} ms | p50 {np.percentile(lat, 50):.1f}"
                f" | p95 {np.percentile(lat, 95):.1f} | max {lat.max():.1f}"
                f"   ({lat.size} plans, {lat.size / max(wall, 1e-9):.1f} Hz replanning)")
            lines.append(
                f"control period: {self.period * 1000:.0f} ms "
                f"({1 / self.period:.0f} Hz); horizon {self.horizon} steps = "
                f"{self.horizon * self.period:.2f} s lookahead")

        if cfg.run.mode == "async":
            applied = sum(log.staleness.values())
            hist = ", ".join(f"k={k}: {n} ({100 * n / max(applied, 1):.0f}%)"
                             for k, n in sorted(log.staleness.items()))
            lines.append(f"action staleness (applied U[k]): {hist or 'none'}")
            lines.append(f"expired plans (k>=horizon -> zero velocity): {log.expired}; "
                         f"boundaries before the first plan: {log.boundaries_without_plan}")
            overrun = self.plant.last_overrun if self.plant is not None else 0.0
            lines.append(f"worst real-time pacing overrun: {overrun * 1000:.1f} ms")

        if log.violations:
            bad = sum(1 for v in log.violations if v)
            lines.append(
                f"whole-arm collision (referee, margin-free): {bad}/{len(log.violations)} "
                "control boundaries with an arm sphere inside an obstacle"
                + ("" if bad == 0 else "  <- NOT collision-free"))

        for seg in log.segments:
            status = (f"reached in {seg.t_reach:.2f} s" if seg.t_reach is not None
                      else f"NOT reached (min err {seg.min_error:.3f} m)")
            lines.append(f"target {seg.target_id}: {status}")

        if self.tracking:
            e = np.asarray(self.tracking) * 1000.0
            lines.append(f"target: {self.source.label().splitlines()[0]}")
            lines.append(
                f"tracking error: mean {e.mean():.1f} mm | p50 {np.percentile(e, 50):.1f}"
                f" | p95 {np.percentile(e, 95):.1f} | max {e.max():.1f}"
                f"   ({e.size} boundaries)")
            settled = e[len(e) // 3:]            # after the target has ramped in
            lines.append(f"  steady state (last two thirds): mean {settled.mean():.1f} mm"
                         f" | max {settled.max():.1f} mm")
            lines.append(f"  horizon prediction of the target: "
                         f"{'on' if cfg.goal.predict else 'OFF (expect a standing lag)'}")

        reached_all = bool(log.segments) and all(s.t_reach is not None for s in log.segments)
        if self.source.dynamic:
            lines.append("verdict: n/a - a moving target never 'completes'; read the "
                         "tracking error above")
        elif cfg.run.mode == "async":
            if reached_all and log.expired == 0:
                lines.append(
                    "verdict: PASS - the plant ran in wall-clock real time and every "
                    "target was reached with stale-indexed plans; the planner is usable "
                    f"at {1 / self.period:.0f} Hz control.")
            elif reached_all:
                lines.append("verdict: MARGINAL - targets reached, but some plans "
                             "expired before use; the planner only barely keeps up.")
            else:
                lines.append("verdict: FAIL - the planner could not keep up with real "
                             "time (targets missed under wall-clock execution).")
        elif lat.size:
            lines.append(
                f"serial replanning rate if deployed lockstep: {1000.0 / lat.mean():.1f} Hz "
                f"(real-time factor {self.period * 1000 / lat.mean():.2f}x vs the "
                f"{self.period * 1000:.0f} ms control period)")
        lines.append("=" * 72)
        return "\n".join(lines)

    def _maybe_save_replay(self) -> Optional[Path]:
        mode = self.cfg.run.save_replay
        if mode == "never" or not self.log.controls:
            return None
        if mode == "ask":
            try:
                answer = input("\nsave this trajectory for replay? [y/N]: ").strip().lower()
            except EOFError:
                answer = "n"
            if answer not in ("y", "yes"):
                print("not saved.")
                return None

        out_dir = self.cfg.paths.output_dir / "replays"
        out_dir.mkdir(parents=True, exist_ok=True)
        path = out_dir / (f"{self.planner.name}_{self.cfg.run.mode}_"
                          f"{time.strftime('%Y%m%d_%H%M%S')}.npz")
        log = self.log
        np.savez_compressed(
            path,
            qpos=np.stack(log.qpos), qvel=np.stack(log.qvel),
            controls=np.stack(log.controls), ee=np.stack(log.ee),
            goals=np.stack(log.goals), errors=np.asarray(log.errors),
            applied_k=np.asarray(log.applied_k, dtype=np.int64),
            plan_age_ms=np.asarray(log.plan_age_ms),
            violations=np.asarray(log.violations, dtype=np.int64),
            period=self.period, mode=self.cfg.run.mode,
            method=self.planner.name, config=str(self.cfg.source),
        )
        print(f"replay saved: {path}")
        return path


def run(cfg: Config) -> None:
    """Entry point for ``python main.py run``."""

    if cfg.run.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"run.device is {cfg.run.device!r} but CUDA is unavailable")
    if cfg.run.torch_threads > 0:
        # The library default (one thread per core) causes 10-30x straggler
        # spikes in plan latency on a loaded desktop; a small fixed pool is flat.
        torch.set_num_threads(cfg.run.torch_threads)
    device = torch.device(cfg.run.device)

    env = build_environment(cfg, device=device)
    for line in describe_setup(cfg, env):
        print(line)
    planner = build_planner(cfg, env, device=device)
    print(f"mode={cfg.run.mode} plant={cfg.run.plant} targets={cfg.run.target_ids}"
          + (f"  goal={cfg.goal.mode}" if cfg.goal.mode != "fixed" else ""))

    report = Runner(cfg, env, planner).run()
    print(report)
