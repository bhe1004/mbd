"""The real-time execution harness -- planner-agnostic.

Drives the FR3 through a target sequence, either:

- ``async``: the sim advances in wall-clock real time at the control period; a
  background thread replans from the newest snapshot as fast as it can, and
  every control boundary applies the newest finished plan indexed by its age
  (``u = U[k]``). Slow planning shows up as stale actions or expired plans.
- ``lockstep``: plan -> apply -> step, serialized, with live timing.

The planner is used only through its ``plan(x, goal, u_prev, U_warm, err, rng)``
and ``warmup(...)`` interface, so BK-MBD and SQP-MPC run through the same loop.
The referee (margin-free ``obstacle.violations``) counts whole-arm collisions
independently of the planner's own penalty.
"""

from __future__ import annotations

import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402
import mujoco.viewer  # noqa: E402

from envs.franka import NUM_JOINTS  # noqa: E402

from secy.config import Config  # noqa: E402
from secy.environment import Scene  # noqa: E402
from secy import viewer as vz  # noqa: E402
from secy import linearization_view as linview  # noqa: E402


@dataclass
class Plan:
    U: np.ndarray          # (T, 7) planned control sequence
    t_state: float         # wall time of the snapshot the plan starts from
    ee_pred: np.ndarray    # predicted EE path for the overlay
    latency: float         # plan() wall time [s]
    index: int


class _SharedState:
    """Snapshot / plan exchange between the control and planner threads."""

    def __init__(self, x0: np.ndarray, goal0: np.ndarray) -> None:
        self.lock = threading.Lock()
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.x_time = time.perf_counter()
        self.goal = np.asarray(goal0, dtype=np.float64).copy()
        self.u_prev = np.zeros(NUM_JOINTS)
        self.plan: Optional[Plan] = None
        self.stop = False
        self.latencies: List[float] = []


class RealtimeRunner:
    """Run a chosen planner against the wall clock (async) or lockstep."""

    def __init__(self, scene: Scene, planner, cfg: Config) -> None:
        self.scene = scene
        self.planner = planner
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.mbd.seed)
        self.min_plan_s = cfg.runtime.min_plan_ms / 1000.0

        self.period = scene.period
        self.horizon = scene.horizon
        self.strict = scene.strict
        self.target_ids = list(cfg.runtime.target_ids)

    # ------------------------------------------------------ async planner thread
    def _planner_loop(self, shared: _SharedState) -> None:
        """Continuously replan from the newest snapshot. The warm start shifts
        the previous solution by the number of control periods that elapsed
        between the two snapshots (the receding-horizon roll-by)."""

        U = np.zeros((self.horizon, NUM_JOINTS), dtype=np.float64)
        prev_t_state = None
        index = 0
        while True:
            with shared.lock:
                if shared.stop:
                    return
                x = shared.x.copy()
                t_state = shared.x_time
                goal = shared.goal.copy()
                u_prev = shared.u_prev.copy()

            if prev_t_state is not None:
                shift = int(np.clip(round((t_state - prev_t_state) / self.period),
                                    0, self.horizon))
                if shift >= self.horizon:
                    U = np.zeros_like(U)
                elif shift > 0:
                    U = np.roll(U, -shift, axis=0)
                    U[self.horizon - shift:] = U[self.horizon - shift - 1]

            err = float(np.linalg.norm(
                self.scene.task.ee_of_q(x[:NUM_JOINTS]) - goal))
            t0 = time.perf_counter()
            U, ee_pred = self.planner.plan(x, goal, u_prev, U, err, self.rng)
            latency = time.perf_counter() - t0
            if latency < self.min_plan_s:
                time.sleep(self.min_plan_s - latency)
                latency = time.perf_counter() - t0

            index += 1
            plan = Plan(U.copy(), t_state, ee_pred, latency, index)
            with shared.lock:
                shared.plan = plan
                shared.latencies.append(latency)
            prev_t_state = t_state

    def _prompt_save_replay(self, qpos_log, qvel_log, control_log,
                            ee_log, goal_log) -> Optional[Path]:
        while True:
            try:
                choice = input(
                    "\n실행 trajectory를 저장할까요? 저장=1, 아니오=0: "
                ).strip()
            except EOFError:
                choice = "0"
            if choice in {"0", "1"}:
                break
            print("1 또는 0을 입력하세요.")

        if choice == "0":
            print("replay를 저장하지 않았습니다.")
            return None

        output_dir = PROJECT_ROOT / "out" / "replays"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        path = output_dir / f"{self.planner.name}_{self.cfg.runtime.mode}_{stamp}.npz"
        qpos = np.stack(qpos_log)
        qvel = np.stack(qvel_log)
        controls = (np.stack(control_log)
                    if control_log else np.zeros((0, NUM_JOINTS)))
        np.savez_compressed(
            path,
            qpos=qpos,
            qvel=qvel,
            controls=controls,
            ee=np.stack(ee_log),
            goals=np.stack(goal_log),
            period=np.asarray(self.period),
            method=np.asarray(self.planner.name),
            mode=np.asarray(self.cfg.runtime.mode),
            config_source=np.asarray(str(self.cfg.source.resolve())),
        )
        print(f"replay saved: {path}")
        return path

    # ------------------------------------------------------------------- run
    def run(self, interactive: bool = True,
            max_boundaries: Optional[int] = None) -> str:
        """Drive the planner through the target sequence.

        ``interactive`` False (batch use) skips the save-replay prompt and
        instead populates ``self.last_result`` with per-episode metrics.
        ``max_boundaries`` caps the episode by *sim* control steps (reproducible
        regardless of wall-clock speed) in addition to ``cfg.runtime.max_time``.
        """
        self.last_result = None
        cfg = self.cfg
        scene = self.scene
        sim = scene.sim
        model = scene.scene_model
        tcp = scene.tcp
        color = self.planner.color

        x0 = np.concatenate([sim.qpos[:NUM_JOINTS], sim.qvel[:NUM_JOINTS]])
        shared = _SharedState(x0, scene.goal_for(self.target_ids[0]))

        planner_thread = None
        if cfg.runtime.mode == "async":
            planner_thread = threading.Thread(
                target=self._planner_loop, args=(shared,), daemon=True)

        viewer = None
        if cfg.runtime.viewer:
            viewer = mujoco.viewer.launch_passive(model, sim).__enter__()

        # Optional linearized-region overlay: a planner that exposes
        # ``last_linearization`` (the SQP-MPC) has its committed convex faces
        # drawn translucently. The closure reads the attribute live each frame.
        overlays = []
        if hasattr(self.planner, "last_linearization") and cfg.linearization.enabled:
            lin = cfg.linearization
            overlays.append(lambda scn: linview.draw_faces(
                scn, getattr(self.planner, "last_linearization", None),
                arrows=lin.arrows, fill=lin.fill, fill_depth=lin.fill_depth))

        # Ghost trail: leave a translucent robot at every ``interval_s`` of sim
        # time. ``ghosts`` grows during the run; the closure reads it live, and
        # it is drawn first (behind the trail / faces).
        ghosts: list = []
        last_ghost_t = [-1e9]
        if cfg.ghost.enabled and viewer is not None:
            overlays.insert(0, lambda scn: vz.draw_ghosts(
                scn, ghosts, model, cfg.ghost.alpha, cfg.ghost.fade))
        overlays = overlays or None

        # ---- logs ----------------------------------------------------------
        errors: List[float] = []
        k_log: List[int] = []
        age_log: List[float] = []
        k_counter: Counter = Counter()
        misses = 0
        warmup_boundaries = 0
        max_overrun = 0.0
        lockstep_latencies: List[float] = []
        segments: List[dict] = []
        trail: List[np.ndarray] = []
        viol_log: List[int] = []
        replay_qpos = [sim.qpos[:NUM_JOINTS].copy()]
        replay_qvel = [sim.qvel[:NUM_JOINTS].copy()]
        replay_controls: List[np.ndarray] = []
        replay_ee = [sim.site_xpos[tcp].copy()]

        seg_idx = 0
        goal = scene.goal_for(self.target_ids[seg_idx])
        replay_goals = [goal.copy()]
        seg = {"target_id": self.target_ids[seg_idx], "t_start": 0.0,
               "t_reach": None, "min_err": np.inf}
        done = False
        U_lock = np.zeros((self.horizon, NUM_JOINTS), dtype=np.float64)
        u_prev_lock = np.zeros(NUM_JOINTS)
        last_status = 0.0
        sync_every = max(1, scene.nsub // 5)

        def viewer_alive() -> bool:
            return viewer is None or viewer.is_running()

        def advance_segment(wall: float) -> bool:
            nonlocal seg_idx, goal, seg
            segments.append(seg)
            seg_idx += 1
            if seg_idx >= len(self.target_ids):
                return True
            goal = scene.goal_for(self.target_ids[seg_idx])
            seg = {"target_id": self.target_ids[seg_idx], "t_start": wall,
                   "t_reach": None, "min_err": np.inf}
            print(f"[{wall:6.2f}s] next target -> {self.target_ids[seg_idx]} "
                  f"{np.round(goal, 3)}")
            return False

        # ---- warm-up + wait-for-start -------------------------------------
        self.planner.warmup(x0, goal, cfg.runtime.warmup_plans)
        if cfg.runtime.warmup_plans > 0:
            print(f"warm-up: {cfg.runtime.warmup_plans} throwaway plans")

        if cfg.env.wait_for_start:
            if viewer is not None:
                vz.draw_start_pose(viewer, goal, self.strict, color,
                                   scene.obstacle,
                                   scene.fk if cfg.collision_view.spheres else None,
                                   sim.qpos[:NUM_JOINTS])
            try:
                input("\nready at the start pose — press Enter to start motion... ")
            except EOFError:
                pass

        if planner_thread:
            planner_thread.start()
        t0_wall = time.perf_counter()
        boundary = 0
        try:
            while viewer_alive() and not done:
                now = time.perf_counter()
                wall = now - t0_wall
                t_seg = wall if cfg.runtime.mode == "async" else boundary * self.period
                if wall >= cfg.runtime.max_time:
                    break
                if max_boundaries is not None and boundary >= max_boundaries:
                    break

                # ---- ghost trail: snapshot the pose every interval_s --------
                if (cfg.ghost.enabled and viewer is not None
                        and t_seg - last_ghost_t[0] >= cfg.ghost.interval_s):
                    ghosts.append(vz.build_ghost(model, sim.qpos[:NUM_JOINTS].copy()))
                    if len(ghosts) > cfg.ghost.max_ghosts:
                        ghosts.pop(0)
                    last_ghost_t[0] = t_seg

                # ---- control boundary: measure, (plan,) apply --------------
                x = np.concatenate([sim.qpos[:NUM_JOINTS], sim.qvel[:NUM_JOINTS]])
                ee = sim.site_xpos[tcp].copy()
                err = float(np.linalg.norm(ee - goal))
                seg["min_err"] = min(seg["min_err"], err)
                errors.append(err)
                if scene.obstacle is not None:
                    nv = scene.obstacle.violations(
                        scene.fk.spheres_np(x[:NUM_JOINTS]), scene.fk.radii_np)
                    viol_log.append(nv)
                    if nv and (not viol_log[-2:-1] or not viol_log[-2]):
                        print(f"[{wall:6.2f}s] COLLISION: {nv} arm sphere(s) "
                              f"inside an obstacle")
                trail.append(ee)

                if seg["t_reach"] is None and err < self.strict:
                    seg["t_reach"] = t_seg - seg["t_start"]
                    print(f"[{t_seg:6.2f}s] target {seg['target_id']} strict reach "
                          f"({err * 1000:.1f} mm) in {seg['t_reach']:.2f} s")
                if seg["t_reach"] is not None and (
                        t_seg - seg["t_start"] - seg["t_reach"] >= cfg.runtime.settle_time):
                    done = advance_segment(t_seg)
                    if done:
                        break

                ee_pred = None
                if cfg.runtime.mode == "async":
                    with shared.lock:
                        shared.x = x
                        shared.x_time = now
                        shared.goal = goal.copy()
                        plan = shared.plan
                    if plan is None:
                        u = np.zeros(NUM_JOINTS)
                        warmup_boundaries += 1
                    else:
                        k = int((now - plan.t_state) / self.period)
                        age_log.append((now - plan.t_state) * 1000.0)
                        if k >= self.horizon:
                            u = np.zeros(NUM_JOINTS)
                            misses += 1
                            k_log.append(-1)
                        else:
                            u = plan.U[k]
                            k_counter[k] += 1
                            k_log.append(k)
                        ee_pred = plan.ee_pred
                    with shared.lock:
                        shared.u_prev = u.copy()
                else:  # lockstep: plan synchronously (the viewer freezes meanwhile)
                    t_plan = time.perf_counter()
                    U_lock, ee_pred = self.planner.plan(
                        x, goal, u_prev_lock, U_lock, err, self.rng)
                    if time.perf_counter() - t_plan < self.min_plan_s:
                        time.sleep(self.min_plan_s - (time.perf_counter() - t_plan))
                    lockstep_latencies.append(time.perf_counter() - t_plan)
                    u = U_lock[0]
                    U_lock = np.roll(U_lock, -1, axis=0)
                    U_lock[-1] = U_lock[-2]
                    u_prev_lock = u.copy()

                u = np.clip(u, scene.task.action_bounds[0], scene.task.action_bounds[1])
                sim.ctrl[:] = u
                replay_controls.append(u.copy())

                if viewer:
                    vz.draw_overlay(viewer, goal, self.strict, trail, ee_pred,
                                    color, obstacle=scene.obstacle, overlays=overlays,
                                    show_trail=cfg.path_line.trail,
                                    show_pred=cfg.path_line.prediction)

                # ---- advance the plant one control period ------------------
                # async: paced against the global wall clock (hard real time);
                # lockstep: paced locally so the motion segment plays at 1x.
                base = t0_wall if cfg.runtime.mode == "async" else time.perf_counter()
                for i in range(scene.nsub):
                    mujoco.mj_step(model, sim)
                    if cfg.runtime.mode == "async":
                        deadline = base + (boundary * scene.nsub + i + 1) * scene.sub_dt
                    else:
                        deadline = base + (i + 1) * scene.sub_dt
                    if viewer and (i % sync_every == sync_every - 1):
                        viewer.sync()
                    slack = deadline - time.perf_counter()
                    if slack > 0:
                        if viewer or cfg.runtime.mode == "async":
                            time.sleep(slack)
                    else:
                        max_overrun = max(max_overrun, -slack)
                replay_qpos.append(sim.qpos[:NUM_JOINTS].copy())
                replay_qvel.append(sim.qvel[:NUM_JOINTS].copy())
                replay_ee.append(sim.site_xpos[tcp].copy())
                replay_goals.append(goal.copy())
                boundary += 1

                if wall - last_status >= 1.0:
                    last_status = wall
                    with shared.lock:
                        lat = list(shared.latencies)
                    lat = lockstep_latencies if cfg.runtime.mode == "lockstep" else lat
                    lat_ms = 1000 * np.mean(lat[-20:]) if lat else float("nan")
                    print(f"[{wall:6.2f}s] target {seg['target_id']} err={err:.3f} m | "
                          f"{len(lat)} plans, recent {lat_ms:.0f} ms/plan "
                          f"({len(lat) / max(wall, 1e-9):.1f} Hz)")
        except KeyboardInterrupt:
            print("\ninterrupted")
        finally:
            with shared.lock:
                shared.stop = True
            # Let the planner thread finish its current plan and exit before we
            # tear down torch/MuJoCo, so its teardown does not race interpreter
            # shutdown (which otherwise prints "terminate called ...").
            if planner_thread is not None and planner_thread.is_alive():
                planner_thread.join(timeout=10.0)

        wall = time.perf_counter() - t0_wall
        if not done and seg["min_err"] < np.inf:
            segments.append(seg)
        with shared.lock:
            latencies = (list(shared.latencies) if cfg.runtime.mode == "async"
                         else lockstep_latencies)

        # per-episode metrics for batch runs (interactive=False)
        seg0 = segments[0] if segments else None
        any_viol = bool(viol_log) and any(v > 0 for v in viol_log)
        self.last_result = {
            "method": self.planner.name,
            "reached_strict": bool(seg0 is not None and seg0["t_reach"] is not None),
            "t_reach": (float(seg0["t_reach"])
                        if seg0 and seg0["t_reach"] is not None else None),
            "min_err": (float(seg0["min_err"]) if seg0 else float("inf")),
            "strict_threshold": float(self.strict),
            "executed_violation": any_viol,
            "n_viol_boundaries": (int(sum(1 for v in viol_log if v))
                                  if viol_log else 0),
            "n_boundaries": int(boundary),
            "mean_latency_ms": (float(1000.0 * np.mean(latencies))
                                if latencies else float("nan")),
        }

        report = summarize(
            mode=cfg.runtime.mode, method=self.planner.name, period=self.period,
            horizon=self.horizon, wall=wall, latencies=latencies,
            k_counter=k_counter, misses=misses, warmup=warmup_boundaries,
            max_overrun=max_overrun, segments=segments,
            viol_log=viol_log if scene.obstacle is not None else None)
        if interactive:
            replay_path = self._prompt_save_replay(
                replay_qpos, replay_qvel, replay_controls, replay_ee, replay_goals)
            if replay_path is not None:
                report += f"\nreplay file: {replay_path}"
        return report


def summarize(*, mode, method, period, horizon, wall, latencies, k_counter,
              misses, warmup, max_overrun, segments, viol_log=None) -> str:
    lines = ["", "=" * 72, f"REAL-TIME VERDICT  ({method}, mode={mode})", "=" * 72]
    lat = np.asarray(latencies) * 1000.0
    if len(lat):
        rate = len(lat) / wall
        lines.append(
            f"plan latency: mean {lat.mean():.1f} ms | p50 {np.percentile(lat, 50):.1f}"
            f" | p95 {np.percentile(lat, 95):.1f} | max {lat.max():.1f}"
            f"   ({len(lat)} plans, {rate:.1f} Hz replanning)")
        lines.append(
            f"control period: {period * 1000:.0f} ms ({1 / period:.0f} Hz);"
            f" horizon {horizon} steps = {horizon * period:.2f} s lookahead")
    if mode == "async":
        applied = sum(k_counter.values())
        hist = ", ".join(f"k={k}: {n} ({100 * n / max(applied, 1):.0f}%)"
                         for k, n in sorted(k_counter.items()))
        lines.append(f"action staleness (applied U[k]): {hist or 'none'}")
        lines.append(
            f"expired plans (k>=horizon -> zero velocity): {misses};"
            f" warm-up boundaries before the first plan: {warmup}")
        lines.append(f"worst real-time pacing overrun: {max_overrun * 1000:.1f} ms")
    if viol_log is not None:
        n_bad = sum(1 for v in viol_log if v)
        lines.append(
            f"whole-arm collision (referee, margin-free): "
            f"{n_bad}/{len(viol_log)} control boundaries with an arm sphere "
            f"inside an obstacle" + ("" if n_bad == 0 else "  <- NOT collision-free"))
    for seg in segments:
        status = (f"strict reach in {seg['t_reach']:.2f} s"
                  if seg["t_reach"] is not None
                  else f"NOT reached (min err {seg['min_err']:.3f} m)")
        lines.append(f"target {seg['target_id']}: {status}")
    all_reached = segments and all(s["t_reach"] is not None for s in segments)
    if mode == "async":
        if all_reached and misses == 0:
            lines.append(
                "verdict: PASS - ran in wall-clock real time and every target "
                f"was reached with the stale-indexed plans (usable at {1 / period:.0f} Hz).")
        elif all_reached:
            lines.append(
                "verdict: MARGINAL - targets reached, but some plans expired before use.")
        else:
            lines.append(
                "verdict: FAIL - the planner could not keep up with real time.")
    elif len(lat):
        lines.append(
            f"serial replanning rate if deployed lockstep: {1000.0 / lat.mean():.1f} Hz"
            f" (real-time factor {period * 1000 / lat.mean():.2f}x vs the"
            f" {period * 1000:.0f} ms control period)")
    lines.append("=" * 72)
    return "\n".join(lines)
