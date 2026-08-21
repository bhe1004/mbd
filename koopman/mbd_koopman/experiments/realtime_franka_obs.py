"""Live real-time MBD planning on the FR3 in MuJoCo (plan while moving).

`view_franka.py` replays trajectories that were planned offline; this script
runs the planner *during* the motion, against the wall clock, to answer one
question: is ~64 ms per plan (~15 Hz replanning) actually usable as a
real-time planner?

Modes:

- ``async`` (default): the simulation advances in wall-clock real time with
  the 50 ms control period regardless of the planner. A background thread
  replans from the newest state snapshot as fast as it can; every control
  boundary applies the newest finished plan, indexed by the plan's age
  (``u = U[k]``, ``k = floor((t_boundary - t_snapshot) / dt)``). Slow
  planning shows up as stale actions (k >= 1) or, if planning is far too
  slow, as plans that expire before they are applied (miss -> zero
  velocity). This mirrors deployment on the real arm.
- ``lockstep``: the offline protocol (plan -> apply -> step, serialized)
  with live timing; the viewer freezes while each plan is computed.
  Reports the serial replanning rate 1 / plan-time (the 64 ms claim).

The verdict block at the end summarizes plan latency, replanning rate,
action staleness, and reach success per target.

Examples:

    python experiments/realtime_franka.py --method bk_mbd --target-id 0
    python experiments/realtime_franka.py --method bk_mbd --targets 0 2 5 4 --max-time 40
    python experiments/realtime_franka.py --method vanilla_mbd_true --max-time 20
        (the oracle needs seconds per plan: in async mode the arm barely moves)
    python experiments/realtime_franka.py --mode lockstep --no-viewer --max-time 10
        (headless latency benchmark, mirrors run_franka.py numbers)
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, List, Optional

import mujoco
import mujoco.viewer
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import MBDConfig, MethodName, UpdateRule  # noqa: E402
from bk_mbd.mbd import MBDOptimizer  # noqa: E402
from bk_mbd.train import load_checkpoint  # noqa: E402
from bk_mbd.tube import (  # noqa: E402
    bilinear_norm_bounds,
    compute_one_step_residuals,
    cost_sensitivity_torch,
    fit_tube_constants,
    propagate_tube_batch_torch,
)
from envs.franka import (  # noqa: E402
    NUM_JOINTS,
    SCENE_XML_PATH,
    FrankaTask,
    FrankaTaskConfig,
)

METHOD_COLORS = {
    "vanilla_mbd_true": (0.13, 0.65, 0.22),
    "dk_mbd": (0.12, 0.47, 0.71),
    "dk_mbd_split": (1.00, 0.50, 0.05),
    "bk_mbd": (0.84, 0.15, 0.16),
}

# ======================================================================
# TUNE HERE: the keep-out obstacles, spheres and/or axis-aligned boxes.
# Each entry is one of:
#     ((x, y, z), radius)                     # sphere (shorthand)
#     ("sphere", (x, y, z), radius)           # sphere (explicit)
#     ("box",    (x, y, z), (hx, hy, hz))     # box: center + half-extents
# Add, remove, move, or resize entries freely; every obstacle is charged to
# the whole-arm sphere cloud by the planner and scored margin-free by the
# referee. A sphere center may be the string "auto": that ball is placed 40%
# of the way from the home tool position to the FIRST target (on the straight
# line, biased toward home: at the midpoint it ends up so close to the target
# that reaching would need the wrist and gripper to enter it, and the penalty
# balances the goal cost ~50 mm short). Empty list = no obstacles.
# ======================================================================
OBSTACLES = [
    # ((0.613, 0.14, 0.453), 0.05),
    # ((0.613, 0.14, 0.55), 0.05),
    # ((0.613, 0.14, 0.35), 0.05),
    # ((0.613, 0.14, 0.65), 0.05),
    # ((0.613, 0.14, 0.25), 0.05),
    ("box", (0.6, 0.14, 0.453), (0.1, 0.01, 0.16)),
    ("box", (0.7, -0.05, 0.453), (0.01, 0.16, 0.16)),
    ("box", (0.6, -0.2, 0.453), (0.1, 0.01, 0.16)),
]

# ======================================================================
# ADAPTIVE NOISE (flip in code, no CLI flag): shrink the per-plan sigma
# schedule as the tool nears the goal. MBD already anneals sigma high->low
# WITHIN each plan (explore -> refine); this adds a SECOND, across-plans
# shrink so that once the tool is close, the same optimizer stops re-blasting
# its converged plan with the full sigma_start every replan (which otherwise
# leaves the tool jittering tens of mm short of the goal). Set ADAPT_NOISE
# = False to use the optimizer verbatim (the paper-protocol fixed schedule).
# ======================================================================
ADAPT_NOISE = False
ADAPT_ERR_FULL = 0.40   # metres: at/above this reach error, the schedule is
                        # left at full scale (maximum exploration)
ADAPT_FLOOR = 0.05      # smallest scale factor applied right at the goal

# ======================================================================
# ROBOT SPEED: the joint-velocity limit [rad/s] the planner may command and
# the arm executes. Lower = slower motion. The task default is 1.5; set None
# to use it unchanged. Same control period, so a lower cap simply reduces how
# far each joint can move per step.
# ======================================================================
MAX_JOINT_VELOCITY = 1.0

# ======================================================================
# WHOLE-ARM COLLISION BODY: how densely the arm is covered by check spheres.
# Fewer spheres = faster planning (the penalty is evaluated per sphere).
#   ARM_FIRST_LINK: which frame the cover STARTS at (higher = drop the
#     body/proximal links, keep the reaching arm). 3=upper arm+elbow onward,
#     4=elbow onward, 5=forearm/wrist onward (drops the bulky upper arm).
#   ARM_LINK_SAMPLES: extra spheres interpolated per covered segment. Keep
#     this to keep the reaching arm dense.
#   ARM_GRIPPER_FINGERS: True = 4 gripper spheres (crossbar + finger prongs),
#     False = 2 (crossbar only).
# ======================================================================
ARM_FIRST_LINK = 4
ARM_LINK_SAMPLES = 3
ARM_GRIPPER_FINGERS = True


class AdaptiveNoise:
    """Wraps an MBDOptimizer and, when enabled, scales BOTH ends of its noise
    schedule by ``clip(err / err_full, floor, 1)`` per plan.

    Far from the goal the scale is 1 and the optimizer runs its full schedule;
    near the goal the schedule shrinks proportionally, turning the same
    sampler into a fine-positioning loop. Scaled optimizers are cached by
    (rounded) scale so a new one is not built every plan.
    """

    def __init__(self, optimizer: MBDOptimizer, base_config: MBDConfig,
                 enabled: bool, err_full: float, floor: float) -> None:
        self.optimizer = optimizer
        self.base_config = base_config
        self.enabled = bool(enabled)
        self.err_full = float(err_full)
        self.floor = float(floor)
        self._cache: dict = {}

    def optimize(self, U, evaluate, rng, err: float):
        if not self.enabled:
            return self.optimizer.optimize(U, evaluate, rng=rng)
        scale = min(1.0, max(err / self.err_full, self.floor))
        key = round(scale, 2)
        opt = self._cache.get(key)
        if opt is None:
            cfg = replace(self.base_config,
                          sigma_start=self.base_config.sigma_start * scale,
                          sigma_end=self.base_config.sigma_end * scale)
            opt = MBDOptimizer(cfg, self.optimizer.action_low,
                               self.optimizer.action_high)
            self._cache[key] = opt
        return opt.optimize(U, evaluate, rng=rng)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=[item.value for item in MethodName],
        default=MethodName.BK_MBD.value,
    )
    parser.add_argument("--mode", choices=["async", "lockstep"], default="async")
    parser.add_argument("--target-id", type=int, default=0, help="target index 0..6")
    parser.add_argument(
        "--targets",
        type=int,
        nargs="+",
        default=None,
        help="sequence of targets; the planner retargets live after each reach "
        "(overrides --target-id)",
    )
    parser.add_argument(
        "--cycle",
        action="store_true",
        help="loop over the target sequence until --max-time",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Koopman checkpoint; defaults to out/franka/models/{dk,bk}_seed{seed}.pt",
    )
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--beta-e", type=float, default=0.0002)
    parser.add_argument(
        "--tube-mode",
        choices=["none", "plain", "cost-sens"],
        default="none",
        help="bk_mbd tube penalty: 'none' = no tube at all (skips fitting and "
        "propagation entirely - the beta_e=0 ablation without the tube's "
        "compute); 'plain' = beta_e * sum_t e_t (raw lifted error, bk-mppi "
        "style); 'cost-sens' = beta_e * sum_t ||grad c(b_t)|| e_t "
        "(first-order bound on the candidate cost error - model error is "
        "only charged where it can move the cost / distort the score). NOTE: "
        "the modes live on different scales; 'cost-sens' puts the penalty "
        "in cost units, so re-tune --beta-e (try ~1.0) instead of reusing the "
        "plain default",
    )
    # MBD settings: identical defaults to run_franka.py so latency matches.
    parser.add_argument("--num-samples", type=int, default=512)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--sigma-start", type=float, default=0.5)
    parser.add_argument("--sigma-end", type=float, default=0.1)
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument("--eta", type=float, default=1.0)
    parser.add_argument(
        "--update-rule",
        choices=[item.value for item in UpdateRule],
        default=UpdateRule.SCORE_LANGEVIN.value,
    )
    parser.add_argument("--langevin-noise", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--torch-threads",
        type=int,
        default=4,
        help="pin torch intra-op threads (0 = library default). The default of "
        "10 threads on a loaded desktop causes 10-30x straggler spikes in plan "
        "latency; 4 threads gives flat ~20 ms plans on this 10-core machine",
    )
    parser.add_argument("--max-time", type=float, default=15.0, help="wall-clock limit [s]")
    parser.add_argument(
        "--settle-time",
        type=float,
        default=0.4,
        help="hold time after a strict reach before switching to the next target",
    )
    parser.add_argument(
        "--min-plan-ms",
        type=float,
        default=0.0,
        help="pad every plan to at least this wall time (sleep), to emulate a "
        "slower planner: --min-plan-ms 64 reproduces the paper's 64 ms / "
        "~15 Hz claim against the 20 Hz real-time control loop",
    )
    parser.add_argument(
        "--warmup-plans",
        type=int,
        default=3,
        help="throwaway plans before the clock starts (torch/threadpool warm-up, "
        "as deployment on the real arm would do); excluded from all statistics",
    )
    parser.add_argument(
        "--obs-margin", type=float, default=0.02,
        help="planning margin: the penalty inflates the ball by this much, "
        "the referee never does",
    )
    parser.add_argument("--w-obs", type=float, default=5000.0,
                        help="obstacle penalty weight: flat cost per "
                        "overlapping (arm sphere, ball) pair in the default "
                        "hard mode; per squared metre of penetration when "
                        "--graded-obs is set")
    parser.add_argument("--graded-obs", action="store_true",
                        help="use the depth-squared penalty instead of the "
                        "default binary overlap/no-overlap penalty")
    parser.add_argument(
        "--obs-substeps", type=int, default=2,
        help="extra penalty samples between control instants, so a fast "
        "candidate cannot cross the ball unseen between two steps",
    )
    parser.add_argument("--no-obstacle", action="store_true",
                        help="run the original obstacle-free experiment")
    parser.add_argument("--no-viewer", action="store_true")
    parser.add_argument(
        "--interactive", action="store_true",
        help="drive the target live from the keyboard while the planner tracks "
        "it in real time (async mode + viewer). Arrow keys move x/y, E/Q move z, "
        "[ / ] shrink/grow the step, R resets to the start target. Auto-advance "
        "and the max-time limit are disabled; close the viewer (ESC) to stop.",
    )
    parser.add_argument(
        "--goal-step", type=float, default=0.02,
        help="metres the interactive target moves per key press",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "franka" / "realtime"
    )
    return parser.parse_args()


# --------------------------------------------------------------------- backends
def build_backend(
    task: FrankaTask,
    method: str,
    model,
    device,
    tube_constants,
    beta_e,
    tube_mode: str = "plain",
    fk=None,
    obstacle=None,
    obs_substeps: int = 2,
):
    """Return (make_evaluate, predict_ee) for one planner backend.

    `make_evaluate(x, goal, u_prev)` builds the candidate-scoring closure the
    MBD optimizer calls (same math as the offline closed-loop runs);
    `predict_ee(x, U)` returns the planner's own predicted EE path for the
    final control sequence (viewer overlay only).

    With ``obstacle`` set, every candidate additionally pays the whole-arm
    penetration penalty: the decoded joints of every rollout step are pushed
    through the true batched kinematics (``fk.spheres``) and each of the arm
    spheres is charged against the ball -- per step and at interpolated
    points between steps. No convexification, no linearization: this is the
    sampling planner's native way of eating geometry.
    """

    def arm_penalty(q_path: torch.Tensor, q_now: np.ndarray) -> torch.Tensor:
        """q_path: (N, T, 7) decoded joints -> (N,) summed obstacle penalty."""

        pts = fk.spheres(q_path)                              # (N, T, P, 3)
        pen = obstacle.penalty(pts, fk.radii).sum(dim=1)      # (N,)
        q0 = torch.as_tensor(np.asarray(q_now, dtype=np.float32)[:NUM_JOINTS],
                             device=pts.device)
        p_prev = torch.cat([fk.spheres(q0[None])[None].expand(pts.shape[0], 1, -1, -1),
                            pts[:, :-1]], dim=1)
        for i in range(obs_substeps):
            f = (i + 1.0) / (obs_substeps + 1.0)
            pen = pen + obstacle.penalty(p_prev + f * (pts - p_prev),
                                         fk.radii).sum(dim=1)
        return pen

    if method == MethodName.VANILLA_MBD_TRUE.value:
        if obstacle is not None:
            raise SystemExit(
                "--method vanilla_mbd_true has no obstacle support (its "
                "scoring lives inside task.make_true_evaluate); run bk_mbd, "
                "or pass --no-obstacle")

        def make_evaluate(x, goal, u_prev):
            return task.make_true_evaluate(x, goal, u_prev, device)

        def predict_ee(x, U):
            _, ees = task._batch_rollout(x, U[None])
            ee0 = task.ee_of_q(np.asarray(x)[:NUM_JOINTS])
            return np.concatenate([ee0[None], ees[0]], axis=0)

        return make_evaluate, predict_ee

    if method == MethodName.DK_MBD_SPLIT.value:

        def rollout_base(x, U_t):
            q0 = torch.as_tensor(
                np.asarray(x, dtype=np.float64)[:NUM_JOINTS],
                dtype=torch.float32,
                device=device,
            )
            z0 = model.lift(q0).expand(U_t.shape[0], -1)
            q_hat = model.decode(model.rollout(z0, U_t))
            return torch.cat([q_hat, task.forward_kinematics_torch(q_hat)], dim=-1)

        def make_evaluate(x, goal, u_prev):
            def evaluate(candidates):
                with torch.no_grad():
                    U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                    bs = rollout_base(x, U_t)
                    costs = task.trajectory_cost_base_torch(bs, U_t, goal, u_prev=u_prev)
                    if obstacle is not None:
                        costs = costs + arm_penalty(bs[..., :NUM_JOINTS], x)
                    return costs.cpu().numpy()

            return evaluate

        def predict_ee(x, U):
            with torch.no_grad():
                U_t = torch.as_tensor(U[None], dtype=torch.float32, device=device)
                bs = rollout_base(x, U_t)
                return bs[0, :, NUM_JOINTS : NUM_JOINTS + 3].cpu().numpy()

        return make_evaluate, predict_ee

    # dk_mbd / bk_mbd: lifted rollout (+ tube penalty for bk).
    use_tube = tube_constants is not None
    if use_tube:
        norm_a, norm_bs_np = bilinear_norm_bounds(model.bilinear_params())
        norm_bs = torch.as_tensor(norm_bs_np, dtype=torch.float32, device=device)

    def rollout_candidates(x, U_t):
        b0 = task.state_to_base_torch(x, device)
        z0 = model.lift(b0).expand(U_t.shape[0], -1)
        zs = model.rollout(z0, U_t)
        return zs, model.decode(zs)

    def make_evaluate(x, goal, u_prev):
        def evaluate(candidates):
            with torch.no_grad():
                U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                zs, bs = rollout_candidates(x, U_t)
                costs = task.trajectory_cost_base_torch(bs, U_t, goal, u_prev=u_prev)
                if obstacle is not None:
                    costs = costs + arm_penalty(bs[..., :NUM_JOINTS], x)
                if use_tube:
                    tubes = propagate_tube_batch_torch(
                        zs, U_t, norm_a=norm_a, norm_bs=norm_bs, constants=tube_constants
                    )
                    if tube_mode == "cost-sens":
                        # Weight each e_t by the local cost slope so the
                        # penalty bounds the *cost* error (score fidelity),
                        # not the raw state error.
                        L = cost_sensitivity_torch(
                            bs,
                            lambda b: task.trajectory_cost_base_torch(
                                b, U_t, goal, u_prev=u_prev
                            ),
                        )
                        costs = costs + beta_e * (L * tubes).sum(dim=1)
                    else:
                        costs = costs + beta_e * tubes.sum(dim=1)
                return costs.cpu().numpy()

        return evaluate

    def predict_ee(x, U):
        with torch.no_grad():
            U_t = torch.as_tensor(U[None], dtype=torch.float32, device=device)
            _, bs = rollout_candidates(x, U_t)
            return bs[0, :, NUM_JOINTS : NUM_JOINTS + 3].cpu().numpy()

    return make_evaluate, predict_ee


# ------------------------------------------------------------------ shared state
@dataclass
class Plan:
    U: np.ndarray  # (T, 7) planned control sequence
    t_state: float  # wall time of the state snapshot the plan starts from
    ee_pred: np.ndarray  # predicted EE path for the overlay
    latency: float  # optimize() wall time [s]
    index: int


class SharedState:
    """State snapshot / plan exchange between the control and planner threads."""

    def __init__(self, x0: np.ndarray, goal0: np.ndarray) -> None:
        self.lock = threading.Lock()
        self.x = np.asarray(x0, dtype=np.float64).copy()
        self.x_time = time.perf_counter()
        self.goal = np.asarray(goal0, dtype=np.float64).copy()
        self.u_prev = np.zeros(NUM_JOINTS)
        self.plan: Optional[Plan] = None
        self.stop = False
        self.latencies: List[float] = []


def planner_loop(
    shared: SharedState,
    optimizer: "AdaptiveNoise",
    make_evaluate: Callable,
    predict_ee: Callable,
    horizon: int,
    period: float,
    rng: np.random.Generator,
    task: FrankaTask,
    min_plan_s: float = 0.0,
) -> None:
    """Continuously replan from the newest snapshot (async mode).

    The warm start shifts the previous solution by the number of control
    periods that elapsed between the two snapshots (the receding-horizon
    analogue of the offline roll-by-one).
    """

    U = np.zeros((horizon, NUM_JOINTS), dtype=np.float64)
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
            shift = int(np.clip(round((t_state - prev_t_state) / period), 0, horizon))
            if shift >= horizon:
                U = np.zeros_like(U)
            elif shift > 0:
                U = np.roll(U, -shift, axis=0)
                U[horizon - shift :] = U[horizon - shift - 1]

        evaluate = make_evaluate(x, goal, u_prev)
        err = float(np.linalg.norm(task.ee_of_q(x[:NUM_JOINTS]) - goal))
        t0 = time.perf_counter()
        result = optimizer.optimize(U, evaluate, rng, err)
        latency = time.perf_counter() - t0
        if latency < min_plan_s:
            time.sleep(min_plan_s - latency)
            latency = time.perf_counter() - t0
        U = result.controls
        ee_pred = predict_ee(x, U)

        index += 1
        plan = Plan(U.copy(), t_state, ee_pred, latency, index)
        with shared.lock:
            shared.plan = plan
            shared.latencies.append(latency)
        prev_t_state = t_state


# --------------------------------------------------------------------- overlay
def draw_overlay(viewer, target, threshold, trail, pred, color,
                 obstacle=None) -> None:
    """Target sphere + reach shell + actual TCP trail + planned EE path."""

    scn = viewer.user_scn
    scn.ngeom = 0

    def add_sphere(pos, size, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0],
            pos=np.asarray(pos, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    def add_box(pos, half, rgba):
        if scn.ngeom >= scn.maxgeom:
            return
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=np.asarray(half, dtype=np.float64),
            pos=np.asarray(pos, dtype=np.float64),
            mat=np.eye(3).flatten(),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    def add_path(points, radius, rgba, budget):
        pts = np.asarray(points)
        if len(pts) < 2:
            return
        stride = max(1, int(np.ceil((len(pts) - 1) / max(budget, 1))))
        prev = pts[0]
        for j in range(stride, len(pts), stride):
            if scn.ngeom >= scn.maxgeom:
                return
            geom = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.zeros(3),
                pos=np.zeros(3),
                mat=np.eye(3).flatten(),
                rgba=np.asarray(rgba, dtype=np.float32),
            )
            mujoco.mjv_connector(
                geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, prev, pts[j]
            )
            scn.ngeom += 1
            prev = pts[j]

    if obstacle is not None:
        for c, r in obstacle.spheres_draw:
            add_sphere(c, r, [0.8, 0.25, 0.2, 0.45])
        for c, h in obstacle.boxes_draw:
            add_box(c, h, [0.8, 0.25, 0.2, 0.45])
    add_sphere(target, 0.012, [0.1, 0.8, 0.1, 1.0])
    add_sphere(target, threshold, [0.1, 0.8, 0.1, 0.15])
    if pred is not None:
        add_path(pred, 0.0015, [*color, 0.4], budget=len(pred))
    if trail:
        add_path(trail, 0.002, [*color, 0.9], budget=scn.maxgeom - scn.ngeom - 2)


# --------------------------------------------------------------------- reporting
def summarize(
    *,
    mode: str,
    method: str,
    period: float,
    horizon: int,
    wall: float,
    latencies: List[float],
    k_counter: Counter,
    misses: int,
    warmup: int,
    max_overrun: float,
    segments: List[dict],
    viol_log: Optional[List[int]] = None,
) -> str:
    lines = ["", "=" * 72, f"REAL-TIME VERDICT  ({method}, mode={mode})", "=" * 72]
    lat = np.asarray(latencies) * 1000.0
    if len(lat):
        rate = len(lat) / wall
        lines.append(
            f"plan latency: mean {lat.mean():.1f} ms | p50 {np.percentile(lat, 50):.1f}"
            f" | p95 {np.percentile(lat, 95):.1f} | max {lat.max():.1f}"
            f"   ({len(lat)} plans, {rate:.1f} Hz replanning)"
        )
        lines.append(
            f"control period: {period * 1000:.0f} ms ({1 / period:.0f} Hz);"
            f" horizon {horizon} steps = {horizon * period:.2f} s lookahead"
        )
    if mode == "async":
        applied = sum(k_counter.values())
        hist = ", ".join(
            f"k={k}: {n} ({100 * n / max(applied, 1):.0f}%)"
            for k, n in sorted(k_counter.items())
        )
        lines.append(f"action staleness (applied U[k]): {hist or 'none'}")
        lines.append(
            f"expired plans (k>=horizon -> zero velocity): {misses};"
            f" warm-up boundaries before the first plan: {warmup}"
        )
        lines.append(f"worst real-time pacing overrun: {max_overrun * 1000:.1f} ms")
    if viol_log is not None:
        n_bad = sum(1 for v in viol_log if v)
        lines.append(
            f"whole-arm collision (referee, margin-free): "
            f"{n_bad}/{len(viol_log)} control boundaries with an arm sphere "
            f"inside the ball" + ("" if n_bad == 0 else "  <- NOT collision-free")
        )
    for seg in segments:
        status = (
            f"strict reach in {seg['t_reach']:.2f} s"
            if seg["t_reach"] is not None
            else f"NOT reached (min err {seg['min_err']:.3f} m)"
        )
        lines.append(f"target {seg['target_id']}: {status}")
    all_reached = segments and all(s["t_reach"] is not None for s in segments)
    if mode == "async":
        if all_reached and misses == 0:
            lines.append(
                "verdict: PASS - the simulation ran in wall-clock real time and "
                "every target was reached with the stale-indexed plans; the "
                f"planner is usable as a real-time planner at {1 / period:.0f} Hz "
                "control with the measured replanning rate."
            )
        elif all_reached:
            lines.append(
                "verdict: MARGINAL - targets reached, but some plans expired "
                "before use; the planner only barely keeps up."
            )
        else:
            lines.append(
                "verdict: FAIL - the planner could not keep up with real time "
                "(targets missed under wall-clock execution)."
            )
    else:
        if len(lat):
            lines.append(
                f"serial replanning rate if deployed lockstep: {1000.0 / lat.mean():.1f} Hz"
                f" (real-time factor {period * 1000 / lat.mean():.2f}x vs the"
                f" {period * 1000:.0f} ms control period)"
            )
    lines.append("=" * 72)
    return "\n".join(lines)


# ------------------------------------------------------------------------ main
def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    setup_t0 = time.perf_counter()
    task = FrankaTask(
        None if MAX_JOINT_VELOCITY is None
        else FrankaTaskConfig(action_limit=float(MAX_JOINT_VELOCITY)))
    print(f"robot speed: joint-velocity limit {task.config.action_limit} rad/s"
          + ("" if MAX_JOINT_VELOCITY is not None else " (task default)"))
    period = task.config.control_dt
    horizon = task.config.horizon

    mbd_config = MBDConfig(
        num_samples=args.num_samples,
        num_diffusion_steps=args.num_diffusion_steps,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        alpha=args.alpha,
        eta=args.eta,
        update_rule=UpdateRule(args.update_rule),
        seed=args.seed,
        add_langevin_noise=args.langevin_noise,
    )
    optimizer = MBDOptimizer(
        mbd_config, action_low=task.action_bounds[0], action_high=task.action_bounds[1]
    )
    adaptive = AdaptiveNoise(optimizer, mbd_config, ADAPT_NOISE,
                             ADAPT_ERR_FULL, ADAPT_FLOOR)
    print(f"adaptive noise: {'ON' if ADAPT_NOISE else 'OFF'}"
          + (f" (full schedule at err >= {ADAPT_ERR_FULL} m, floor "
             f"{ADAPT_FLOOR})" if ADAPT_NOISE else ""))

    # Koopman checkpoint + tube constants, exactly as run_franka.py.
    model = None
    tube_constants = None
    koopman_methods = {
        MethodName.DK_MBD.value: "dk",
        MethodName.DK_MBD_SPLIT.value: "dk_split",
        MethodName.BK_MBD.value: "bk",
    }
    if args.method in koopman_methods:
        short = koopman_methods[args.method]
        checkpoint = args.checkpoint or (
            PROJECT_ROOT / "out" / "franka" / "models" / f"{short}_seed{args.seed}.pt"
        )
        if not checkpoint.exists():
            raise SystemExit(
                f"checkpoint not found: {checkpoint}\n"
                "train it first: python experiments/train_franka_koopman.py"
            )
        model, _ = load_checkpoint(checkpoint, device=device)
        model.eval()
        print(f"loaded={checkpoint}")
        if args.method == MethodName.BK_MBD.value and args.tube_mode != "none":
            print("fitting tube constants (one-step residuals on the training set)...")
            dataset = task.sample_dataset(args.data_seed)
            zs, us, residuals = compute_one_step_residuals(
                model, dataset["base_states"], dataset["controls"], device=device
            )
            tube_constants = fit_tube_constants(zs, us, residuals, quantile=0.999)
            print(
                f"tube constants: c_x={tube_constants.c_x:.4e} "
                f"c_u={tube_constants.c_u:.4e} beta_e={args.beta_e} "
                f"tube_mode={args.tube_mode}"
            )
        elif args.method == MethodName.BK_MBD.value:
            print("tube disabled (--tube-mode none): plain bilinear rollout, no penalty")

    # ---- the keep-out obstacles + the whole-arm body that must avoid them --
    fk = None
    obstacle = None
    if not args.no_obstacle and OBSTACLES:
        from experiments.arm_collision import ArmFK, ObstacleField

        first_goal = task.targets[(args.targets or [args.target_id])[0]]
        home_ee = task.ee_of_q(task.home_qpos)
        auto = home_ee + 0.4 * (np.asarray(first_goal) - home_ee)
        print(f"auto obstacle center = {np.round(auto, 3)}")

        def parse_entry(e):
            # ((x,y,z), r) sphere shorthand, or (kind, center, size) tagged.
            kind, c, size = ("sphere", e[0], e[1]) if len(e) == 2 else e
            center = auto if (isinstance(c, str) and c == "auto") \
                else np.asarray(c, dtype=np.float64)
            return kind, center, size

        entries = [parse_entry(e) for e in OBSTACLES]
        fk = ArmFK(task, device=device, first_link=ARM_FIRST_LINK,
                   link_samples=ARM_LINK_SAMPLES,
                   hand_fingers=ARM_GRIPPER_FINGERS)
        obstacle = ObstacleField(entries, margin=args.obs_margin,
                                 weight=args.w_obs, device=device,
                                 hard=not args.graded_obs)
        mode = "graded (depth^2)" if args.graded_obs else "HARD (flat per overlap)"
        print(f"penalty: {mode}, weight {args.w_obs}, margin {args.obs_margin}")
        for kind, c, size in entries:
            print(f"obstacle: {kind} at {np.round(np.asarray(c), 3)} "
                  f"{'radius' if kind == 'sphere' else 'half'} "
                  f"{np.round(np.asarray(size), 3)}")
        d0 = obstacle.clearance(fk.spheres_np(task.home_qpos), fk.radii_np)
        print(f"whole arm = {fk.num_points} spheres (chain + gripper T) | "
              f"closest arm sphere at start {d0 * 1000:.0f} mm clear")
        if d0 <= 0:
            raise SystemExit("the arm already penetrates an obstacle at the "
                             "home pose -- edit OBSTACLES")

    make_evaluate, predict_ee = build_backend(
        task, args.method, model, device, tube_constants, args.beta_e,
        args.tube_mode, fk=fk, obstacle=obstacle,
        obs_substeps=args.obs_substeps,
    )

    # Simulation model: the visualization scene *includes* the task model, so
    # it has the same joints, velocity actuators, options, and tcp site; the
    # planner never touches this MjData (thread ownership: sim = main thread,
    # task's internal MjData = planner thread).
    scene_model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
    if scene_model.nq != NUM_JOINTS or scene_model.nu != NUM_JOINTS:
        raise SystemExit("scene model does not match the 7-DoF task model")
    nsub = int(round(period / scene_model.opt.timestep))
    sub_dt = period / nsub
    sim = mujoco.MjData(scene_model)
    mujoco.mj_resetDataKeyframe(scene_model, sim, scene_model.key("home").id)
    sim.qvel[:] = 0.0
    mujoco.mj_forward(scene_model, sim)
    tcp = scene_model.site("tcp").id

    target_ids = args.targets if args.targets else [args.target_id]
    for t_id in target_ids:
        if not 0 <= t_id < len(task.targets):
            raise SystemExit(f"target id {t_id} out of range 0..{len(task.targets) - 1}")
    strict = task.config.strict_threshold
    color = METHOD_COLORS.get(args.method, (0.6, 0.6, 0.6))
    print(
        f"setup done in {time.perf_counter() - setup_t0:.1f} s | mode={args.mode} "
        f"method={args.method} targets={target_ids} control={period * 1000:.0f} ms "
        f"({nsub} substeps of {sub_dt * 1000:.1f} ms)"
    )

    x0 = np.concatenate([sim.qpos[:NUM_JOINTS], sim.qvel[:NUM_JOINTS]])
    shared = SharedState(x0, task.targets[target_ids[0]])
    rng = np.random.default_rng(args.seed)

    planner = None
    if args.mode == "async":
        planner = threading.Thread(
            target=planner_loop,
            args=(shared, adaptive, make_evaluate, predict_ee, horizon,
                  period, rng, task),
            kwargs={"min_plan_s": args.min_plan_ms / 1000.0},
            daemon=True,
        )

    # ---- interactive target: a keyboard-driven goal the planner chases live ---
    # The passive viewer forwards key presses to key_callback (GLFW keycodes). We
    # keep the goal in a 1-element dict so the closure can mutate it in place; the
    # control loop reads it each boundary and pushes it into shared.goal.
    GOAL_LOW = np.array([0.25, -0.45, 0.15])
    GOAL_HIGH = np.array([0.75, 0.45, 0.95])
    interactive = args.interactive
    if interactive:
        if args.no_viewer:
            raise SystemExit("--interactive needs the viewer (drop --no-viewer)")
        if args.mode != "async":
            raise SystemExit("--interactive requires --mode async (the default)")
    ig = {"p": task.targets[target_ids[0]].astype(np.float64).copy(),
          "step": float(args.goal_step)}

    def key_callback(keycode: int) -> None:
        p, s = ig["p"], ig["step"]
        if keycode == 265:      # Up arrow  -> +x (away from base)
            p[0] += s
        elif keycode == 264:    # Down arrow -> -x
            p[0] -= s
        elif keycode == 263:    # Left arrow -> +y
            p[1] += s
        elif keycode == 262:    # Right arrow -> -y
            p[1] -= s
        elif keycode in (69, 334):   # 'E' or keypad '+' -> +z
            p[2] += s
        elif keycode in (81, 333):   # 'Q' or keypad '-' -> -z
            p[2] -= s
        elif keycode == 93:     # ']' -> larger step
            ig["step"] = min(0.1, s * 1.5)
        elif keycode == 91:     # '[' -> smaller step
            ig["step"] = max(0.002, s / 1.5)
        elif keycode == 82:     # 'R' -> reset to the start target
            ig["p"][:] = task.targets[target_ids[0]]
        np.clip(ig["p"], GOAL_LOW, GOAL_HIGH, out=ig["p"])

    viewer_ctx = (
        None
        if args.no_viewer
        else mujoco.viewer.launch_passive(
            scene_model, sim, key_callback=key_callback if interactive else None)
    )
    viewer = viewer_ctx.__enter__() if viewer_ctx else None
    if interactive:
        print(
            "\nINTERACTIVE TARGET — move it while the arm tracks in real time:\n"
            "  arrows: Up/Down = +x/-x (forward/back),  Left/Right = +y/-y\n"
            "  E / Q : up / down (z)     [ / ] : smaller / larger step\n"
            "  R     : reset target      ESC (close viewer): quit\n"
            f"  step = {ig['step']*1000:.0f} mm/press\n"
        )

    # Logs shared by both modes.
    states = [x0.copy()]
    controls: List[np.ndarray] = []
    errors: List[float] = []
    k_log: List[int] = []
    age_log: List[float] = []
    k_counter: Counter = Counter()
    misses = 0
    warmup = 0
    max_overrun = 0.0
    lockstep_latencies: List[float] = []
    segments: List[dict] = []
    trail: List[np.ndarray] = []
    viol_log: List[int] = []   # per-boundary arm spheres inside the ball

    seg_idx = 0
    goal = task.targets[target_ids[seg_idx]].copy()
    seg = {"target_id": target_ids[seg_idx], "t_start": 0.0, "t_reach": None, "min_err": np.inf}
    done = False
    U_lock = np.zeros((horizon, NUM_JOINTS), dtype=np.float64)
    u_prev_lock = np.zeros(NUM_JOINTS)
    last_status = 0.0
    sync_every = max(1, nsub // 5)

    def viewer_alive() -> bool:
        return viewer is None or viewer.is_running()

    def advance_segment(wall: float) -> bool:
        """Close the current segment; return True when the run is finished."""
        nonlocal seg_idx, goal, seg
        segments.append(seg)
        seg_idx += 1
        if seg_idx >= len(target_ids):
            if not args.cycle:
                return True
            seg_idx = 0
        goal = task.targets[target_ids[seg_idx]].copy()
        seg = {
            "target_id": target_ids[seg_idx],
            "t_start": wall,
            "t_reach": None,
            "min_err": np.inf,
        }
        print(f"[{wall:6.2f}s] next target -> {target_ids[seg_idx]} {np.round(goal, 3)}")
        return False

    if args.warmup_plans > 0:
        warm_rng = np.random.default_rng(10**6 + args.seed)
        warm_eval = make_evaluate(x0, goal, np.zeros(NUM_JOINTS))
        t_warm = time.perf_counter()
        for _ in range(args.warmup_plans):
            optimizer.optimize(
                np.zeros((horizon, NUM_JOINTS), dtype=np.float64), warm_eval, rng=warm_rng
            )
        print(
            f"warm-up: {args.warmup_plans} throwaway plans in "
            f"{time.perf_counter() - t_warm:.2f} s"
        )

    if planner:
        planner.start()
    t0_wall = time.perf_counter()
    boundary = 0
    try:
        while viewer_alive() and not done:
            now = time.perf_counter()
            wall = now - t0_wall
            # Segment timing: wall time IS sim time in async (hard pacing);
            # in lockstep the sim is decoupled from the wall clock, so report
            # reach times in sim time (boundaries x control period).
            t_seg = wall if args.mode == "async" else boundary * period
            if wall >= args.max_time and not interactive:
                break

            # In interactive mode the goal is whatever the keyboard last set.
            if interactive:
                goal = ig["p"].copy()

            # ---- control boundary: measure, (plan,) apply -----------------
            x = np.concatenate([sim.qpos[:NUM_JOINTS], sim.qvel[:NUM_JOINTS]])
            ee = sim.site_xpos[tcp].copy()
            err = float(np.linalg.norm(ee - goal))
            seg["min_err"] = min(seg["min_err"], err)
            errors.append(err)
            if obstacle is not None:
                nv = obstacle.violations(fk.spheres_np(x[:NUM_JOINTS]),
                                         fk.radii_np)
                viol_log.append(nv)
                if nv and (not viol_log[-2:-1] or not viol_log[-2]):
                    print(f"[{wall:6.2f}s] COLLISION: {nv} arm sphere(s) "
                          f"inside a ball")
            trail.append(ee)
            if interactive and len(trail) > 300:   # keep only a recent trail
                del trail[0]

            if seg["t_reach"] is None and err < strict:
                seg["t_reach"] = t_seg - seg["t_start"]
                print(
                    f"[{t_seg:6.2f}s] target {seg['target_id']} strict reach "
                    f"({err * 1000:.1f} mm) in {seg['t_reach']:.2f} s"
                )
            # Auto-advance through the preset targets only in scripted mode; the
            # interactive target never "completes", it just keeps tracking.
            if not interactive and seg["t_reach"] is not None and (
                t_seg - seg["t_start"] - seg["t_reach"] >= args.settle_time
            ):
                done = advance_segment(t_seg)
                if done:
                    break

            ee_pred = None
            if args.mode == "async":
                with shared.lock:
                    shared.x = x
                    shared.x_time = now
                    shared.goal = goal.copy()
                    plan = shared.plan
                if plan is None:
                    u = np.zeros(NUM_JOINTS)
                    warmup += 1
                else:
                    k = int((now - plan.t_state) / period)
                    age_log.append((now - plan.t_state) * 1000.0)
                    if k >= horizon:
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
                evaluate = make_evaluate(x, goal, u_prev_lock)
                t_plan = time.perf_counter()
                result = adaptive.optimize(U_lock, evaluate, rng, err)
                if time.perf_counter() - t_plan < args.min_plan_ms / 1000.0:
                    time.sleep(args.min_plan_ms / 1000.0 - (time.perf_counter() - t_plan))
                lockstep_latencies.append(time.perf_counter() - t_plan)
                U_lock = result.controls
                u = U_lock[0]
                ee_pred = predict_ee(x, U_lock)
                U_lock = np.roll(U_lock, -1, axis=0)
                U_lock[-1] = U_lock[-2]
                u_prev_lock = u.copy()

            u = np.clip(u, task.action_bounds[0], task.action_bounds[1])
            sim.ctrl[:] = u
            states.append(x.copy())
            controls.append(u.copy())

            if viewer:
                draw_overlay(viewer, goal, strict, trail, ee_pred, color,
                             obstacle=obstacle)

            # ---- advance the plant one control period ---------------------
            # async: paced against the global wall clock (hard real time);
            # lockstep: paced locally so the motion segment plays at 1x.
            base = t0_wall if args.mode == "async" else time.perf_counter()
            for i in range(nsub):
                mujoco.mj_step(scene_model, sim)
                if args.mode == "async":
                    deadline = t0_wall + (boundary * nsub + i + 1) * sub_dt
                else:
                    deadline = base + (i + 1) * sub_dt
                if viewer and (i % sync_every == sync_every - 1):
                    viewer.sync()
                slack = deadline - time.perf_counter()
                if slack > 0:
                    if viewer or args.mode == "async":
                        time.sleep(slack)
                else:
                    max_overrun = max(max_overrun, -slack)
            boundary += 1

            if wall - last_status >= 1.0:
                last_status = wall
                with shared.lock:
                    n_plans = len(shared.latencies)
                    lat = list(shared.latencies)
                lat = lockstep_latencies if args.mode == "lockstep" else lat
                lat_ms = 1000 * np.mean(lat[-20:]) if lat else float("nan")
                n_plans = len(lat)
                print(
                    f"[{wall:6.2f}s] target {seg['target_id']} err={err:.3f} m | "
                    f"{n_plans} plans, recent {lat_ms:.0f} ms/plan "
                    f"({n_plans / max(wall, 1e-9):.1f} Hz)"
                )
    except KeyboardInterrupt:
        print("\ninterrupted")
    finally:
        with shared.lock:
            shared.stop = True

    wall = time.perf_counter() - t0_wall
    if not done and seg["min_err"] < np.inf:
        segments.append(seg)
    with shared.lock:
        latencies = list(shared.latencies) if args.mode == "async" else lockstep_latencies

    report = summarize(
        mode=args.mode,
        method=args.method,
        period=period,
        horizon=horizon,
        wall=wall,
        latencies=latencies,
        k_counter=k_counter,
        misses=misses,
        warmup=warmup,
        max_overrun=max_overrun,
        segments=segments,
        viol_log=viol_log if obstacle is not None else None,
    )
    print(report)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Tag the tube mode in the filename so none/plain/cost-sens runs don't
    # clobber each other (only bk_mbd actually uses the tube; 'plain' keeps
    # the historical untagged name).
    tube_tag = ""
    if args.method == MethodName.BK_MBD.value:
        tube_tag = {"none": "_notube", "cost-sens": "_costsens"}.get(args.tube_mode, "")
    stem = f"{args.method}{tube_tag}_{args.mode}_seed{args.seed}"
    np.savez(
        args.output_dir / f"{stem}.npz",
        states=np.stack(states),
        controls=np.stack(controls) if controls else np.zeros((0, NUM_JOINTS)),
        errors=np.asarray(errors),
        plan_latencies=np.asarray(latencies),
        applied_k=np.asarray(k_log, dtype=np.int64),
        plan_age_ms=np.asarray(age_log),
        wall_time=wall,
        method=args.method,
        mode=args.mode,
        tube_mode=args.tube_mode,
        obstacle_viol=np.asarray(viol_log, dtype=np.int64),
        obstacle_sphere_centers=(obstacle.s_centers_np if obstacle is not None
                                 else np.zeros((0, 3))),
        obstacle_sphere_radii=(obstacle.s_radii_np if obstacle is not None
                               else np.zeros(0)),
        obstacle_box_centers=(obstacle.b_centers_np if obstacle is not None
                              else np.zeros((0, 3))),
        obstacle_box_halfs=(obstacle.b_halfs_np if obstacle is not None
                            else np.zeros((0, 3))),
    )
    print(f"saved={args.output_dir / (stem + '.npz')}")

    if planner and planner.is_alive():
        planner.join(timeout=10.0)
    if viewer:
        print("holding at zero velocity - close the viewer (ESC) to exit")
        try:
            while viewer.is_running():
                sim.ctrl[:] = 0.0
                mujoco.mj_step(scene_model, sim)
                viewer.sync()
                time.sleep(sub_dt)
        except KeyboardInterrupt:
            pass
    if viewer_ctx:
        viewer_ctx.__exit__(None, None, None)


if __name__ == "__main__":
    main()
