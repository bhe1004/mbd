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
from dataclasses import dataclass
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
from envs.franka import NUM_JOINTS, SCENE_XML_PATH, FrankaTask  # noqa: E402

METHOD_COLORS = {
    "vanilla_mbd_true": (0.13, 0.65, 0.22),
    "dk_mbd": (0.12, 0.47, 0.71),
    "dk_mbd_split": (1.00, 0.50, 0.05),
    "bk_mbd": (0.84, 0.15, 0.16),
}


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
        default="plain",
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
    parser.add_argument("--num-samples", type=int, default=800)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--sigma-start", type=float, default=1.2)
    parser.add_argument("--sigma-end", type=float, default=0.3)
    parser.add_argument("--alpha", type=float, default=0.4)
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
):
    """Return (make_evaluate, predict_ee) for one planner backend.

    `make_evaluate(x, goal, u_prev)` builds the candidate-scoring closure the
    MBD optimizer calls (same math as the offline closed-loop runs);
    `predict_ee(x, U)` returns the planner's own predicted EE path for the
    final control sequence (viewer overlay only).
    """

    if method == MethodName.VANILLA_MBD_TRUE.value:

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
    optimizer: MBDOptimizer,
    make_evaluate: Callable,
    predict_ee: Callable,
    horizon: int,
    period: float,
    rng: np.random.Generator,
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
        t0 = time.perf_counter()
        result = optimizer.optimize(U, evaluate, rng=rng)
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
def draw_overlay(viewer, target, threshold, trail, pred, color) -> None:
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
    task = FrankaTask()
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

    make_evaluate, predict_ee = build_backend(
        task, args.method, model, device, tube_constants, args.beta_e, args.tube_mode
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
            args=(shared, optimizer, make_evaluate, predict_ee, horizon, period, rng),
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
                result = optimizer.optimize(U_lock, evaluate, rng=rng)
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
                draw_overlay(viewer, goal, strict, trail, ee_pred, color)

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
