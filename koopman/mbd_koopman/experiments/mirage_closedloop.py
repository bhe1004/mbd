"""Closed-loop mirage experiment: same model error, MPPI-like vs annealed MBD.

Completes the witness left open by `annealing_robustness.py` (which showed the
*update-level* contamination asymmetry but only tested open-loop solves): here
the planner runs in closed loop - plan, execute the first action, the state
moves, replan - so a per-step contamination floor can accumulate through the
plant ("chasing the mirage").

Controlled setting: the plant is the unperturbed BK model (synthetic ground
truth, full lifted-state feedback), the planner rolls a delta-perturbed copy.
At delta = 0 the planner's model is exact, so any degradation is purely the
injected error. All arms share the sample budget (K samples x N rounds per
replan) and differ ONLY in the sigma schedule:

- annealed  (1.2 -> 0.05): bk-mbd
- fixed-mid (0.35 const):  tube-less fixed-sigma sampler (MPPI-like; a real
  single-round MPPI would be strictly weaker than this N-round variant)
- fixed-low (0.05 const):  refinement-only extreme (no exploration)

Metric: steady-state end-effector error (mean over the last steps), where a
contaminated per-step commit shows up as jitter/offset the plant cannot shed.

Example:

    python experiments/mirage_closedloop.py            # full run
    python experiments/mirage_closedloop.py --quick    # smoke test
"""

from __future__ import annotations

import argparse
import copy
import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import MBDConfig, UpdateRule  # noqa: E402
from bk_mbd.mbd import MBDOptimizer  # noqa: E402
from bk_mbd.train import load_checkpoint  # noqa: E402
from envs.franka import NUM_JOINTS, FrankaTask  # noqa: E402

ALPHA = 0.4
ARMS = {
    "annealed": (1.2, 0.05),
    "fixed-mid": (0.35, 0.35),
    "fixed-low": (0.05, 0.05),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "out" / "franka" / "models" / "bk_seed0.pt",
    )
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--closed-loop-steps", type=int, default=60)
    parser.add_argument("--steady-window", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "mirage_closedloop"
    )
    return parser.parse_args()


def perturb_model(model, delta: float, pert_seed: int):
    """delta * ||P||_F-scaled random direction on A, B0, Bs (fixed per seed)."""

    if delta == 0.0:
        return model
    pert = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(pert_seed)
    with torch.no_grad():
        for name in ("A", "B0", "Bs"):
            p = getattr(pert, name)
            g = torch.randn(p.shape, generator=gen)
            p.add_(delta * p.norm() * g / g.norm())
    return pert


def closed_loop_run(
    *,
    true_model,
    plan_model,
    task,
    goal,
    b_init: torch.Tensor,
    optimizer: MBDOptimizer,
    horizon: int,
    steps: int,
    rng: np.random.Generator,
    device,
) -> np.ndarray:
    """Plan with plan_model, execute on true_model; return EE error per step.

    The plant lives in base-state space and re-lifts every step
    (b <- decode(step(lift(b), u))), matching the model's training-time usage;
    free-running the lifted state for many steps would drift off the lift
    manifold. The planner re-lifts the current base state at every replan,
    exactly as in deployment.
    """

    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
    action_low, action_high = task.action_bounds
    b = b_init.clone()
    U = np.zeros((horizon, NUM_JOINTS))
    errors = np.zeros(steps)
    for s in range(steps):
        with torch.no_grad():
            z0 = true_model.lift(b).detach()

        @torch.no_grad()
        def evaluate(candidates: np.ndarray) -> np.ndarray:
            U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
            zs = plan_model.rollout(z0.unsqueeze(0).expand(U_t.shape[0], -1), U_t)
            bs = plan_model.decode(zs)
            return task.trajectory_cost_base_torch(bs, U_t, goal_t).cpu().numpy()

        result = optimizer.optimize(U, evaluate, rng=rng)
        U = result.controls
        u = np.clip(U[0], action_low, action_high)
        with torch.no_grad():
            u_t = torch.as_tensor(u, dtype=torch.float32, device=device)
            z = true_model.step(z0.unsqueeze(0), u_t.unsqueeze(0)).squeeze(0)
            b = true_model.decode(z)
            errors[s] = float(
                torch.linalg.vector_norm(b[NUM_JOINTS : NUM_JOINTS + 3] - goal_t)
            )
        U = np.roll(U, -1, axis=0)  # receding-horizon warm start
        U[-1] = U[-2]
    return errors


# ------------------------------------------------------------------ plotting
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = {"annealed": "#2a78d6", "fixed-mid": "#1baf7a", "fixed-low": "#eda100"}
MARKERS = {"annealed": "o", "fixed-mid": "s", "fixed-low": "^"}


def style_axis(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(BASE)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figure(traces, deltas, steady, delta_show, control_dt, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Panel A: EE error vs time at the shown delta (mean across seeds).
    for arm in ARMS:
        runs = np.stack(traces[(arm, delta_show)])  # (n_runs, steps)
        t = np.arange(runs.shape[1]) * control_dt
        mean = runs.mean(axis=0) * 1000.0
        lo = runs.min(axis=0) * 1000.0
        hi = runs.max(axis=0) * 1000.0
        ax1.fill_between(t, lo, hi, color=SERIES[arm], alpha=0.15, lw=0)
        ax1.plot(t, mean, color=SERIES[arm], lw=2, label=arm)
    ax1.set_yscale("log")
    ax1.set_xlabel("time [s]", fontsize=10, color=INK_2)
    ax1.set_ylabel("EE error [mm]", fontsize=10, color=INK_2)
    ax1.set_title(
        f"A    closed-loop tracking, same model error ($\\delta$={delta_show})",
        loc="left", fontsize=10.5, color=INK,
    )
    ax1.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper right")
    style_axis(ax1)

    # Panel B: steady-state error vs delta.
    d_pos = [d for d in deltas if d > 0]
    for arm in ARMS:
        mean = np.array([steady[(arm, d)]["mean"] for d in deltas]) * 1000.0
        se = np.array([steady[(arm, d)]["se"] for d in deltas]) * 1000.0
        x = np.array(deltas)
        x_plot = np.where(x > 0, x, d_pos[0] / 3)  # show delta=0 at left edge
        ax2.fill_between(
            x_plot, mean - se, mean + se, color=SERIES[arm], alpha=0.15, lw=0
        )
        ax2.plot(
            x_plot, mean, color=SERIES[arm], lw=2,
            marker=MARKERS[arm], ms=5.5, label=arm,
        )
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    ticks = [d_pos[0] / 3] + d_pos
    ax2.set_xticks(ticks, ["0"] + [f"{d:g}" for d in d_pos])
    ax2.set_xlabel(
        r"model perturbation $\delta$ (relative)", fontsize=10, color=INK_2
    )
    ax2.set_ylabel("steady-state EE error [mm]", fontsize=10, color=INK_2)
    ax2.set_title(
        "B    steady-state error vs model error", loc="left", fontsize=10.5, color=INK
    )
    ax2.legend(frameon=False, fontsize=9, labelcolor=INK_2, loc="upper left")
    style_axis(ax2)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURFACE, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURFACE, bbox_inches="tight")
    print(f"figure saved: {out_path} (+ .pdf)")


# ---------------------------------------------------------------------- main
def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if args.torch_threads > 0:
        torch.set_num_threads(args.torch_threads)

    task = FrankaTask()
    horizon = task.config.horizon
    start, goal, case_name = task.case(args.case_id)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    b_init = task.state_to_base_torch(start, device)
    print(f"loaded={args.checkpoint} case={case_name} horizon={horizon}")

    if args.quick:
        deltas = [0.0, 0.1]
        pert_seeds, opt_seeds = [0], [0]
        steps = 25
    else:
        deltas = [0.0, 0.01, 0.03, 0.1]
        pert_seeds, opt_seeds = [0, 1, 2, 3], [0, 1, 2, 3]
        steps = args.closed_loop_steps

    traces: dict = {}
    steady: dict = {}
    window = args.steady_window
    for arm, (sig_start, sig_end) in ARMS.items():
        config = MBDConfig(
            num_samples=args.num_samples,
            num_diffusion_steps=args.num_diffusion_steps,
            sigma_start=sig_start,
            sigma_end=sig_end,
            alpha=ALPHA,
            eta=1.0,
            update_rule=UpdateRule.WEIGHTED_MEAN,
            add_langevin_noise=False,
        )
        optimizer = MBDOptimizer(config, *task.action_bounds)
        t0 = time.perf_counter()
        for delta in deltas:
            runs = []
            for pert_seed in pert_seeds:
                plan_model = perturb_model(model, delta, pert_seed)
                for opt_seed in opt_seeds:
                    errors = closed_loop_run(
                        true_model=model, plan_model=plan_model, task=task,
                        goal=goal, b_init=b_init, optimizer=optimizer,
                        horizon=horizon, steps=steps,
                        rng=np.random.default_rng(opt_seed), device=device,
                    )
                    runs.append(errors)
            traces[(arm, delta)] = runs
            tails = np.array([r[-window:].mean() for r in runs])
            steady[(arm, delta)] = {
                "mean": float(tails.mean()),
                "se": float(tails.std(ddof=1) / np.sqrt(len(tails)))
                if len(tails) > 1
                else 0.0,
            }
        print(f"  arm {arm:10s} done in {time.perf_counter() - t0:.1f} s")

    print(f"\nsteady-state EE error [mm] (last {window} steps)")
    print(f"  {'arm':10s} " + " ".join(f"d={d:<7g}" for d in deltas))
    for arm in ARMS:
        print(
            f"  {arm:10s} "
            + " ".join(f"{steady[(arm, d)]['mean'] * 1000:<9.1f}" for d in deltas)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "results.npz",
        deltas=np.asarray(deltas),
        arms=list(ARMS.keys()),
        steady_mean=np.array([[steady[(a, d)]["mean"] for d in deltas] for a in ARMS]),
        steady_se=np.array([[steady[(a, d)]["se"] for d in deltas] for a in ARMS]),
        traces_keys=np.array([f"{a}|{d}" for a in ARMS for d in deltas]),
        traces_values=np.array(
            [np.stack(traces[(a, d)]) for a in ARMS for d in deltas]
        ),
    )
    delta_show = 0.03 if 0.03 in deltas else deltas[-1]
    make_figure(
        traces, deltas, steady, delta_show, task.config.control_dt,
        args.output_dir / "mirage_closedloop.png",
    )
    print(f"saved={args.output_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
