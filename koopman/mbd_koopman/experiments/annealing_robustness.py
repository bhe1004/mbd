"""Witness experiment: annealing self-robustness of MBD to smooth model error.

Verifies the Proposition in `plan/annealing_robustness.md` on the trained
franka BK model, which serves as *synthetic ground truth* so the injected
error is the only model error in play (no true-simulator confound):

- Exp 1 (direct check): at a realistic iterate U_bar, sample one candidate
  batch per sigma, weight it under the true vs the perturbed rollout cost,
  and measure the softmax-mean contamination ||mu_hat - mu||. Prediction:
  slope ~= 2 on log-log axes (contamination = O(sigma^2)).
- Exp 2 (end-to-end): with the same sample budget, plan with an annealed
  schedule vs fixed-sigma (MPPI-like) schedules under perturbation size
  delta, and compare the true-cost degradation
  Delta(delta) = J_true(U_delta) - J_true(U_0) per arm (self-referenced at
  delta = 0 to separate error sensitivity from optimizer quality).
  Prediction: the annealed curve is the flattest.

The perturbation adds delta * ||P||_F * G/||G||_F (G fixed per seed) to each
lifted-dynamics matrix P in {A, B0, B_i} - a smooth error field in U, the
regime the Proposition covers.

Example:

    python experiments/annealing_robustness.py            # full run
    python experiments/annealing_robustness.py --quick    # smoke test
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
from bk_mbd.costs import stable_softmax_from_cost  # noqa: E402
from bk_mbd.mbd import MBDOptimizer  # noqa: E402
from bk_mbd.train import load_checkpoint  # noqa: E402
from envs.franka import NUM_JOINTS, FrankaTask  # noqa: E402

ALPHA = 0.4


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "out" / "franka" / "models" / "bk_seed0.pt",
    )
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=256)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--torch-threads", type=int, default=4)
    parser.add_argument("--quick", action="store_true", help="reduced smoke-test settings")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "annealing_robustness"
    )
    return parser.parse_args()


def perturb_model(model, delta: float, pert_seed: int):
    """Return a copy with A, B0, Bs shifted by a fixed random direction.

    Each matrix P receives delta * ||P||_F * G/||G||_F with G drawn from a
    generator seeded by pert_seed, so the error *direction* is fixed across
    delta values and only its magnitude changes.
    """

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


def make_cost_fn(model, task, goal, z0, device):
    """Batched rollout cost U (K,T,m) -> costs (K,) under one model."""

    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)

    @torch.no_grad()
    def cost_fn(candidates: np.ndarray) -> np.ndarray:
        U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
        zs = model.rollout(z0.expand(U_t.shape[0], -1), U_t)
        bs = model.decode(zs)
        return task.trajectory_cost_base_torch(bs, U_t, goal_t).cpu().numpy()

    return cost_fn


def exp1_update_contamination(
    *,
    model,
    task,
    goal,
    z0,
    U_bar,
    action_low,
    action_high,
    sigmas,
    num_samples,
    reps,
    delta,
    pert_seed,
    device,
):
    """||mu_hat - mu|| per sigma: same candidates, true vs perturbed weights.

    Candidates are clipped to the action bounds exactly as in the MBD
    sampler; at sigma >~ 0.5 clipping compresses the effective spread, so the
    O(sigma^2) scaling is read off the small-sigma half of the range.
    """

    cost_true = make_cost_fn(model, task, goal, z0, device)
    cost_pert = make_cost_fn(
        perturb_model(model, delta, pert_seed), task, goal, z0, device
    )
    rng = np.random.default_rng(pert_seed)
    contamination = np.zeros((len(sigmas), reps))
    for i, sigma in enumerate(sigmas):
        for r in range(reps):
            xi = rng.normal(size=(num_samples,) + U_bar.shape)
            cand = np.clip(U_bar[None] + sigma * xi, action_low, action_high)
            w = stable_softmax_from_cost(cost_true(cand), ALPHA)
            w_hat = stable_softmax_from_cost(cost_pert(cand), ALPHA)
            mu = np.einsum("k,kij->ij", w, cand)
            mu_hat = np.einsum("k,kij->ij", w_hat, cand)
            contamination[i, r] = float(np.linalg.norm(mu_hat - mu))
    return contamination


def exp2_degradation(
    *,
    model,
    task,
    goal,
    z0,
    horizon,
    action_low,
    action_high,
    arms,
    deltas,
    pert_seeds,
    opt_seeds,
    num_samples,
    num_steps,
    device,
):
    """True cost of the plan each arm returns under perturbation delta."""

    cost_true = make_cost_fn(model, task, goal, z0, device)
    costs = {}  # (arm, delta, pert_seed, opt_seed) -> J_true of returned plan
    for arm_name, (sig_start, sig_end) in arms.items():
        config = MBDConfig(
            num_samples=num_samples,
            num_diffusion_steps=num_steps,
            sigma_start=sig_start,
            sigma_end=sig_end,
            alpha=ALPHA,
            eta=1.0,
            update_rule=UpdateRule.WEIGHTED_MEAN,
            add_langevin_noise=False,
        )
        optimizer = MBDOptimizer(config, action_low, action_high)
        t0 = time.perf_counter()
        for delta in deltas:
            for pert_seed in pert_seeds:
                evaluate = make_cost_fn(
                    perturb_model(model, delta, pert_seed), task, goal, z0, device
                )
                for opt_seed in opt_seeds:
                    result = optimizer.optimize(
                        np.zeros((horizon, NUM_JOINTS)),
                        evaluate,
                        rng=np.random.default_rng(opt_seed),
                    )
                    j_true = float(cost_true(result.controls[None])[0])
                    costs[(arm_name, delta, pert_seed, opt_seed)] = j_true
        print(f"  arm {arm_name:12s} done in {time.perf_counter() - t0:.1f} s")
    return costs


# ------------------------------------------------------------------ plotting
INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASE = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = {  # fixed categorical slot order (validated default palette)
    "annealed": "#2a78d6",
    "fixed-mid": "#1baf7a",
    "fixed-high": "#eda100",
}
MARKERS = {"annealed": "o", "fixed-mid": "s", "fixed-high": "^"}


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


def make_figure(sigmas, contamination, deltas, degradation, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.5, 4.0), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    # Panel A: contamination vs sigma, log-log, slope-2 guide.
    med = np.median(contamination, axis=1)
    q25, q75 = np.percentile(contamination, [25, 75], axis=1)
    ax1.fill_between(sigmas, q25, q75, color=SERIES["annealed"], alpha=0.18, lw=0)
    ax1.plot(sigmas, med, color=SERIES["annealed"], lw=2, marker="o", ms=4.5)
    # slope-2 guide anchored at the smallest-sigma median
    guide = med[0] * (sigmas / sigmas[0]) ** 2
    ax1.plot(sigmas, guide, ls="--", lw=1.2, color=MUTED)
    ax1.annotate(
        "slope 2", xy=(sigmas[len(sigmas) // 2], guide[len(sigmas) // 2]),
        xytext=(4, -11), textcoords="offset points", fontsize=9, color=MUTED,
    )
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel(r"sampling scale $\sigma$", fontsize=10, color=INK_2)
    ax1.set_ylabel(r"update contamination $\|\hat\mu-\mu\|$", fontsize=10, color=INK_2)
    ax1.set_title(
        "A    one-update contamination is $O(\\sigma^2)$",
        loc="left", fontsize=10.5, color=INK,
    )
    style_axis(ax1)

    # Panel B: degradation vs delta per arm.
    for arm, stats in degradation.items():
        d = np.array([x for x in deltas if x > 0.0])
        mean = np.array([stats[x]["mean"] for x in d])
        se = np.array([stats[x]["se"] for x in d])
        ax2.fill_between(d, mean - se, mean + se, color=SERIES[arm], alpha=0.15, lw=0)
        ax2.plot(
            d, mean, color=SERIES[arm], lw=2,
            marker=MARKERS[arm], ms=5.5, label=arm,
        )
    ax2.set_xscale("log")
    ax2.set_xlabel(r"model perturbation $\delta$ (relative)", fontsize=10, color=INK_2)
    ax2.set_ylabel(
        r"true-cost degradation $J_{\rm true}(U_\delta)-J_{\rm true}(U_0)$",
        fontsize=10, color=INK_2,
    )
    ax2.set_title(
        "B    planning degradation under smooth model error",
        loc="left", fontsize=10.5, color=INK,
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
    action_low, action_high = task.action_bounds
    start, goal, case_name = task.case(args.case_id)

    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    b0 = task.state_to_base_torch(start, device)
    z0 = model.lift(b0)
    print(f"loaded={args.checkpoint} case={case_name} horizon={horizon}")

    if args.quick:
        sigmas = np.geomspace(0.02, 1.2, 6)
        exp1_reps, num_steps = 5, 6
        deltas = [0.0, 0.01, 0.1]
        pert_seeds, opt_seeds = [0], [0, 1]
    else:
        sigmas = np.geomspace(0.02, 1.2, 10)
        exp1_reps, num_steps = 20, 10
        deltas = [0.0, 0.003, 0.01, 0.03, 0.1]
        pert_seeds, opt_seeds = [0, 1, 2], [0, 1, 2]

    # A realistic iterate for Exp 1: one annealed solve under the true model.
    warm_config = MBDConfig(
        num_samples=args.num_samples,
        num_diffusion_steps=8,
        sigma_start=1.2,
        sigma_end=0.1,
        alpha=ALPHA,
        eta=1.0,
        update_rule=UpdateRule.WEIGHTED_MEAN,
        add_langevin_noise=False,
    )
    warm = MBDOptimizer(warm_config, action_low, action_high).optimize(
        np.zeros((horizon, NUM_JOINTS)),
        make_cost_fn(model, task, goal, z0, device),
        rng=np.random.default_rng(0),
    )
    U_bar = warm.controls
    print(f"Exp 1: contamination vs sigma (reps={exp1_reps}, delta=0.01)")
    contamination = exp1_update_contamination(
        model=model, task=task, goal=goal, z0=z0, U_bar=U_bar,
        action_low=action_low, action_high=action_high,
        sigmas=sigmas, num_samples=args.num_samples, reps=exp1_reps,
        delta=0.01, pert_seed=0, device=device,
    )
    med = np.median(contamination, axis=1)
    lo = sigmas <= 0.3  # clip-free regime for the slope fit
    slope = np.polyfit(np.log(sigmas[lo]), np.log(med[lo]), 1)[0]
    print(f"  empirical log-log slope (sigma <= 0.3): {slope:.2f}  (prediction: 2)")

    print("Exp 2: end-to-end degradation (same sample budget per arm)")
    arms = {
        "annealed": (1.2, 0.05),
        "fixed-mid": (0.35, 0.35),
        "fixed-high": (1.2, 1.2),
    }
    costs = exp2_degradation(
        model=model, task=task, goal=goal, z0=z0, horizon=horizon,
        action_low=action_low, action_high=action_high,
        arms=arms, deltas=deltas, pert_seeds=pert_seeds, opt_seeds=opt_seeds,
        num_samples=args.num_samples, num_steps=num_steps, device=device,
    )

    # Degradation relative to each arm's own delta=0 solution (same seeds).
    degradation = {}
    print(f"  {'arm':12s} " + " ".join(f"d={d:<7g}" for d in deltas if d > 0))
    for arm in arms:
        stats = {}
        for delta in deltas:
            if delta == 0.0:
                continue
            diffs = [
                costs[(arm, delta, ps, os_)] - costs[(arm, 0.0, ps, os_)]
                for ps in pert_seeds
                for os_ in opt_seeds
            ]
            diffs = np.asarray(diffs)
            stats[delta] = {
                "mean": float(diffs.mean()),
                "se": float(diffs.std(ddof=1) / np.sqrt(len(diffs)))
                if len(diffs) > 1
                else 0.0,
            }
        degradation[arm] = stats
        print(
            f"  {arm:12s} "
            + " ".join(f"{stats[d]['mean']:<9.3f}" for d in deltas if d > 0)
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "results.npz",
        sigmas=sigmas,
        contamination=contamination,
        slope=slope,
        deltas=np.asarray(deltas),
        arms=list(arms.keys()),
        degradation_mean=np.array(
            [[degradation[a][d]["mean"] for d in deltas if d > 0] for a in arms]
        ),
        degradation_se=np.array(
            [[degradation[a][d]["se"] for d in deltas if d > 0] for a in arms]
        ),
        raw_costs_keys=np.array([str(k) for k in costs.keys()]),
        raw_costs_values=np.array(list(costs.values())),
    )
    make_figure(
        sigmas, contamination, deltas, degradation,
        args.output_dir / "annealing_robustness.png",
    )
    print(f"saved={args.output_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
