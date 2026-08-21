"""THE per-update figure: how sampling scale sigma filters the optimism impostor.

At a fixed, realistic iterate Ubar, with a delta-perturbed BK model (the
impostor-prone surrogate), sweep the sampling scale sigma and measure - all
purely open-loop, single-update quantities, so there is NO closed-loop
confound (no fixed-low paradox, no tuning dependence):

Panel A (mechanism, log-log):
  - common error   ||mean_k eps_k||   ~ slope 0 (flat), LARGE -> softmax cancels it
  - differential   std_k(eps_k)        ~ slope 1        -> the ONLY part that biases selection
  - contamination  ||mu_hat - mu||      ~ slope 2 = (differential prop sigma) x (spread prop sigma)
  Two vertical markers at sigma_commit (MBD sigma_end) and sigma_fixed (MPPI):
  read the contamination ratio straight off the curve -> "MBD commit is Nx less
  contaminated" with no closed-loop experiment.

Panel B (that the surviving bias is OPTIMISM, not generic error):
  - exploitation regret  J(mu_hat) - J(mu)      > 0, ->0 as sigma->0  (true cost of being fooled)
  - model's claimed gain  Jhat(mu) - Jhat(mu_hat) > 0                  (the lie: model rates its pick better)

eps_k = Jhat_k - J_k is the surrogate's per-candidate cost error (perturbed - true).
mu     = softmax(-J/alpha)-weighted mean  (the honest pick, true model)
mu_hat = softmax(-Jhat/alpha)-weighted mean (the fooled pick, perturbed model)

Example:
    python experiments/optimism_scaling.py            # full
    python experiments/optimism_scaling.py --quick    # smoke test
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
SIGMA_COMMIT = 0.05   # MBD sigma_end (realtime_franka)
SIGMA_FIXED = 0.35    # fixed-mid MPPI analog


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "out" / "franka" / "models" / "bk_seed0.pt",
    )
    p.add_argument("--case-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--delta", type=float, default=0.03, help="smooth model perturbation")
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "optimism_scaling"
    )
    return p.parse_args()


def perturb_model(model, delta: float, pert_seed: int):
    """delta * ||P||_F-scaled fixed random direction on A, B0, Bs (smooth error)."""

    if delta == 0.0:
        return model
    pert = copy.deepcopy(model)
    gen = torch.Generator().manual_seed(pert_seed)
    with torch.no_grad():
        for name in ("A", "B0", "Bs"):
            pp = getattr(pert, name)
            g = torch.randn(pp.shape, generator=gen)
            pp.add_(delta * pp.norm() * g / g.norm())
    return pert


def make_cost_fn(model, task, goal, z0, device):
    """Batched rollout cost: U (K,T,m) -> costs (K,) under one model."""

    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)

    @torch.no_grad()
    def cost_fn(candidates: np.ndarray) -> np.ndarray:
        U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
        zs = model.rollout(z0.expand(U_t.shape[0], -1), U_t)
        bs = model.decode(zs)
        return task.trajectory_cost_base_torch(bs, U_t, goal_t).cpu().numpy()

    return cost_fn


def measure(
    *,
    cost_true,
    cost_pert,
    U_bar,
    action_low,
    action_high,
    sigmas,
    num_samples,
    reps,
    seed,
):
    """Per-(sigma, rep): common/differential error, contamination, optimism."""

    rng = np.random.default_rng(seed)
    S, R = len(sigmas), reps
    out = {
        k: np.zeros((S, R))
        for k in ("common", "differential", "contamination", "regret", "claimed", "opt_ok")
    }
    for i, sigma in enumerate(sigmas):
        for r in range(R):
            xi = rng.normal(size=(num_samples,) + U_bar.shape)
            cand = np.clip(U_bar[None] + sigma * xi, action_low, action_high)
            J = cost_true(cand)          # true per-candidate cost
            Jh = cost_pert(cand)         # perturbed per-candidate cost
            eps = Jh - J                 # per-candidate model error
            w = stable_softmax_from_cost(J, ALPHA)
            wh = stable_softmax_from_cost(Jh, ALPHA)
            mu = np.einsum("k,kij->ij", w, cand)     # honest pick
            mu_hat = np.einsum("k,kij->ij", wh, cand)  # fooled pick

            out["common"][i, r] = abs(float(np.mean(eps)))
            out["differential"][i, r] = float(np.std(eps))
            out["contamination"][i, r] = float(np.linalg.norm(mu_hat - mu))

            # Optimism signature: model rates its pick better (Jhat(mu)>Jhat(mu_hat)),
            # but it is truly worse (J(mu_hat)>J(mu)).
            Jt = cost_true(np.stack([mu, mu_hat]))       # [J(mu), J(mu_hat)]
            Jp = cost_pert(np.stack([mu, mu_hat]))       # [Jhat(mu), Jhat(mu_hat)]
            regret = float(Jt[1] - Jt[0])                # J(mu_hat) - J(mu)
            claimed = float(Jp[0] - Jp[1])               # Jhat(mu) - Jhat(mu_hat)
            out["regret"][i, r] = regret
            out["claimed"][i, r] = claimed
            out["opt_ok"][i, r] = 1.0 if (regret > 0 and claimed > 0) else 0.0
    return out


def loglog_slope(x, y, mask):
    return float(np.polyfit(np.log(x[mask]), np.log(np.maximum(y[mask], 1e-30)), 1)[0])


# ------------------------------------------------------------------ plotting
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
C_COMMON, C_DIFF, C_CONT = "#eda100", "#1baf7a", "#2a78d6"
C_REGRET, C_LIE = "#2a78d6", "#e34948"


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8)
    ax.grid(True, which="major", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def make_figure(sigmas, med, slopes, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)

    # ---- Panel A: the mechanism ------------------------------------------
    def guide(anchor_x, anchor_y, p, x):
        return anchor_y * (x / anchor_x) ** p

    for key, color, label, mk in (
        ("common", C_COMMON, "common  $\\|\\overline{\\epsilon}\\|$  (cancelled)", "o"),
        ("differential", C_DIFF, "differential  $\\mathrm{std}_k\\,\\epsilon_k$", "s"),
        ("contamination", C_CONT, "contamination  $\\|\\hat\\mu-\\mu\\|$", "D"),
    ):
        axA.plot(sigmas, med[key], color=color, lw=2, marker=mk, ms=4.5, label=label, zorder=3)

    # slope guides anchored at the small-sigma end
    xs = sigmas
    lo = sigmas <= 0.3
    x0 = sigmas[lo][len(sigmas[lo]) // 2]
    for p, key in ((1.0, "differential"), (2.0, "contamination")):
        y0 = med[key][np.argmin(np.abs(sigmas - x0))]
        axA.plot(xs, guide(x0, y0, p, xs), ls="--", lw=1.1, color=MUTED, zorder=1)
    axA.annotate("slope 1", (sigmas[2], med["differential"][2]), (6, -12),
                 textcoords="offset points", fontsize=8.5, color=MUTED)
    axA.annotate("slope 2", (sigmas[2], med["contamination"][2]), (6, -12),
                 textcoords="offset points", fontsize=8.5, color=MUTED)

    # commit vs fixed markers + contamination ratio
    def cont_at(s):
        return float(np.interp(np.log(s), np.log(sigmas), np.log(med["contamination"])))
    r_commit = np.exp(cont_at(SIGMA_COMMIT))
    r_fixed = np.exp(cont_at(SIGMA_FIXED))
    import matplotlib.transforms as mtransforms
    blend = mtransforms.blended_transform_factory(axA.transData, axA.transAxes)
    for s, c, lab, ha in (
        (SIGMA_COMMIT, C_CONT, f"MBD commit $\\sigma_{{end}}$={SIGMA_COMMIT}", "right"),
        (SIGMA_FIXED, C_LIE, f"MPPI $\\sigma$={SIGMA_FIXED}", "left"),
    ):
        axA.axvline(s, color=c, lw=1.2, ls=":", alpha=0.9, zorder=0)
        axA.text(s, 0.97, "  " + lab + "  ", transform=blend, fontsize=8.5, color=c,
                 ha=ha, va="top", rotation=90)
    axA.set_xscale("log")
    axA.set_yscale("log")
    axA.set_xlabel(r"sampling scale $\sigma$", fontsize=10, color=INK2)
    axA.set_ylabel("cost-space magnitude", fontsize=10, color=INK2)
    axA.set_title(
        f"A   common cancels, only differential ($\\propto\\sigma$) fools "
        f"selection $\\Rightarrow\\ \\|\\hat\\mu-\\mu\\|\\propto\\sigma^2$",
        loc="left", fontsize=10, color=INK,
    )
    axA.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    axA.text(0.97, 0.03,
             f"contamination ratio  MPPI / MBD-commit = {r_fixed / r_commit:.0f}x",
             transform=axA.transAxes, ha="right", va="bottom", fontsize=8.5, color=INK)
    style(axA)

    # ---- Panel B: it is optimism -----------------------------------------
    axB.plot(sigmas, med["regret"], color=C_REGRET, lw=2, marker="D", ms=4.5,
             label=r"true cost of being fooled  $J(\hat\mu)-J(\mu)$", zorder=3)
    axB.plot(sigmas, med["claimed"], color=C_LIE, lw=2, marker="o", ms=4.5,
             label=r"model's claimed gain  $\hat J(\mu)-\hat J(\hat\mu)$", zorder=3)
    axB.set_xscale("log")
    axB.set_yscale("log")
    axB.set_xlabel(r"sampling scale $\sigma$", fontsize=10, color=INK2)
    axB.set_ylabel("true cost units", fontsize=10, color=INK2)
    axB.set_title(
        "B   the surviving bias is optimism (model claims a gain it does not deliver),"
        "\n      and both vanish as $\\sigma\\to0$",
        loc="left", fontsize=10, color=INK,
    )
    axB.legend(frameon=False, fontsize=8.5, labelcolor=INK2, loc="upper left")
    style(axB)

    fig.tight_layout()
    fig.savefig(out_path, facecolor=SURF, bbox_inches="tight")
    fig.savefig(out_path.with_suffix(".pdf"), facecolor=SURF, bbox_inches="tight")
    print(f"figure saved: {out_path} (+ .pdf)")


# ---------------------------------------------------------------------- main
def main():
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
    z0 = model.lift(task.state_to_base_torch(start, device))
    print(f"loaded={args.checkpoint} case={case_name} delta={args.delta}")

    if args.quick:
        sigmas = np.geomspace(0.015, 1.2, 8)
        reps = 8
    else:
        sigmas = np.geomspace(0.015, 1.2, 12)
        reps = 40

    # Realistic iterate: one annealed solve under the true model.
    warm_cfg = MBDConfig(
        num_samples=args.num_samples, num_diffusion_steps=8,
        sigma_start=1.2, sigma_end=0.1, alpha=ALPHA, eta=1.0,
        update_rule=UpdateRule.WEIGHTED_MEAN, add_langevin_noise=False,
    )
    cost_true = make_cost_fn(model, task, goal, z0, device)
    U_bar = MBDOptimizer(warm_cfg, action_low, action_high).optimize(
        np.zeros((horizon, NUM_JOINTS)), cost_true, rng=np.random.default_rng(0)
    ).controls
    cost_pert = make_cost_fn(
        perturb_model(model, args.delta, 0), task, goal, z0, device
    )

    t0 = time.perf_counter()
    res = measure(
        cost_true=cost_true, cost_pert=cost_pert, U_bar=U_bar,
        action_low=action_low, action_high=action_high, sigmas=sigmas,
        num_samples=args.num_samples, reps=reps, seed=0,
    )
    print(f"measured in {time.perf_counter() - t0:.1f} s")

    med = {k: np.median(v, axis=1) for k, v in res.items()}
    lo = sigmas <= 0.3
    slopes = {
        "common": loglog_slope(sigmas, med["common"], lo),
        "differential": loglog_slope(sigmas, med["differential"], lo),
        "contamination": loglog_slope(sigmas, med["contamination"], lo),
    }
    opt_frac = float(res["opt_ok"].mean())
    print("log-log slopes (sigma<=0.3):")
    print(f"  common       {slopes['common']:+.2f}   (predict 0)")
    print(f"  differential {slopes['differential']:+.2f}   (predict 1)")
    print(f"  contamination{slopes['contamination']:+.2f}   (predict 2)")
    print(f"optimism signature holds in {opt_frac * 100:.0f}% of (sigma,rep)")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "results.npz",
        sigmas=sigmas, delta=args.delta, opt_frac=opt_frac,
        **{f"med_{k}": v for k, v in med.items()},
        **{f"slope_{k}": s for k, s in slopes.items()},
    )
    make_figure(sigmas, med, slopes, args.output_dir / "optimism_scaling.png")
    print(f"saved={args.output_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
