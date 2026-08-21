"""Decoupling experiment: explore-sigma vs commit-sigma as independent knobs.

Completes caveat (iii): annealing's unique value is decoupling explore-sigma
(sigma_start, early/large -> reach) from commit-sigma (sigma_end, late/small
-> optimism immunity). Fixed-sigma ties them together (one sigma for both) and
is trapped in a reach-vs-robustness tradeoff; only annealing (large sigma_start,
small sigma_end) reaches the "reach AND robust" corner.

Closed-loop (plant = unperturbed BK synthetic truth, planner = delta-perturbed
BK, replan every step) over a 2D grid of (sigma_start, sigma_end), sigma_end <=
sigma_start. Each cell run at delta=0 (reach) and delta>0 (under model error).

Reads as:
  A  reach heatmap (EE err @ delta=0):     horizontal gradient -> reach set by sigma_start
  B  under-error heatmap (EE err @ delta):  minimum only in the off-diagonal
     (large sigma_start, small sigma_end = annealing) corner; the diagonal
     (fixed-sigma / MPPI) is excluded from it
  C  degradation vs sigma_end, one line per sigma_start: collapse => robustness
     is set by sigma_end alone (clean decoupling); spread => partial coupling
     (exploration carries residual vulnerability, caveat ii). Slope vs 2 tests
     whether the O(sigma^2) update law carries to closed-loop steady state.

Example:
    python experiments/explore_commit_grid.py            # full
    python experiments/explore_commit_grid.py --quick    # smoke test
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
DELTA = 0.03
SIGMA_STARTS = [0.1, 0.2, 0.35, 0.6, 1.2]   # explore-sigma (x axis)
SIGMA_ENDS = [0.02, 0.05, 0.1, 0.2, 0.35]   # commit-sigma  (y axis)


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "out" / "franka" / "models" / "bk_seed0.pt",
    )
    p.add_argument("--case-id", type=int, default=0)
    p.add_argument("--num-samples", type=int, default=256)
    p.add_argument("--num-diffusion-steps", type=int, default=5)
    p.add_argument("--closed-loop-steps", type=int, default=50)
    p.add_argument("--steady-window", type=int, default=12)
    p.add_argument("--device", default="cpu")
    p.add_argument("--torch-threads", type=int, default=4)
    p.add_argument("--quick", action="store_true")
    p.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "explore_commit_grid"
    )
    return p.parse_args()


def perturb_model(model, delta, pert_seed):
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


def closed_loop_steady(
    *, true_model, plan_model, task, goal, b_init, optimizer, horizon, steps, window, rng, device
):
    """Closed-loop run; return mean EE error over the last `window` steps.

    Plant lives in base-state space and re-lifts each step (train-time usage);
    the planner re-lifts the measured base state at every replan.
    """

    goal_t = torch.as_tensor(goal, dtype=torch.float32, device=device)
    action_low, action_high = task.action_bounds
    b = b_init.clone()
    U = np.zeros((horizon, NUM_JOINTS))
    errs = np.zeros(steps)
    for s in range(steps):
        with torch.no_grad():
            z0 = true_model.lift(b).detach()

        @torch.no_grad()
        def evaluate(candidates):
            U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
            zs = plan_model.rollout(z0.unsqueeze(0).expand(U_t.shape[0], -1), U_t)
            bs = plan_model.decode(zs)
            return task.trajectory_cost_base_torch(bs, U_t, goal_t).cpu().numpy()

        U = optimizer.optimize(U, evaluate, rng=rng).controls
        u = np.clip(U[0], action_low, action_high)
        with torch.no_grad():
            u_t = torch.as_tensor(u, dtype=torch.float32, device=device)
            z = true_model.step(z0.unsqueeze(0), u_t.unsqueeze(0)).squeeze(0)
            b = true_model.decode(z)
            errs[s] = float(torch.linalg.vector_norm(b[NUM_JOINTS : NUM_JOINTS + 3] - goal_t))
        U = np.roll(U, -1, axis=0)
        U[-1] = U[-2]
    return float(errs[-window:].mean())


# ------------------------------------------------------------------ plotting
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#898781"
GRID, BASE, SURF = "#e1e0d9", "#c3c2b7", "#fcfcfb"
# one line per sigma_start (sequential blue ramp, light->dark = small->large explore)
RAMP = ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281"]


def style(ax):
    ax.set_facecolor(SURF)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(BASE)
        ax.spines[s].set_linewidth(0.8)
    ax.tick_params(colors=MUTED, labelsize=9, width=0.8)


def heatmap(ax, M, title, mark_corner=False):
    import matplotlib.pyplot as plt

    Mm = np.ma.masked_invalid(M) * 1000.0  # mm
    cmap = plt.cm.Blues.copy()
    cmap.set_bad("#e8e8e4")
    im = ax.imshow(Mm, origin="lower", cmap=cmap, aspect="auto")
    ax.set_xticks(range(len(SIGMA_STARTS)), [f"{s:g}" for s in SIGMA_STARTS])
    ax.set_yticks(range(len(SIGMA_ENDS)), [f"{s:g}" for s in SIGMA_ENDS])
    ax.set_xlabel(r"explore  $\sigma_{start}$", fontsize=9.5, color=INK2)
    ax.set_ylabel(r"commit  $\sigma_{end}$", fontsize=9.5, color=INK2)
    ax.set_title(title, loc="left", fontsize=9.5, color=INK)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if np.isfinite(M[i, j]):
                v = Mm[i, j]
                tc = "#ffffff" if v > 0.6 * Mm.max() else INK
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color=tc)
    # diagonal (fixed-sigma) outline
    for k_s, ss in enumerate(SIGMA_STARTS):
        for k_e, se in enumerate(SIGMA_ENDS):
            if abs(ss - se) < 1e-9:
                ax.add_patch(plt.Rectangle((k_s - 0.5, k_e - 0.5), 1, 1, fill=False,
                                           edgecolor="#e34948", lw=1.6))
    if mark_corner:
        # annealing corner: max sigma_start, min sigma_end
        ax.add_patch(plt.Rectangle((len(SIGMA_STARTS) - 1.5, -0.5), 1, 1, fill=False,
                                   edgecolor="#0ca30c", lw=2.2))
    return im


def make_figure(reach, under, degr, slopes, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(13.5, 4.2), dpi=200)
    fig.patch.set_facecolor(SURF)

    imA = heatmap(axA, reach, "A   reach: EE err @ $\\delta$=0  [mm]")
    fig.colorbar(imA, ax=axA, fraction=0.046, pad=0.04)
    imB = heatmap(axB, under, f"B   under error: EE err @ $\\delta$={DELTA}  [mm]",
                  mark_corner=True)
    fig.colorbar(imB, ax=axB, fraction=0.046, pad=0.04)
    axB.text(0.5, -0.42, "red = fixed-$\\sigma$ (MPPI);  green = annealing corner",
             transform=axB.transAxes, ha="center", fontsize=8, color=INK2)
    style(axA)
    style(axB)

    # Panel C: degradation vs sigma_end, one line per sigma_start.
    se = np.asarray(SIGMA_ENDS)
    for k, ss in enumerate(SIGMA_STARTS):
        y = degr[:, k] * 1000.0  # mm, indexed [sigma_end, sigma_start]
        m = np.isfinite(y)
        if m.sum() >= 2:
            axC.plot(se[m], y[m], color=RAMP[k], lw=2, marker="o", ms=4.5,
                     label=f"$\\sigma_{{start}}$={ss:g}")
    axC.set_xscale("log")
    axC.set_yscale("log")
    axC.set_xlabel(r"commit  $\sigma_{end}$", fontsize=9.5, color=INK2)
    axC.set_ylabel(r"degradation $\Delta$ EE [mm]", fontsize=9.5, color=INK2)
    axC.set_title("C   robustness vs commit-$\\sigma$ (lines collapse $\\Rightarrow$ decoupled)",
                  loc="left", fontsize=9.5, color=INK)
    axC.legend(frameon=False, fontsize=8, labelcolor=INK2, loc="upper left")
    axC.grid(True, which="major", color=GRID, linewidth=0.6)
    axC.set_axisbelow(True)
    style(axC)

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
    start, goal, case_name = task.case(args.case_id)
    model, _ = load_checkpoint(args.checkpoint, device=device)
    model.eval()
    b_init = task.state_to_base_torch(start, device)
    print(f"loaded={args.checkpoint} case={case_name} delta={DELTA}")

    if args.quick:
        pert_seeds, opt_seeds, steps = [0], [0], 25
    else:
        pert_seeds, opt_seeds, steps = [0, 1, 2], [0, 1], args.closed_loop_steps

    ne, ns = len(SIGMA_ENDS), len(SIGMA_STARTS)
    reach = np.full((ne, ns), np.nan)     # err @ delta=0
    under = np.full((ne, ns), np.nan)     # err @ delta
    degr = np.full((ne, ns), np.nan)      # under - reach

    t0 = time.perf_counter()
    for js, ss in enumerate(SIGMA_STARTS):
        for ie, se in enumerate(SIGMA_ENDS):
            if se > ss + 1e-9:
                continue
            cfg = MBDConfig(
                num_samples=args.num_samples, num_diffusion_steps=args.num_diffusion_steps,
                sigma_start=ss, sigma_end=se, alpha=ALPHA, eta=1.0,
                update_rule=UpdateRule.WEIGHTED_MEAN, add_langevin_noise=False,
            )
            opt = MBDOptimizer(cfg, *task.action_bounds)
            cell = {0.0: [], DELTA: []}
            for delta in (0.0, DELTA):
                for ps in pert_seeds:
                    plan_model = perturb_model(model, delta, ps)
                    for os_ in opt_seeds:
                        e = closed_loop_steady(
                            true_model=model, plan_model=plan_model, task=task, goal=goal,
                            b_init=b_init, optimizer=opt, horizon=horizon, steps=steps,
                            window=args.steady_window, rng=np.random.default_rng(os_),
                            device=device,
                        )
                        cell[delta].append(e)
            reach[ie, js] = float(np.mean(cell[0.0]))
            under[ie, js] = float(np.mean(cell[DELTA]))
            degr[ie, js] = under[ie, js] - reach[ie, js]
        print(f"  sigma_start={ss:<4g} column done ({time.perf_counter() - t0:.0f}s)")

    # decoupling diagnostics
    print("\nreach (err@0, mm) [rows=sigma_end, cols=sigma_start]:")
    print(np.array2string(reach * 1000, precision=0, suppress_small=True))
    print("under-error (err@delta, mm):")
    print(np.array2string(under * 1000, precision=0, suppress_small=True))

    # slope of degradation vs sigma_end at the largest sigma_start (sharp sigma^2 test)
    se = np.asarray(SIGMA_ENDS)
    slopes = {}
    col = ns - 1  # sigma_start = 1.2
    y = degr[:, col]
    m = np.isfinite(y) & (y > 0)
    if m.sum() >= 2:
        slopes["degr_vs_commit"] = float(
            np.polyfit(np.log(se[m]), np.log(y[m]), 1)[0]
        )
        print(f"\ndegradation vs sigma_end slope (sigma_start=1.2): "
              f"{slopes['degr_vs_commit']:.2f}  (O(sigma^2) predicts 2)")

    # decoupling metric: how much does robustness depend on sigma_start? (line spread)
    # for each sigma_end row, coefficient of variation of degradation across sigma_start
    cvs = []
    for ie in range(ne):
        row = degr[ie, :]
        r = row[np.isfinite(row) & (row > 0)]
        if len(r) >= 2:
            cvs.append(float(np.std(r) / np.mean(r)))
    if cvs:
        print(f"robustness spread across sigma_start (median CV): {np.median(cvs):.2f} "
              f"(0 = perfectly decoupled)")

    # annealing corner vs fixed-sigma diagonal, under error
    corner = under[0, -1]  # smallest sigma_end, largest sigma_start
    diag = [under[k, k] for k in range(min(ne, ns)) if np.isfinite(under[k, k])]
    print(f"\nunder-error: annealing corner = {corner*1000:.0f} mm  |  "
          f"best fixed-sigma (diagonal) = {min(diag)*1000:.0f} mm")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        args.output_dir / "results.npz",
        sigma_starts=SIGMA_STARTS, sigma_ends=SIGMA_ENDS, delta=DELTA,
        reach=reach, under=under, degradation=degr, **slopes,
    )
    make_figure(reach, under, degr, slopes, args.output_dir / "explore_commit_grid.png")
    print(f"saved={args.output_dir / 'results.npz'}")


if __name__ == "__main__":
    main()
