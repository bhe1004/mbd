"""Mechanism-analysis figures for the BK-MBD paper.

(1) pred_error_horizon.png -- open-loop prediction error vs. horizon for the
    trained DK (linear) and BK (bilinear) checkpoints actually used in the
    closed-loop experiments (5 seeds, held-out data), decomposed into the
    end-effector rows (where the configuration-dependent coupling acts) and
    the joint rows (input enters state-independently). Two panels:
    synthetic arm | Franka FR3 (MuJoCo).

(2) tube_validation.png -- empirical error tube e_k (quantile-fitted
    constants, exactly as used by the MBD planner) vs. the true lifted error
    on held-out FR3 rollouts of length 40; prints the violation rate and
    conservatism statistics.

Usage:
    python experiments/analysis_figs.py [--device cuda] [--figs pred tube]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.train import load_checkpoint  # noqa: E402
from bk_mbd.tube import (  # noqa: E402
    bilinear_norm_bounds,
    compute_one_step_residuals,
    fit_tube_constants,
)
from envs.arm import ArmTask  # noqa: E402
from envs.franka import FrankaDatasetConfig, FrankaTask  # noqa: E402

plt.rcParams.update(
    {
        "font.size": 12,
        "axes.grid": True,
        "grid.alpha": 0.35,
        "axes.labelsize": 13,
        "legend.fontsize": 9,
        "figure.dpi": 150,
    }
)

SEEDS = [0, 1, 2, 3, 4]
DK_COLOR = "#1f77b4"
BK_COLOR = "#d62728"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--figs", nargs="+", choices=["pred", "tube"], default=["pred", "tube"]
    )
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "paper" / "figs"
    )
    return parser.parse_args()


@torch.no_grad()
def per_step_errors(model, base_states, controls, device):
    """Per-step open-loop RMSE on the EE rows (7:10) and joint rows (0:7)."""

    b = torch.as_tensor(base_states, dtype=torch.float32, device=device)
    u = torch.as_tensor(controls, dtype=torch.float32, device=device)
    z = model.lift(b[:, 0])
    ee, joints = [], []
    for k in range(u.shape[1]):
        z = model.step(z, u[:, k])
        err = model.decode(z) - b[:, k + 1]
        ee.append(err[:, 7:10].norm(dim=-1).mean().item())
        joints.append(err[:, 0:7].norm(dim=-1).mean().item())
    return np.array(ee), np.array(joints)


def fig_pred(args) -> None:
    panels = []
    for task_name, task, ee_label in (
        ("Synthetic 7-DOF arm", ArmTask(), "end-effector"),
        (
            "Franka FR3 (MuJoCo)",
            FrankaTask(dataset_config=FrankaDatasetConfig(num_snippets=2000)),
            "TCP",
        ),
    ):
        print(f"[pred] held-out data for {task_name} ...")
        ds = task.sample_dataset(2)
        curves = {"dk": {"ee": [], "q": []}, "bk": {"ee": [], "q": []}}
        for name in ("dk", "bk"):
            for seed in SEEDS:
                ckpt = (
                    PROJECT_ROOT
                    / "out"
                    / ("arm" if "arm" in task_name else "franka")
                    / "models"
                    / f"{name}_seed{seed}.pt"
                )
                model, _ = load_checkpoint(ckpt, device=args.device)
                model.eval()
                ee, q = per_step_errors(
                    model, ds["base_states"], ds["controls"], args.device
                )
                curves[name]["ee"].append(ee)
                curves[name]["q"].append(q)
        panels.append((task_name, ee_label, curves))
        h = len(curves["dk"]["ee"][0])
        dk_ee = np.mean(curves["dk"]["ee"], axis=0)
        bk_ee = np.mean(curves["bk"]["ee"], axis=0)
        print(
            f"[pred] {task_name}: EE ratio dk/bk @H1={dk_ee[0]/bk_ee[0]:.2f}x "
            f"@H{h}={dk_ee[-1]/bk_ee[-1]:.2f}x"
        )

    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.4), sharey=False)
    for ax, (task_name, ee_label, curves) in zip(axes, panels):
        hs = np.arange(1, len(curves["dk"]["ee"][0]) + 1)
        for name, color, marker, label in (
            ("dk", DK_COLOR, "o", "linear (DK)"),
            ("bk", BK_COLOR, "s", "bilinear (BK)"),
        ):
            ee = np.array(curves[name]["ee"])
            q = np.array(curves[name]["q"])
            ax.plot(
                hs, ee.mean(0), marker=marker, ms=4, lw=2,
                color=color, label=f"{label} --- {ee_label}",
            )
            ax.fill_between(
                hs, ee.mean(0) - ee.std(0), ee.mean(0) + ee.std(0),
                color=color, alpha=0.18, lw=0,
            )
            ax.plot(
                hs, q.mean(0), marker=marker, ms=3.5, lw=1.4, ls="--",
                alpha=0.5, color=color, label=f"{label} --- joints",
            )
        ax.set_yscale("log")
        ax.set_xlabel("horizon")
        ax.set_title(task_name, fontsize=11)
        ax.legend(loc="lower right", fontsize=7.5)
    axes[0].set_ylabel("open-loop prediction error [m]")
    fig.tight_layout()
    path = args.output_dir / "pred_error_horizon.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved={path}")


def fig_tube(args) -> None:
    device = torch.device(args.device)
    # Constants fitted exactly as in the experiment runners: training data,
    # quantile coverage 0.999, seed-0 BK model.
    task_train = FrankaTask()
    ds_train = task_train.sample_dataset(1)
    model, _ = load_checkpoint(
        PROJECT_ROOT / "out" / "franka" / "models" / "bk_seed0.pt", device=device
    )
    model.eval()
    zs, us, res = compute_one_step_residuals(
        model, ds_train["base_states"], ds_train["controls"], device=device
    )
    tc = fit_tube_constants(zs, us, res, quantile=0.999)
    params = model.bilinear_params()
    norm_a, norm_bs = bilinear_norm_bounds(params)
    print(
        f"[tube] c_x={tc.c_x:.4e} c_u={tc.c_u:.4e} "
        f"||A||2={norm_a:.4f} sum||B_i||2={norm_bs.sum():.4f}"
    )

    # Held-out rollouts, longer than the planning horizon (40 vs 15 steps).
    horizon = 40
    task_test = FrankaTask(
        dataset_config=FrankaDatasetConfig(num_snippets=2000, snippet_horizon=horizon)
    )
    ds_test = task_test.sample_dataset(2)
    b = torch.as_tensor(ds_test["base_states"], dtype=torch.float32, device=device)
    u = torch.as_tensor(ds_test["controls"], dtype=torch.float32, device=device)

    with torch.no_grad():
        z_true = model.lift(b)
        z_hat = model.rollout(model.lift(b[:, 0]), u)
        delta = (z_true - z_hat).norm(dim=-1).cpu().numpy()  # (N, H+1)
        # scalar tube recursion e_{k+1} = (mbar + c_x) e_k + c_x||zhat|| + c_u||u||
        mbar = norm_a + (u.abs().cpu().numpy() * norm_bs[None, None, :]).sum(-1)
        zn = z_hat.norm(dim=-1).cpu().numpy()
        un = u.norm(dim=-1).cpu().numpy()
    n, _ = un.shape[0], un.shape[1]
    tube = np.zeros((n, horizon + 1))
    for k in range(horizon):
        tube[:, k + 1] = (
            (mbar[:, k] + tc.c_x) * tube[:, k] + tc.c_x * zn[:, k] + tc.c_u * un[:, k]
        )

    steps = delta[:, 1:]
    tube_steps = tube[:, 1:]
    viol = int(np.sum(steps > tube_steps + 1e-9))
    total = steps.size
    ratio15 = np.median(tube[:, 15] / np.maximum(delta[:, 15], 1e-12))
    ratio40 = np.median(tube[:, -1] / np.maximum(delta[:, -1], 1e-12))
    print(
        f"[tube] held-out H={horizon}: violations={viol}/{total} "
        f"({100.0*viol/total:.3f}%)  median conservatism "
        f"@15={ratio15:.1f}x @40={ratio40:.1f}x"
    )

    idx = int(np.argsort(delta[:, -1])[n // 2])  # median-final-error rollout
    fig, ax = plt.subplots(figsize=(4.7, 3.2))
    ax.plot(
        delta[idx], lw=2.2, color="#1f77b4",
        label=r"true lifted error $\|z_k-\hat z_k\|$",
    )
    ax.plot(tube[idx], "--", lw=2.2, color="#d62728", label=r"empirical tube $e_k$")
    ax.axvline(15, ls=":", lw=1.2, color="gray")
    ax.set_yscale("log")
    ax.text(
        15.8, delta[idx, 1:].min() * 1.4, "planning horizon",
        fontsize=8, color="gray", rotation=90, va="bottom",
    )
    ax.set_xlabel(r"rollout step $k$")
    ax.set_ylabel("lifted-space error")
    ax.legend(loc="lower right")
    fig.tight_layout()
    path = args.output_dir / "tube_validation.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved={path}")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if "pred" in args.figs:
        fig_pred(args)
    if "tube" in args.figs:
        fig_tube(args)


if __name__ == "__main__":
    main()
