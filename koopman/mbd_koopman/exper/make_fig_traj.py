"""Figure: executed tool paths under four rollout classes on one goal.

Draws the paths recorded by ``exper.run_traj``. One goal, one training seed, one
planner stream: the four paths differ in the rollout model alone.

    python -m exper.make_fig_traj --target 5
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent
FIGS = ROOT.parent / "paper" / "figs"

# Validated categorical slots 1-3 for the learned rollouts; the oracle is the
# reference, so it carries neutral ink and a dashed stroke rather than a hue.
SERIES = [
    ("oracle",   "MBD-true",        "#52514e", (0, (4.5, 2.2)), 1.5, 2),
    ("linear",   "DK-MBD",          "#2a78d6", "solid",         1.9, 3),
    ("split",    "DK-MBD-split",    "#eb6834", "solid",         1.9, 4),
    ("bilinear", "BK-MBD (ours)",   "#1baf7a", "solid",         2.3, 5),
]
INK = "#0b0b0b"
INK_SOFT = "#52514e"


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Nimbus Roman", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.linewidth": 0.6,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 8,
        "figure.dpi": 400,
        "savefig.dpi": 400,
        "pdf.fonttype": 42,
    })


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="traj_fig", help="directory under exper/out")
    ap.add_argument("--target", type=int, default=5)
    ap.add_argument("--elev", type=float, default=18.0)
    ap.add_argument("--azim", type=float, default=-70.0)
    ap.add_argument("--out", default="traj_compare")
    args = ap.parse_args()

    run_dir = ROOT / "out" / args.run
    data = np.load(run_dir / "paths.npz")
    t = args.target
    goal = data[f"goal_t{t}"]
    tol = json.loads((run_dir / "config.json").read_text())["task"]["strict"]

    def drawn(key):
        """The path up to the first control step inside the tolerance.

        A trial runs the full budget without early stopping, so a run that
        arrives early spends the remainder hovering at the goal. That tail is
        the same for every condition that reaches and only crowds the panel,
        so the figure stops each path where the trial is scored as reaching.
        """
        p = data[f"{key}_t{t}"]
        err = data[f"{key}_t{t}_err"]
        hit = np.flatnonzero(err <= tol)
        return p if hit.size == 0 else p[: hit[0] + 2], hit.size > 0

    style()
    fig = plt.figure(figsize=(3.45, 2.65))
    ax = fig.add_subplot(111, projection="3d", computed_zorder=False)

    for key, label, color, ls, lw, z in SERIES:
        p, _ = drawn(key)
        ax.plot(p[:, 0], p[:, 1], p[:, 2], color=color, linestyle=ls,
                linewidth=lw, solid_capstyle="round", zorder=z, label=label)

    start = data[f"oracle_t{t}"][0]
    ax.scatter(*start, s=26, color=INK, marker="o", depthshade=False, zorder=6)
    ax.scatter(*goal, s=75, color=INK, marker="*", depthshade=False, zorder=6)
    ax.text(*(start + np.array([0.0, -0.012, -0.025])), "start",
            color=INK_SOFT, ha="right", va="top", fontsize=7, zorder=7)
    ax.text(*(goal + np.array([0.0, 0.0, 0.04])), "goal", color=INK,
            ha="center", va="bottom", fontsize=7, zorder=7)

    ax.set_xlabel("$x$ [m]", labelpad=-4)
    ax.set_ylabel("$y$ [m]", labelpad=-4)
    ax.set_zlabel("$z$ [m]", labelpad=-6)
    ax.tick_params(axis="both", pad=-2, colors=INK_SOFT, length=2)
    ax.locator_params(nbins=4)
    ax.view_init(elev=args.elev, azim=args.azim)

    # recessive panes and grid
    for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
        pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
        pane._axinfo["grid"].update(color="#e2e1dd", linewidth=0.5)
        pane.line.set_color("#c9c8c3")
    ax.set_box_aspect((1, 1, 0.72), zoom=1.0)
    # a 3-D axes reserves a wide margin it never draws in; take some of it back
    fig.subplots_adjust(left=-0.04, right=1.04, bottom=-0.01, top=1.02)

    fig.legend(loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=2,
               frameon=False, handlelength=1.9, columnspacing=1.4,
               handletextpad=0.55, labelcolor=INK, borderaxespad=0.0)

    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{args.out}.{ext}")
    print(f"wrote {FIGS / (args.out + '.pdf')}")

    for key, label, *_ in SERIES:
        e = data[f"{key}_t{t}_err"]
        p, reached = drawn(key)
        end = f"cut at {len(p) - 1} steps" if reached else f"full {len(p) - 1} steps"
        print(f"  {label:16s} final {e[-1]:.4f} m  min {e.min():.4f} m  {end}")


if __name__ == "__main__":
    main()
