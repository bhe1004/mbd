"""Render the drone window figure (Section C port): BK-MBD threads the circular
window while convexified QP-MPC commits to the near-side branch and stalls.

Reads out/window/trajs.npz (from drone_window.py). By default it auto-selects a
representative target on which QP-MPC stalls (the lowest one -> the deepest
detour) and draws a BK-MBD | QP-MPC pair of 3D panels; pass --targets to pick
specific target indices (one row of panels each).

    python drone/drone_window_fig.py
    python drone/drone_window_fig.py --targets 9 26
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
from matplotlib.lines import Line2D      # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

OUT_DIR = Path(__file__).resolve().parent / "out" / "window"
C_BK = "#2563eb"
C_QP = "#dc2626"
C_WIN = "#16a34a"
C_WALL = "#64748b"
C_START = "#6b7280"
C_TGT = "#111827"
C_DRONE = "#1a1a1a"
WA, WIN = 1, [0, 2]


def wall_with_hole_quads(lims, y_wall, win_c, r_win, n=140):
    (xlo, xhi), _, (zlo, zhi) = lims
    cx, cz = win_c
    th = np.linspace(0, 2 * np.pi, n, endpoint=False)
    cs, sn = np.cos(th), np.sin(th)
    inner = np.column_stack([cx + r_win * cs, cz + r_win * sn])
    outer = np.zeros((n, 2))
    for i in range(n):
        dx, dz = cs[i], sn[i]
        cand = []
        if dx > 1e-9:
            cand.append((xhi - cx) / dx)
        if dx < -1e-9:
            cand.append((xlo - cx) / dx)
        if dz > 1e-9:
            cand.append((zhi - cz) / dz)
        if dz < -1e-9:
            cand.append((zlo - cz) / dz)
        t = min(c for c in cand if c > 0)
        outer[i] = [cx + t * dx, cz + t * dz]
    quads = []
    for i in range(n):
        j = (i + 1) % n
        quads.append([(inner[i, 0], y_wall, inner[i, 1]),
                      (outer[i, 0], y_wall, outer[i, 1]),
                      (outer[j, 0], y_wall, outer[j, 1]),
                      (inner[j, 0], y_wall, inner[j, 1])])
    return quads


def draw_drone(ax, s, color=C_DRONE, L=0.06, r_rot=0.028, zorder=13):
    x, y, z, ya = s
    c, sn = np.cos(ya), np.sin(ya)
    tips = []
    for dx, dy in [(L, L), (-L, L), (-L, -L), (L, -L)]:
        ex, ey = x + c * dx - sn * dy, y + sn * dx + c * dy
        tips.append((ex, ey))
        ax.plot([x, ex], [y, ey], [z, z], "-", color=color, lw=1.6,
                solid_capstyle="round", zorder=zorder)
    th = np.linspace(0, 2 * np.pi, 24)
    for ex, ey in tips:
        ax.plot(ex + r_rot * np.cos(th), ey + r_rot * np.sin(th),
                np.full_like(th, z), "-", color=color, lw=1.4, zorder=zorder + 1)
    ax.scatter([x], [y], [z], color=color, s=14, depthshade=False, zorder=zorder + 1)


def split_by_wall(path, y_wall):
    """Runs on one side of the wall plane, crossing points duplicated, so the
    behind-wall part can be drawn under the ring and the front part over it."""
    runs, cur = [], [path[0]]
    side = path[0][WA] >= y_wall
    for a, b in zip(path[:-1], path[1:]):
        sb = b[WA] >= y_wall
        if sb == side:
            cur.append(b)
        else:
            t = (y_wall - a[WA]) / (b[WA] - a[WA])
            xc = a + t * (b - a)
            cur.append(xc)
            runs.append((side, np.array(cur)))
            cur, side = [xc, b], sb
    runs.append((side, np.array(cur)))
    return runs


def draw_panel(ax, path, target, geo, color, title, lims, view):
    y_wall, win_c, r_win = geo
    (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
    ax.computed_zorder = False
    for behind, seg in split_by_wall(path, y_wall):
        if behind:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=3.0,
                    solid_capstyle="round", zorder=2)
    ax.add_collection3d(Poly3DCollection(
        wall_with_hole_quads(lims, y_wall, win_c, r_win),
        facecolor=C_WALL, edgecolor="none", alpha=0.28, zorder=4))
    th = np.linspace(0, 2 * np.pi, 80)
    ax.plot(win_c[0] + r_win * np.cos(th), np.full_like(th, y_wall),
            win_c[1] + r_win * np.sin(th), color=C_WIN, lw=3, zorder=5)
    for behind, seg in split_by_wall(path, y_wall):
        if not behind:
            ax.plot(seg[:, 0], seg[:, 1], seg[:, 2], color=color, lw=3.0,
                    solid_capstyle="round", zorder=12)
    ax.scatter(*path[0, :3], marker="o", s=95, facecolors="none",
               edgecolors=C_START, linewidths=1.5, depthshade=False, zorder=13)
    ax.scatter(*path[0, :3], marker=".", s=18, color=C_START, depthshade=False,
               zorder=13)
    ax.scatter(*target, marker="s", s=80, color=C_TGT, edgecolor="white",
               linewidth=0.6, depthshade=False, zorder=15)
    draw_drone(ax, path[-1], color=C_DRONE, zorder=16)
    ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
    try:
        ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo))
    except Exception:
        pass
    ax.view_init(elev=view[0], azim=view[1])
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_ticklabels([])
    ax.set_xlabel("x", labelpad=-10, fontsize=8)
    ax.set_ylabel("y", labelpad=-10, fontsize=8)
    ax.set_zlabel("z", labelpad=-10, fontsize=8)
    ax.set_title(title, fontsize=12, y=1.02)


def auto_targets(d):
    """Pick a representative QP-stall target (lowest z -> deepest detour)."""
    trials = OUT_DIR / "trials.jsonl"
    stalls = []
    if trials.exists():
        rows = [json.loads(l) for l in trials.read_text().splitlines() if l.strip()]
        qp = {r["target_idx"]: r for r in rows if r["planner"] == "bk_qp_sqp"}
        for i, r in qp.items():
            if r["safe_stall"]:
                stalls.append((d["targets"][i][2], i))
    if stalls:
        return [min(stalls)[1]]
    return [0]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", type=int, nargs="+", default=None)
    ap.add_argument("--out", type=Path, default=OUT_DIR / "drone_window.png")
    ap.add_argument("--dpi", type=int, default=200)
    ap.add_argument("--elev", type=float, default=24.0)
    ap.add_argument("--azim", type=float, default=-58.0)
    args = ap.parse_args()

    npz = OUT_DIR / "trajs.npz"
    if not npz.exists():
        raise SystemExit(f"no trajs at {npz} -- run drone_window.py first")
    d = np.load(npz)
    geo = (float(d["wall"][0]), d["win_c"], float(d["r_win"][0]))
    rows = args.targets if args.targets is not None else auto_targets(d)

    bk = {t: d[f"bk_mbd_{t}"] for t in rows}
    qp = {t: d[f"bk_qp_sqp_{t}"] for t in rows}
    y_wall, win_c, r_win = geo
    win_bbox = np.array([[win_c[0] - r_win, y_wall, win_c[1] - r_win],
                         [win_c[0] + r_win, y_wall, win_c[1] + r_win]])
    allpts = np.concatenate(
        [p[:, :3] for t in rows for p in (bk[t], qp[t])]
        + [d["targets"][rows], win_bbox])
    lo, hi = allpts.min(0) - 0.05, allpts.max(0) + 0.05
    lims = ((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2]))
    view = (args.elev, args.azim)

    nr = len(rows)
    fig = plt.figure(figsize=(9.2, 4.4 * nr))
    for i, t in enumerate(rows):
        ax1 = fig.add_subplot(nr, 2, 2 * i + 1, projection="3d")
        ax2 = fig.add_subplot(nr, 2, 2 * i + 2, projection="3d")
        draw_panel(ax1, bk[t], d["targets"][t], geo, C_BK,
                   "BK-MBD  (sampling)", lims, view)
        draw_panel(ax2, qp[t], d["targets"][t], geo, C_QP,
                   "Convexified bilinear QP-MPC", lims, view)

    handles = [
        Line2D([0], [0], color=C_WIN, lw=3, label="window (only passage)"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor="none",
               markeredgecolor=C_START, markersize=9, label="start"),
        Line2D([0], [0], marker="s", color="none", markerfacecolor=C_TGT,
               markersize=9, label="target"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=9.5, bbox_to_anchor=(0.5, 1.0))
    fig.subplots_adjust(left=0.0, right=1.0, top=0.9, bottom=0.02,
                        wspace=-0.15, hspace=0.1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {args.out}  (targets={rows})")


if __name__ == "__main__":
    main()
