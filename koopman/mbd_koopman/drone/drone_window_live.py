"""Live viewer for the drone window experiment (Section C port).

Pops a real Tk window and animates the closed loop AS IT PLANS: BK-MBD (left,
blue) threads the circular window and reaches the target; convexified bilinear
QP-MPC (right, red) commits to the near-side branch and stalls in front of the
wall. The drone quadrotor marker and its trail update after every control step.

    python drone/drone_window_live.py                # BK then QP, one target
    python drone/drone_window_live.py --target 1     # pick a target index
    python drone/drone_window_live.py --planner bk_mbd
    python drone/drone_window_live.py --steps 60     # cap closed-loop steps

All experiment CONDITIONS (geometry, horizon, sigma schedule, wall penalty, cost
weights, limits, ...) live in drone/config.json and are read through
`drone_window`; edit config.json and re-run this viewer to see a new condition.
To sweep a single knob without touching config.json, override it in main() right
after `geo = dw.Geometry()`, e.g. `dw.MBD_SIG_START = 2.0` or
`geo.win_c = np.array([0.0, 0.78])`.

Needs a display (uses the TkAgg backend). For a headless render use a GIF
instead.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt          # noqa: E402
import numpy as np                       # noqa: E402
import torch                             # noqa: E402
from mpl_toolkits.mplot3d.art3d import Poly3DCollection  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import drone_window as dw                # noqa: E402

C_BK = "#2563eb"     # BK-MBD (sampling)
C_QP = "#dc2626"     # convexified QP-MPC
C_WIN = "#16a34a"    # window ring
C_WALL = "#64748b"   # wall panel
C_START = "#6b7280"
C_TGT = "#111827"
C_TRAIL = None       # per-panel color


def wall_with_hole_quads(lims, y_wall, win_c, r_win, n=120):
    """Tessellate the wall rectangle MINUS the circular window into quads, so the
    hole is a real opening (ported from version 2 make_window_reach.py)."""

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


C_DRONE = "#1a1a1a"      # quadrotor body/arms/rotors (near-black, like a real drone)


def draw_drone(ax, s, L=0.05, r_rot=0.02, color=C_DRONE, zorder=13):
    """Draw a quadrotor at pose s=[x,y,z,yaw] as an X-frame with four rotor disks
    (in the level x-y plane at height z). Returns the list of created artists so
    the caller can remove them on the next frame."""

    x, y, z, ya = s
    c, sn = np.cos(ya), np.sin(ya)
    arts = []
    # X-configuration arm tips (rotors at the four corners)
    tips = []
    for dx, dy in [(L, L), (-L, L), (-L, -L), (L, -L)]:
        ex, ey = x + c * dx - sn * dy, y + sn * dx + c * dy
        tips.append((ex, ey))
        (arm,) = ax.plot([x, ex], [y, ey], [z, z], "-", color=color, lw=1.7,
                         solid_capstyle="round", zorder=zorder)
        arts.append(arm)
    # rotor disks
    th = np.linspace(0, 2 * np.pi, 24)
    cs, ss = r_rot * np.cos(th), r_rot * np.sin(th)
    for ex, ey in tips:
        (rot,) = ax.plot(ex + cs, ey + ss, np.full_like(th, z), "-",
                         color=color, lw=1.5, zorder=zorder + 1)
        arts.append(rot)
    # small central hub
    hub = ax.scatter([x], [y], [z], color=color, s=16, depthshade=False,
                     zorder=zorder + 1)
    arts.append(hub)
    return arts


class LivePanel:
    def __init__(self, ax, geo, target, color, title, lims):
        self.ax = ax
        self.geo = geo
        self.color = color
        self.lims = lims
        self.dyn = []                       # dynamic artists to clear each frame
        self.title = title
        self.target = np.asarray(target, np.float64)
        self._draw_static(target)

    def _draw_static(self, target):
        ax, geo, lims = self.ax, self.geo, self.lims
        (xlo, xhi), (ylo, yhi), (zlo, zhi) = lims
        ax.computed_zorder = False
        # wall panel with a real circular hole
        ax.add_collection3d(Poly3DCollection(
            wall_with_hole_quads(lims, geo.y_wall, geo.win_c, geo.r_win),
            facecolor=C_WALL, edgecolor="none", alpha=0.30, zorder=4))
        th = np.linspace(0, 2 * np.pi, 80)
        ax.plot(geo.win_c[0] + geo.r_win * np.cos(th),
                np.full_like(th, geo.y_wall),
                geo.win_c[1] + geo.r_win * np.sin(th),
                color=C_WIN, lw=3, zorder=5)
        # start marker (ringed dot, like the reference image's target-symbol)
        ax.scatter(*geo.start[:3], marker="o", s=110, facecolors="none",
                   edgecolors=C_START, linewidths=1.6, depthshade=False, zorder=6)
        ax.scatter(*geo.start[:3], marker=".", s=22, color=C_START,
                   depthshade=False, zorder=6)
        # target (filled black square)
        ax.scatter(*target, marker="s", s=90, color=C_TGT, edgecolor="white",
                   linewidth=0.6, depthshade=False, zorder=7)
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi); ax.set_zlim(zlo, zhi)
        try:
            ax.set_box_aspect((xhi - xlo, yhi - ylo, zhi - zlo))
        except Exception:
            pass
        ax.view_init(elev=22, azim=-60)
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_ticklabels([])
        ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
        ax.set_title(self.title, fontsize=12)

    def update(self, traj, step):
        for a in self.dyn:
            a.remove()
        self.dyn = []
        tr = np.asarray(traj)
        # trail
        (ln,) = self.ax.plot(tr[:, 0], tr[:, 1], tr[:, 2], "-", color=self.color,
                             lw=2.2, zorder=10)
        self.dyn.append(ln)
        # quadrotor at the current pose (black drone, like the reference image)
        self.dyn.extend(draw_drone(self.ax, tr[-1]))
        err = float(np.linalg.norm(tr[-1, :3] - self.target))
        self.ax.set_title(f"{self.title}\nstep {step + 1}   err {err:.3f}",
                          fontsize=11)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--target", type=int, default=2)
    ap.add_argument("--planner", choices=["bk_mbd", "bk_qp_sqp", "both"],
                    default="both")
    ap.add_argument("--steps", type=int, default=70,
                    help="closed-loop step cap for the live demo")
    ap.add_argument("--pause", type=float, default=0.03,
                    help="seconds to pause between rendered control steps")
    args = ap.parse_args()

    torch.set_num_threads(4)
    dw.STEPS = args.steps
    geo = dw.Geometry()
    # Sample IDENTICALLY to the batch experiment (same seed + count) so that
    # --target K is exactly experiment target K, consistent with the figure and
    # stats. target_x/y/z in config.json is the sampling box; EXP_SEED fixes the
    # draw. (Override geo.win_c / dw.MBD_* etc. here to try a new condition.)
    targets = geo.sample_targets(n=dw.N_TARGETS, seed=dw.EXP_SEED)
    if not 0 <= args.target < len(targets):
        raise SystemExit(f"--target must be in [0, {len(targets) - 1}]")
    target = np.asarray(targets[args.target], np.float64)
    print(f"target {args.target} = {np.round(target, 3)}  "
          f"(sampled from x{geo.target_x} y{geo.target_y} z{geo.target_z}, "
          f"seed {dw.EXP_SEED})", flush=True)
    model = dw.get_model(dw.SEED)

    # shared plot limits (start, target, window bbox, some wall span)
    span = np.array([[-0.45, geo.y_wall, 0.02], [0.45, 0.0, 1.0]])
    pts = np.vstack([geo.start[:3], target,
                     [geo.win_c[0] - geo.r_win, geo.y_wall, geo.win_c[1] - geo.r_win],
                     [geo.win_c[0] + geo.r_win, geo.y_wall, geo.win_c[1] + geo.r_win],
                     span])
    lo, hi = pts.min(0) - 0.05, pts.max(0) + 0.05
    lims = ((lo[0], hi[0]), (lo[1], hi[1]), (lo[2], hi[2]))

    plt.ion()
    show_both = args.planner == "both"
    ncol = 2 if show_both else 1
    fig = plt.figure(figsize=(6.4 * ncol, 5.6))
    fig.suptitle(f"Drone window pass — target {args.target} "
                 f"{np.round(target, 2)}", fontsize=12)

    panels = {}
    if args.planner in ("bk_mbd", "both"):
        ax = fig.add_subplot(1, ncol, 1, projection="3d")
        p = LivePanel(ax, geo, target, C_BK, "BK-MBD  (sampling)", lims)
        p.target = target
        panels["bk_mbd"] = p
    if args.planner in ("bk_qp_sqp", "both"):
        ax = fig.add_subplot(1, ncol, ncol, projection="3d")
        p = LivePanel(ax, geo, target, C_QP, "Convexified bilinear QP-MPC", lims)
        p.target = target
        panels["bk_qp_sqp"] = p
    fig.canvas.draw()
    plt.pause(0.2)

    def make_cb(panel):
        def cb(step, s, traj):
            panel.update(traj, step)
            fig.canvas.draw_idle()
            fig.canvas.flush_events()
            plt.pause(args.pause)
        return cb

    # animate BK-MBD first (fast), then QP-MPC (slow, visibly stalls)
    if "bk_mbd" in panels:
        print("running BK-MBD live...", flush=True)
        s, g, tr, st, ms = dw.run_mbd(model, geo, target, 20 + args.target,
                                      on_step=make_cb(panels["bk_mbd"]))
        print(f"  BK-MBD: err={np.linalg.norm(s[:3]-g):.3f} "
              f"viol={geo.executed_violation(tr)} steps={st}", flush=True)
    if "bk_qp_sqp" in panels:
        print("running QP-MPC live (slower)...", flush=True)
        s, g, tr, st, ms = dw.run_qp(model, geo, target,
                                     on_step=make_cb(panels["bk_qp_sqp"]))
        print(f"  QP-MPC: err={np.linalg.norm(s[:3]-g):.3f} "
              f"viol={geo.executed_violation(tr)} steps={st}", flush=True)

    print("done — close the window to exit.", flush=True)
    # Keep the window open after the run. plt.show() does not reliably block when
    # the process is launched detached (no controlling terminal), so hold the GUI
    # alive explicitly until the user closes the figure.
    try:
        while plt.fignum_exists(fig.number):
            plt.pause(0.2)
    except Exception:
        plt.show()


if __name__ == "__main__":
    main()
