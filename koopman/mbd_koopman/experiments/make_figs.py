"""Generate comparison figures from saved experiment outputs.

Unicycle (`--task unicycle`, default): reads
`out/unicycle/{method}_case{c}_seed{s}.npz` and writes

- `unicycle_trajectories.png`: per-case x-y trajectories of all methods,
- `unicycle_final_error.png`: final position error bar plot per case.

Arm (`--task arm`): reads `out/arm/{method}_target{t}_seed{s}.npz` and writes

- `arm_final_error.png`: final end-effector error bar plot per target
  (mean +/- std over seeds), in the style of the reference fig_arm_bar_v2.

Franka (`--task franka`): same three figures as the arm from
`out/franka/{method}_target{t}_seed{s}.npz`, with the TCP path / kinematic
chain evaluated through the MuJoCo model.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.arm import ArmTask, forward_kinematics_np, link_positions_np  # noqa: E402
from envs.unicycle import UnicycleTask  # noqa: E402

METHOD_STYLE = {
    "vanilla_mbd_true": {"color": "#2ca02c", "label": "MBD (true dynamics)"},
    "dk_mbd": {"color": "#1f77b4", "label": "DK-MBD (linear)"},
    "dk_mbd_split": {
        "color": "#ff7f0e",
        "label": "DK-MBD-split (structure-informed)",
    },
    "bk_mbd": {"color": "#d62728", "label": "BK-MBD (bilinear + tube)"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--task", choices=["unicycle", "arm", "franka"], default="unicycle"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--cases", type=int, nargs="+", default=None)
    parser.add_argument("--input-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--pose-target",
        type=int,
        default=0,
        help="arm only: target index for the final-pose panel figure",
    )
    args = parser.parse_args()
    if args.cases is None:
        args.cases = [0, 1, 2, 3] if args.task == "unicycle" else list(range(7))
    if args.input_dir is None:
        args.input_dir = PROJECT_ROOT / "out" / args.task
    return args


def load_trajectory(
    input_dir: Path, method: str, case_id: int, seed: int, stem: str = "case"
):
    path = input_dir / f"{method}_{stem}{case_id}_seed{seed}.npz"
    if not path.exists():
        return None
    with np.load(path) as data:
        return data["trajectory"]


def plot_trajectories(task: UnicycleTask, args: argparse.Namespace, out_dir: Path) -> None:
    num_cases = len(args.cases)
    ncols = min(num_cases, 2)
    nrows = (num_cases + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.0 * nrows))
    axes = np.atleast_1d(axes).ravel()

    multi = len(args.seeds) > 1
    for ax_idx, case_id in enumerate(args.cases):
        ax = axes[ax_idx]
        start, goal, _ = task.case(case_id)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for method, style in METHOD_STYLE.items():
            for si, seed in enumerate(args.seeds):
                traj = load_trajectory(args.input_dir, method, case_id, seed)
                if traj is None:
                    continue
                ax.plot(
                    traj[:, 0],
                    traj[:, 1],
                    lw=1.2 if multi else 1.8,
                    alpha=0.55 if multi else 1.0,
                    color=style["color"],
                    label=style["label"] if si == 0 else None,
                )
                xs.append(traj[:, 0])
                ys.append(traj[:, 1])
        park = plt.Circle(
            (goal[0], goal[1]),
            task.config.park_radius,
            color="gray",
            alpha=0.25,
            zorder=0,
        )
        ax.add_patch(park)
        ax.plot(*start[:2], marker="o", color="black", ms=7, zorder=5)
        ax.plot(*goal[:2], marker="*", color="black", ms=13, zorder=5)
        ax.set_title(f"case {case_id}  start=({start[0]:.1f}, {start[1]:.1f})")
        ax.set_xlabel("x [m]")
        ax.set_ylabel("y [m]")
        # Tightest square window containing every trajectory, the start,
        # and the park disk, so all panels render as identical boxes with
        # nothing clipped.
        r = task.config.park_radius
        x_all = np.concatenate(xs + [np.array([start[0], goal[0] - r, goal[0] + r])])
        y_all = np.concatenate(ys + [np.array([start[1], goal[1] - r, goal[1] + r])])
        cx = 0.5 * (x_all.min() + x_all.max())
        cy = 0.5 * (y_all.min() + y_all.max())
        half = 0.5 * max(x_all.max() - x_all.min(), y_all.max() - y_all.min())
        half *= 1.06
        ax.set_xlim(cx - half, cx + half)
        ax.set_ylim(cy - half, cy + half)
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    for ax in axes[num_cases:]:
        ax.axis("off")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels, loc="lower center", ncol=2, fontsize=11.7, frameon=False
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = out_dir / "unicycle_trajectories.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved={path}")


def plot_final_errors(task: UnicycleTask, args: argparse.Namespace, out_dir: Path) -> None:
    methods = list(METHOD_STYLE)
    errors = np.full((len(methods), len(args.cases), len(args.seeds)), np.nan)
    for mi, method in enumerate(methods):
        for ci, case_id in enumerate(args.cases):
            for si, seed in enumerate(args.seeds):
                traj = load_trajectory(args.input_dir, method, case_id, seed)
                if traj is not None:
                    goal = task.case(case_id)[1]
                    errors[mi, ci, si] = task.final_error(traj[-1], goal)

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    width = 0.8 / len(methods)
    xs = np.arange(len(args.cases))
    multi = len(args.seeds) > 1
    for mi, method in enumerate(methods):
        style = METHOD_STYLE[method]
        mean = np.nanmean(errors[mi], axis=-1)
        std = np.nanstd(errors[mi], axis=-1)
        ax.bar(
            xs + (mi - (len(methods) - 1) / 2) * width,
            mean,
            width,
            yerr=std if multi else None,
            capsize=3 if multi else 0,
            color=style["color"],
            label=style["label"],
        )
    ax.axhline(
        task.config.park_radius, ls="--", color="gray", lw=1.2,
        label=f"park radius ({task.config.park_radius} m)",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"case {c}" for c in args.cases])
    ax.set_ylabel("final position error [m]")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(fontsize=8)

    fig.tight_layout()
    path = out_dir / "unicycle_final_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved={path}")


def plot_arm_trajectories(
    task, args: argparse.Namespace, out_dir: Path, kin, name: str
) -> None:
    """3D end-effector paths per target for all methods (FK of joint trajectories)."""

    num_cases = len(args.cases)
    ncols = min(num_cases, 4)
    nrows = (num_cases + ncols - 1) // ncols
    fig = plt.figure(figsize=(3.6 * ncols, 3.4 * nrows))
    multi = len(args.seeds) > 1

    for ax_idx, target_id in enumerate(args.cases):
        ax = fig.add_subplot(nrows, ncols, ax_idx + 1, projection="3d")
        start_state, goal, _ = task.case(target_id)
        start_ee = kin["ee_path"](start_state[None])[0]
        for method, style in METHOD_STYLE.items():
            for si, seed in enumerate(args.seeds):
                traj = load_trajectory(
                    args.input_dir, method, target_id, seed, stem="target"
                )
                if traj is None:
                    continue
                ee = kin["ee_path"](traj)
                ax.plot(
                    ee[:, 0],
                    ee[:, 1],
                    ee[:, 2],
                    lw=1.2 if multi else 2.2,
                    alpha=0.5 if multi else 0.95,
                    color=style["color"],
                    label=style["label"] if si == 0 else None,
                )
        ax.scatter(*start_ee, color="black", marker="o", s=40, zorder=6)
        ax.scatter(*goal, color="black", marker="*", s=150, zorder=6)
        ax.set_title(
            f"target {ax_idx + 1}  ({goal[0]:.2f}, {goal[1]:.2f}, {goal[2]:.2f})",
            fontsize=10,
        )
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)
        ax.tick_params(labelsize=7)

    # Figure-level legend along the top so it never overlaps a panel.
    handles, labels = fig.axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles, labels, loc="upper center", ncol=len(handles),
            fontsize=9, frameon=False, bbox_to_anchor=(0.5, 1.02),
        )

    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    path = out_dir / f"{name}_trajectories.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved={path}")


POSE_PANELS = [
    ("bk_mbd", "Ours: BK-MBD (bilinear)"),
    ("vanilla_mbd_true", "MBD-true: oracle"),
    ("dk_mbd_split", "DK-MBD-split: structure-informed"),
    ("dk_mbd", "DK-MBD: linear Koopman"),
]


def plot_arm_final_pose(
    task, args: argparse.Namespace, out_dir: Path, kin, name: str
) -> None:
    """One 3D panel per method for a single target: final robot pose plus the
    TCP path.

    The kinematic chain is drawn in the method color (bold, with joint
    markers) and the tool trajectory as a dotted line, over the full arm
    workspace, so the reader sees where each method left the arm relative to
    the target ($\\star$).
    """

    from matplotlib.ticker import MaxNLocator

    target_id = args.pose_target
    seed = args.seeds[0]
    _, goal, _ = task.case(target_id)

    fig = plt.figure(figsize=(3.6 * len(POSE_PANELS), 3.7))
    for panel_idx, (method, title) in enumerate(POSE_PANELS):
        ax = fig.add_subplot(1, len(POSE_PANELS), panel_idx + 1, projection="3d")
        color = METHOD_STYLE[method]["color"]

        traj = load_trajectory(args.input_dir, method, target_id, seed, stem="target")
        if traj is None:
            ax.set_title(f"{title}\n(no data)", fontsize=9)
            continue

        # TCP path: dotted line in the method color.
        ee_path = kin["ee_path"](traj)
        ax.plot(
            ee_path[:, 0], ee_path[:, 1], ee_path[:, 2],
            ls=":", lw=1.4, alpha=0.8, color=color, zorder=3,
        )
        # Final kinematic chain: bold method-color links with joint markers.
        links = kin["links"](traj[-1])
        ax.plot(
            links[:, 0], links[:, 1], links[:, 2],
            "-o", lw=2.6, ms=4, color=color, zorder=4,
        )
        ax.scatter(*goal, color="black", marker="*", s=170, zorder=6)

        steps = traj.shape[0] - 1
        ee_err = task.final_error(traj[-1], goal)
        ax.set_title(f"{title}\nstep {steps}  ee-err {ee_err:.3f}", fontsize=9)
        ax.set_xlabel("x [m]", fontsize=8)
        ax.set_ylabel("y [m]", fontsize=8)
        ax.set_zlabel("z [m]", fontsize=8)
        ax.set_xlim(*kin["xlim"])
        ax.set_ylim(*kin["ylim"])
        ax.set_zlim(*kin["zlim"])
        dx = kin["xlim"][1] - kin["xlim"][0]
        dy = kin["ylim"][1] - kin["ylim"][0]
        dz = kin["zlim"][1] - kin["zlim"][0]
        ax.set_box_aspect((dx, dy, dz))
        # Same viewpoint as the trajectory-overview figure (Fig. 3) so the
        # TCP path reads the same way in both.
        ax.view_init(elev=30, azim=-60)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_locator(MaxNLocator(4))
        ax.yaxis.set_major_locator(MaxNLocator(4))
        ax.zaxis.set_major_locator(MaxNLocator(5))

    fig.subplots_adjust(left=0.0, right=1.0, bottom=0.0, top=0.88, wspace=0.02)
    path = out_dir / f"{name}_final_pose_T{target_id + 1}_seed{seed}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.05)
    plt.close(fig)
    print(f"saved={path}")


def plot_arm_final_errors(
    task, args: argparse.Namespace, out_dir: Path, name: str
) -> None:
    """Final EE error bar plot per target, mean +/- std over seeds."""

    methods = list(METHOD_STYLE)
    errors = np.full((len(methods), len(args.cases), len(args.seeds)), np.nan)
    for mi, method in enumerate(methods):
        for ci, target_id in enumerate(args.cases):
            goal = task.case(target_id)[1]
            for si, seed in enumerate(args.seeds):
                traj = load_trajectory(
                    args.input_dir, method, target_id, seed, stem="target"
                )
                if traj is not None:
                    errors[mi, ci, si] = task.final_error(traj[-1], goal)

    fig, ax = plt.subplots(figsize=(7.2, 3.2))
    width = 0.8 / len(methods)
    xs = np.arange(len(args.cases))
    multi = len(args.seeds) > 1
    for mi, method in enumerate(methods):
        style = METHOD_STYLE[method]
        mean = np.nanmean(errors[mi], axis=-1)
        std = np.nanstd(errors[mi], axis=-1)
        ax.bar(
            xs + (mi - (len(methods) - 1) / 2) * width,
            mean,
            width,
            yerr=std if multi else None,
            capsize=2 if multi else 0,
            color=style["color"],
            label=style["label"],
            error_kw=dict(lw=1, ecolor="0.3"),
        )
    thr = task.config.reach_threshold
    ax.axhline(thr, ls="--", color="k", lw=1.2)
    ax.annotate(
        f"reach threshold ({thr} m)",
        xy=(xs[-1] + 0.4, thr),
        xytext=(xs[-1] + 0.4, thr * 2.5),
        ha="right",
        fontsize=9,
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([f"T{c + 1}" for c in args.cases])
    ax.set_xlabel("reach target")
    ax.set_ylabel("final EE error [m]")
    ax.set_yscale("log")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper left", fontsize=8)

    fig.tight_layout()
    path = out_dir / f"{name}_final_error.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    print(f"saved={path}")


def make_arm_kinematics():
    return {
        "ee_path": forward_kinematics_np,
        "links": link_positions_np,
        "xlim": (-0.7, 0.7),
        "ylim": (-0.7, 0.7),
        "zlim": (0.0, 1.25),
    }


def make_franka_kinematics(task):
    import mujoco

    from envs.franka import NUM_JOINTS

    data = mujoco.MjData(task.model)
    chain_bodies = [
        task.model.body(f"fr3_link{i}").id for i in range(8)
    ] + [task.model.body("hand").id]

    def ee_path(traj):
        traj = np.atleast_2d(np.asarray(traj))
        return np.stack([task.ee_of_q(s[:NUM_JOINTS]) for s in traj])

    def links(state):
        data.qpos[:NUM_JOINTS] = np.asarray(state)[:NUM_JOINTS]
        data.qvel[:] = 0.0
        mujoco.mj_forward(task.model, data)
        pts = [data.xpos[b].copy() for b in chain_bodies]
        pts.append(data.site_xpos[task._ee_site].copy())
        return np.stack(pts)

    return {
        "ee_path": ee_path,
        "links": links,
        "xlim": (-0.05, 0.6),
        "ylim": (-0.05, 0.5),
        "zlim": (0.0, 0.75),
    }


def main() -> None:
    args = parse_args()
    out_dir = args.output_dir or (args.input_dir / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.task == "unicycle":
        task = UnicycleTask()
        plot_trajectories(task, args, out_dir)
        plot_final_errors(task, args, out_dir)
    else:
        if args.task == "arm":
            task = ArmTask()
            kin = make_arm_kinematics()
        else:
            from envs.franka import FrankaTask

            task = FrankaTask()
            kin = make_franka_kinematics(task)
        plot_arm_trajectories(task, args, out_dir, kin, args.task)
        plot_arm_final_errors(task, args, out_dir, args.task)
        plot_arm_final_pose(task, args, out_dir, kin, args.task)


if __name__ == "__main__":
    main()
