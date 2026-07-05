"""Replay a saved FR3 trajectory in the MuJoCo viewer.

Loads `out/franka/{method}_target{t}_seed{s}.npz` and plays the closed-loop
joint trajectory in real time, with the reach target drawn as a green sphere
(and the reach threshold as a translucent shell). The tool-center-point (TCP)
path is drawn as a colored trail as the motion plays; `--ghost` additionally
leaves translucent snapshots of the whole arm behind the motion
(afterimages), so a single screenshot shows the entire reach.

Examples:

    python experiments/view_franka.py --method bk_mbd --target-id 0
    python experiments/view_franka.py --method dk_mbd --target-id 3 --speed 0.5
    python experiments/view_franka.py --compare bk_mbd dk_mbd --target-id 0
        (plays the methods one after another, looping)
    python experiments/view_franka.py --method bk_mbd --ghost --ghost-every 2
        (motion afterimages every 2 control steps)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.franka import NUM_JOINTS, SCENE_XML_PATH, FrankaTask  # noqa: E402

ROBOT_XML_PATH = SCENE_XML_PATH.parent / "fr3_hand_velocity.xml"

# Trail colors matched to the paper figures.
METHOD_COLORS = {
    "mbd_true": (0.13, 0.65, 0.22),
    "dk_mbd": (0.12, 0.47, 0.71),
    "dk_mbd_split": (1.00, 0.50, 0.05),
    "bk_mbd": (0.84, 0.15, 0.16),
}
DEFAULT_COLOR = (0.6, 0.6, 0.6)
GHOST_RGB = (0.85, 0.88, 0.92)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method", default="bk_mbd")
    parser.add_argument(
        "--compare",
        nargs="+",
        default=None,
        help="play several methods back to back (overrides --method)",
    )
    parser.add_argument("--target-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--speed", type=float, default=1.0, help="playback speed")
    parser.add_argument("--input-dir", type=Path, default=PROJECT_ROOT / "out" / "franka")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.001,
        help="success threshold [m] for the reach verdict and the "
        "translucent shell (visualization only; the task config is untouched)",
    )
    parser.add_argument(
        "--no-trail",
        action="store_true",
        help="disable the TCP trajectory trail",
    )
    parser.add_argument(
        "--trail-radius",
        type=float,
        default=0.002,
        help="radius [m] of the TCP trail capsules",
    )
    parser.add_argument(
        "--ghost",
        action="store_true",
        help="leave translucent arm snapshots behind the motion (afterimages)",
    )
    parser.add_argument(
        "--ghost-every",
        type=int,
        default=3,
        help="control steps between ghost snapshots",
    )
    parser.add_argument(
        "--ghost-alpha",
        type=float,
        default=0.22,
        help="opacity of the newest ghost (older ones fade toward 35%% of this)",
    )
    return parser.parse_args()


def load_trajectory(input_dir: Path, method: str, target_id: int, seed: int):
    path = input_dir / f"{method}_target{target_id}_seed{seed}.npz"
    if not path.exists():
        raise SystemExit(f"trajectory not found: {path}")
    with np.load(path) as data:
        return data["trajectory"]


def snapshot_indices(traj_len: int, every: int) -> list[int]:
    """Snapshot steps for the afterimages: every K steps plus the final pose."""
    idx = list(range(0, traj_len, max(every, 1)))
    if idx[-1] != traj_len - 1:
        idx.append(traj_len - 1)
    return idx


def build_model(n_ghosts: int):
    """Compile the visualization scene, optionally with ghost arm copies.

    Each ghost is a full kinematic copy of the arm attached to the world with
    prefixed names, rendered flat translucent and excluded from collisions.
    Returns (model, main_qadr, main_dofadr, ghosts) where each ghost is a
    dict with its 7 qpos addresses and its geom ids.
    """
    if n_ghosts == 0:
        model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
    else:
        spec = mujoco.MjSpec.from_file(str(SCENE_XML_PATH))
        for k in range(n_ghosts):
            child = mujoco.MjSpec.from_file(str(ROBOT_XML_PATH))
            frame = spec.worldbody.add_frame()
            frame.attach_body(child.bodies[1], f"g{k:03d}_", "")
        model = spec.compile()

    main_qadr = np.array(
        [model.joint(f"fr3_joint{i}").qposadr[0] for i in range(1, NUM_JOINTS + 1)]
    )
    main_dofadr = np.array(
        [model.joint(f"fr3_joint{i}").dofadr[0] for i in range(1, NUM_JOINTS + 1)]
    )

    ghosts = []
    for k in range(n_ghosts):
        prefix = f"g{k:03d}_"
        qadr = np.array(
            [
                model.joint(f"{prefix}fr3_joint{i}").qposadr[0]
                for i in range(1, NUM_JOINTS + 1)
            ]
        )
        geoms = np.array(
            [
                g
                for g in range(model.ngeom)
                if model.body(model.geom_bodyid[g]).name.startswith(prefix)
            ]
        )
        # Flat translucent rendering, no collisions; hidden until revealed.
        model.geom_matid[geoms] = -1
        model.geom_rgba[geoms] = (*GHOST_RGB, 0.0)
        model.geom_contype[geoms] = 0
        model.geom_conaffinity[geoms] = 0
        ghosts.append({"qadr": qadr, "geoms": geoms})
    return model, main_qadr, main_dofadr, ghosts


def draw_user_scene(
    viewer,
    target: np.ndarray,
    threshold: float,
    trail: np.ndarray | None,
    upto: int,
    color: tuple,
    radius: float,
    clear: bool = True,
) -> None:
    """Rebuild the overlay: target markers plus the TCP trail up to `upto`.

    `clear=False` appends to an existing scene instead (offscreen rendering,
    where the overlay shares the scene with the model geoms).
    """
    scn = viewer.user_scn
    if clear:
        scn.ngeom = 0

    def add_sphere(pos, size, rgba):
        mujoco.mjv_initGeom(
            scn.geoms[scn.ngeom],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[size, 0, 0],
            pos=pos,
            mat=np.eye(3).flatten(),
            rgba=np.asarray(rgba, dtype=np.float32),
        )
        scn.ngeom += 1

    add_sphere(target, min(0.015, 0.6 * threshold), [0.1, 0.8, 0.1, 1.0])
    add_sphere(target, threshold, [0.1, 0.8, 0.1, 0.15])

    if trail is None or upto < 1:
        return
    # Subsample if the trail would overflow the scene's geom budget.
    budget = scn.maxgeom - scn.ngeom - 2
    stride = max(1, int(np.ceil(upto / max(budget, 1))))
    rgba = np.array([*color, 0.9], dtype=np.float32)
    prev = trail[0]
    for j in range(stride, upto + 1, stride):
        geom = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(
            geom,
            type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=rgba,
        )
        mujoco.mjv_connector(
            geom, mujoco.mjtGeom.mjGEOM_CAPSULE, radius, prev, trail[j]
        )
        scn.ngeom += 1
        prev = trail[j]
    add_sphere(trail[upto], max(radius * 2.5, 0.006), [*color, 1.0])


def main() -> None:
    args = parse_args()
    methods = args.compare if args.compare else [args.method]

    task = FrankaTask()
    _, target, _ = task.case(args.target_id)
    trajectories = {
        m: load_trajectory(args.input_dir, m, args.target_id, args.seed)
        for m in methods
    }
    tcp_paths = {
        m: np.array([task.ee_of_q(s[:NUM_JOINTS]) for s in traj])
        for m, traj in trajectories.items()
    }
    snapshots = {
        m: snapshot_indices(traj.shape[0], args.ghost_every)
        for m, traj in trajectories.items()
    }
    for m, traj in trajectories.items():
        q_final = traj[-1][:NUM_JOINTS]
        err = np.linalg.norm(task.ee_of_q(q_final) - target)
        print(
            f"{m}: {traj.shape[0] - 1} steps, final EE error "
            f"{err:.4f} m ({'reach' if err < args.threshold else 'FAIL'}"
            f" @ {args.threshold * 1000:g} mm)"
        )

    # Visualization scene (floor / skybox / lighting); same joint layout as
    # the task model, cosmetics only. Ghost copies are attached on demand.
    n_ghosts = max(len(s) for s in snapshots.values()) if args.ghost else 0
    model, main_qadr, main_dofadr, ghosts = build_model(n_ghosts)
    data = mujoco.MjData(model)
    dt = task.config.control_dt / max(args.speed, 1e-3)
    if args.ghost:
        print(f"ghosts: {n_ghosts} snapshots (every {args.ghost_every} steps)")

    def set_ghosts_for(m: str) -> list[int]:
        """Pose the ghosts on this method's snapshots and hide them all."""
        snap = snapshots[m]
        traj = trajectories[m]
        for g, ghost in enumerate(ghosts):
            if g < len(snap):
                data.qpos[ghost["qadr"]] = traj[snap[g]][:NUM_JOINTS]
            model.geom_rgba[ghost["geoms"], 3] = 0.0
        return snap

    print("\nviewer: ESC to quit; playback loops over methods")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            for m in methods:
                traj = trajectories[m]
                trail = None if args.no_trail else tcp_paths[m]
                color = METHOD_COLORS.get(m, DEFAULT_COLOR)
                snap = set_ghosts_for(m) if args.ghost else []
                print(f"  playing {m} ...")
                for i, s in enumerate(traj):
                    if not viewer.is_running():
                        return
                    data.qpos[main_qadr] = s[:NUM_JOINTS]
                    data.qvel[main_dofadr] = s[NUM_JOINTS:]
                    # Reveal the afterimages the motion has passed, older
                    # ones fading toward 35% of the newest one's opacity.
                    for g, step in enumerate(snap):
                        if step <= i:
                            age = 1.0 - (g / max(len(snap) - 1, 1))
                            alpha = args.ghost_alpha * (1.0 - 0.65 * age)
                            model.geom_rgba[ghosts[g]["geoms"], 3] = alpha
                    draw_user_scene(
                        viewer,
                        target,
                        args.threshold,
                        trail,
                        i,
                        color,
                        args.trail_radius,
                    )
                    mujoco.mj_forward(model, data)
                    viewer.sync()
                    time.sleep(dt)
                time.sleep(0.8)


if __name__ == "__main__":
    main()
