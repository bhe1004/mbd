"""Entry point: run the wall-avoidance experiment with one planner.

    python secy/main.py --bk_mbd
    python secy/main.py --sqp_mpc
    python secy/main.py --bk_mbd --config path/to/other.json
    python secy/main.py --replay out/replays/bk_mbd_lockstep_YYYYMMDD_HHMMSS.npz

Every tunable lives in ``config.json`` -- there are no other flags. The two
mutually exclusive flags select which planner drives the same shared Scene.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from secy.config import DEFAULT_CONFIG, load_config  # noqa: E402
from secy.environment import Scene  # noqa: E402
from secy.bkmbd import BKMBDPlanner  # noqa: E402
from secy.sqp_mpc import SQPMPCPlanner  # noqa: E402
from secy.runtime import RealtimeRunner  # noqa: E402
from secy.replay import replay_trajectory  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--bk_mbd", action="store_true",
                       help="run the sampling BK-MBD planner")
    group.add_argument("--sqp_mpc", action="store_true",
                       help="run the convexified SQP-MPC planner")
    group.add_argument("--replay", type=Path,
                       help="replay a trajectory saved by a previous run")
    parser.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG,
                        help="config file; a bare name (e.g. config_2.json) is "
                             "resolved against the secy/ directory. "
                             f"default: {DEFAULT_CONFIG.name}")
    return parser.parse_args()


def resolve_config(path: Path) -> Path:
    """Accept a full/relative path, or a bare filename living next to the
    default config (the secy/ directory)."""

    if path.exists():
        return path
    candidate = DEFAULT_CONFIG.parent / path.name
    if candidate.exists():
        return candidate
    raise SystemExit(f"config not found: {path} (also tried {candidate})")


def main() -> None:
    args = parse_args()
    config_path = resolve_config(args.config)
    print(f"config: {config_path}")
    cfg = load_config(config_path)

    if cfg.runtime.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {cfg.runtime.device}, CUDA unavailable")
    device = torch.device(cfg.runtime.device)
    if cfg.runtime.torch_threads > 0:
        torch.set_num_threads(cfg.runtime.torch_threads)

    # ---- the shared world ------------------------------------------------
    scene = Scene(cfg, device=device)
    print(f"robot speed: joint-velocity limit {scene.task.config.action_limit} rad/s")
    print(f"planning horizon: {scene.horizon} steps "
          f"({scene.horizon * scene.period:.2f} s lookahead)")
    if cfg.env.start_tcp is not None:
        tip = np.asarray(scene.task.ee_of_q(scene.start_q))
        print(f"start pose: TCP target {np.round(np.asarray(cfg.env.start_tcp), 3)} "
              f"-> achieved {np.round(tip, 3)} "
              f"(err {np.linalg.norm(tip - np.asarray(cfg.env.start_tcp)) * 1000:.0f} mm)")
    if cfg.env.target is not None:
        print(f"target override: {np.round(np.asarray(cfg.env.target), 3)}")
    if scene.obstacle is not None:
        mode = "HARD (flat per overlap)" if cfg.collision.hard else "graded (depth^2)"
        print(f"penalty: {mode}, weight {cfg.collision.weight}, "
              f"margin {cfg.collision.margin}")
        for c, r in scene.obstacle.spheres_draw:
            print(f"obstacle: sphere at {np.round(c, 3)} radius {np.round(r, 3)}")
        for c, h in scene.obstacle.boxes_draw:
            print(f"obstacle: box at {np.round(c, 3)} half {np.round(h, 3)}")
        d0 = scene.start_clearance()
        print(f"whole arm = {scene.fk.num_points} spheres | "
              f"closest arm sphere at start {d0 * 1000:.0f} mm clear")
        if d0 <= 0:
            print("  WARNING: the arm already penetrates an obstacle at the start "
                  "pose (edit obstacles/start_tcp in config.json to separate them)")

    if args.replay is not None:
        print("mode=replay")
        replay_trajectory(scene, cfg, args.replay)
        return

    # ---- the chosen planner ----------------------------------------------
    if args.bk_mbd:
        planner = BKMBDPlanner(scene, cfg)
    else:
        planner = SQPMPCPlanner(scene, cfg)

    print(f"mode={cfg.runtime.mode} planner={planner.name} "
          f"targets={cfg.runtime.target_ids}")
    report = RealtimeRunner(scene, planner, cfg).run()
    print(report)


if __name__ == "__main__":
    main()
