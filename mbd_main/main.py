"""mbd_main -- BK-MBD on the Franka FR3, end to end.

    python main.py collect       record the training dataset
    python main.py train         learn the bilinear Koopman model
    python main.py run           plan and execute in real time
    python main.py validate FILE score the model against a recorded trajectory
    python main.py view          ROS 2 only: mirror the robot, show the target
    python main.py home          ROS 2 only: go to the start pose safely
    python main.py replay FILE   play back a saved trajectory
    python main.py show          print the resolved config and exit

Every stage reads the same config file, so a run is described by that file
alone. Pick a different one with ``-c``, and override single values without
editing anything::

    python main.py run -c configs/interactive.yaml
    python main.py run -s run.mode=lockstep -s mbd.num_samples=256

The pipeline is three files on disk, and nothing else is shared between the
stages: ``collect`` writes a dataset, ``train`` turns it into a checkpoint,
``run`` loads the checkpoint. Any stage can be replaced -- record the dataset on
a real arm, train elsewhere, run against different hardware -- as long as the
file contracts hold.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config as config_module  # noqa: E402


def parse_args(argv=None) -> argparse.Namespace:
    # The shared options are attached to the top-level parser AND to every
    # subcommand, so both `main.py -c x.yaml run` and `main.py run -c x.yaml`
    # work. SUPPRESS keeps an unset copy from clobbering the set one.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("-c", "--config", type=Path, default=argparse.SUPPRESS,
                        help="config file (a bare name is looked up in configs/); "
                             f"default: {config_module.DEFAULT_CONFIG.name}")
    common.add_argument("-s", "--set", dest="overrides", action="append",
                        default=argparse.SUPPRESS, metavar="SECTION.KEY=VALUE",
                        help="override one config value; repeatable")

    parser = argparse.ArgumentParser(
        prog="main.py", description=__doc__, parents=[common],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="stage", required=True)
    sub.add_parser("collect", parents=[common],
                   help="record the Koopman training dataset")
    sub.add_parser("train", parents=[common], help="train the bilinear Koopman model")
    sub.add_parser("run", parents=[common], help="plan and execute in real time")
    sub.add_parser("show", parents=[common], help="print the resolved config and exit")
    validate = sub.add_parser("validate", parents=[common],
                              help="score the model against a recorded trajectory")
    validate.add_argument("file", type=Path, help="npz written by a previous run")
    sub.add_parser("view", parents=[common],
                   help="ROS 2 only: mirror the robot in a window of our own and "
                        "draw the target, obstacles and planned path")
    sub.add_parser("home", parents=[common],
                   help="ROS 2 only: move the arm to the start pose with the "
                        "point-to-point controller, then hand it back")
    replay = sub.add_parser("replay", parents=[common],
                            help="play back a saved trajectory")
    replay.add_argument("file", type=Path, help="npz written by a previous run")
    replay.add_argument("--speed", type=float, default=1.0, help="playback speed factor")
    replay.add_argument("--loop", action="store_true", help="repeat until closed")

    args = parser.parse_args(argv)
    args.config = getattr(args, "config", None)
    args.overrides = getattr(args, "overrides", [])
    return args


def main(argv=None) -> None:
    args = parse_args(argv)
    cfg = config_module.load(args.config, args.overrides)

    if args.stage == "show":
        print(f"# resolved from {cfg.source}")
        print(config_module.describe(cfg))
        return

    viewer_used = (args.stage == "view"
                   or (args.stage in ("run", "replay") and cfg.viewer.enabled))

    if args.stage == "collect":
        from pipeline import collect
        collect.run(cfg)
    elif args.stage == "train":
        from pipeline import train
        train.run(cfg)
    elif args.stage == "run":
        from pipeline import run as run_stage
        run_stage.run(cfg)
    elif args.stage == "validate":
        from pipeline import validate as validate_stage
        validate_stage.run(cfg, args.file)
    elif args.stage == "view":
        from pipeline import view
        view.run(cfg)
    elif args.stage == "home":
        from pipeline import home
        home.run(cfg)
    elif args.stage == "replay":
        from pipeline import replay
        replay.run(cfg, args.file, speed=args.speed, loop=args.loop)

    if viewer_used:
        _exit_after_viewer()


def _exit_after_viewer() -> None:
    """Leave without running the interpreter's teardown, after a viewer run.

    MuJoCo 3.9's ``launch_passive`` segfaults while its GL context is torn down
    at interpreter shutdown on this platform -- reproducibly, with no planner or
    project code involved (launch, sync a few frames, exit). It happens after
    the run is complete and everything is printed, but it still leaves a core
    dump and a 139 exit status, which would break any script that checks it.
    Flushing and exiting immediately skips that teardown. Remove this once the
    upstream viewer shuts down cleanly.
    """

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
