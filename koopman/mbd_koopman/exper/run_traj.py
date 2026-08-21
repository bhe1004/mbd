"""Executed tool paths for the figure that compares rollout classes.

Four conditions of Sec. V-A run the reaching task on the same goal, from the
same home posture, on the same planner random stream, and the executed tool
center point is recorded at every control step. Nothing but the rollout backend
differs across them, so the four paths are one trial drawn four times.

    python -m exper.run_traj --config exp_b --targets 5 6 7
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .backends import build_backend
from .config import RunLock, build_parser, describe, dump, load_config
from .planner import Schedule
from .plant import FrankaPlant
from .trial import run_trial
from .training import get_model

# condition -> (model variant, backend kind); the names match Sec. V-A
CONDITIONS = [
    ("oracle", None, "oracle"),
    ("linear", "linear", "linear"),
    ("split", "linear", "split"),
    ("bilinear", "bilinear", "bilinear"),
]


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--targets", nargs="+", type=int, default=[5],
                    metavar="IDX", help="target indices to record")
    ap.add_argument("--model-seed", type=int, default=0)
    args = ap.parse_args()
    cfg = load_config(**vars(args))
    torch.set_num_threads(cfg.run.torch_threads)

    out_dir = Path(__file__).resolve().parent / cfg.run.out_dir / f"traj_{cfg.run.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out_dir).__enter__()
    dump(cfg, out_dir / "config.json")
    print(describe(cfg), flush=True)

    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    schedule = Schedule("anneal", cfg.planner.stages, cfg.planner.num_samples,
                        cfg.planner.sigma_start, cfg.planner.sigma_end)
    seed = args.model_seed

    models = {name: get_model(cfg, plant, variant, seed)
              for name, variant, _ in CONDITIONS if variant is not None}

    paths: dict[str, np.ndarray] = {}
    summary = []
    for ti in args.targets:
        goal = targets[ti]
        paths[f"goal_t{ti}"] = np.asarray(goal)
        for name, variant, kind in CONDITIONS:
            backend = build_backend(kind, plant, cfg.planner,
                                    models.get(name))
            path: list = []
            rec = run_trial(cfg, plant, backend, schedule, goal,
                            condition=name, model_seed=seed, target_idx=ti,
                            rng_seed=cfg.run.rng_base + ti,
                            record_curve=True, path_out=path)
            paths[f"{name}_t{ti}"] = np.asarray(path)
            paths[f"{name}_t{ti}_err"] = np.asarray(rec.err_curve)
            summary.append(rec.to_json())
            print(f"t{ti} {name:9s} final {rec.final_err:.4f} "
                  f"min {rec.min_err:.4f} strict {rec.reached_strict} "
                  f"{rec.ms_per_step:.1f} ms", flush=True)
            # written after every condition, so a long run can be drawn early
            np.savez(out_dir / "paths.npz", **paths)

    np.savez(out_dir / "paths.npz", **paths)
    (out_dir / "trials.json").write_text(json.dumps(summary, indent=1))
    print(f"wrote {out_dir/'paths.npz'}", flush=True)
    lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
