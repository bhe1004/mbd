"""Schedule study on the true-dynamics rollout.

The schedule study of Sec. IV-C varies the noise level on a learned rollout
only, so it cannot separate a schedule that fails because it is narrow from one
that fails because it settles on a minimum of the surrogate. This runner repeats
a schedule on the oracle backend, where the rollout carries no model error by
construction, over the same targets and the same planner random stream, so the
two runs are paired trial by trial.

    python -m exper.run_oracle_schedule --config exp_c --schedule staged_narrow
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .backends import build_backend
from .config import RunLock, build_parser, describe, dump, load_config
from .planner import schedule_by_name, schedules_at_equal_budget
from .plant import FrankaPlant
from .trial import run_trial


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--schedule", default="staged_narrow", metavar="NAME",
                    help="schedule to run on the oracle (default: staged_narrow)")
    ap.add_argument("--rng-offsets", type=int, nargs="+", default=[0],
                    metavar="K",
                    help="extra planner streams, as offsets added to rng_base; "
                         "more offsets buy power at a linear cost in time")
    args = ap.parse_args()
    cfg = load_config(**{k: v for k, v in vars(args).items()
                         if k not in ("schedule", "rng_offsets")})
    torch.set_num_threads(cfg.run.torch_threads)

    out_dir = Path(__file__).resolve().parent / cfg.run.out_dir / f"o_{cfg.run.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out_dir).__enter__()
    dump(cfg, out_dir / "config.json")
    trials_path = out_dir / "trials.jsonl"
    print(describe(cfg), flush=True)

    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    schedules = schedules_at_equal_budget(
        cfg.planner.num_samples, cfg.planner.stages,
        cfg.planner.sigma_start, cfg.planner.sigma_end)
    sched = schedule_by_name(schedules, args.schedule)
    backend = build_backend("oracle", plant, cfg.planner)

    # The oracle carries no model seed, so a trial is identified by its target
    # and its planner stream; the offset shifts the stream to add trials.
    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["schedule"], r["model_seed"], r["target_idx"]))

    total = len(args.rng_offsets) * len(targets)
    print(f"oracle x {sched.name}: sigma {sched.sigma_start}->{sched.sigma_end}, "
          f"{sched.stages} x {sched.num_samples} = {sched.total_samples} samples, "
          f"{len(args.rng_offsets)} streams x {len(targets)} targets = {total} trials",
          flush=True)

    idx = 0
    with open(trials_path, "a") as fh:
        for off in args.rng_offsets:
            for ti, goal in enumerate(targets):
                idx += 1
                if (sched.name, off, ti) in done:
                    continue
                rec = run_trial(cfg, plant, backend, sched, goal,
                                condition="oracle", model_seed=off,
                                target_idx=ti,
                                rng_seed=cfg.run.rng_base + off * 100 + ti)
                fh.write(json.dumps(rec.to_json()) + "\n")
                fh.flush()
                print(f"[{idx}/{total}] oracle {sched.name:14s} off{off} t{ti} "
                      f"err={rec.final_err:.4f} steps={rec.steps} "
                      f"plateau={rec.plateau_step} "
                      f"-> {'STRICT' if rec.reached_strict else ('reach' if rec.reached else 'miss')}",
                      flush=True)

    rows = [json.loads(l) for l in trials_path.read_text().splitlines() if l.strip()]
    strict = sum(bool(r["reached_strict"]) for r in rows)
    reach = sum(bool(r["reached"]) for r in rows)
    print(f"\noracle x {sched.name}: strict {strict}/{len(rows)}, "
          f"reach {reach}/{len(rows)}, "
          f"median final err {np.median([r['final_err'] for r in rows]):.4f} m",
          flush=True)
    print(f"done -> {trials_path}", flush=True)


if __name__ == "__main__":
    main()
