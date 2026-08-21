"""The oracle on a GPU: how much of the latency gap hardware closes.

Runs \\textsc{MBD-true} with the true-dynamics rollout batched through MJX
instead of threaded MuJoCo, on the goals, the schedule and the planner stream of
the CPU study, so the only change is the machine the physics runs on.

    python -m exper.run_gpu --config exp_b

Reports the same quantities as ``run_b``: reaching accuracy, planning time per
control step, and deadline misses. It also records how far the single-precision
GPU rollout drifts from the double-precision CPU one, since the comparison is
only fair if the two are the same plant.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .backends import build_backend
from .backends_mjx import MjxOracleBackend, agreement
from .config import RunLock, build_parser, describe, dump, load_config
from .planner import Schedule
from .plant import FrankaPlant
from .trial import run_trial


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--targets", nargs="+", type=int, default=None,
                    metavar="IDX", help="restrict to these target indices")
    ap.add_argument("--skip-agreement", action="store_true",
                    help="do not check the GPU rollout against the CPU one")
    args = ap.parse_args()
    cfg = load_config(**vars(args))

    out_dir = Path(__file__).resolve().parent / cfg.run.out_dir / f"gpu_{cfg.run.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out_dir).__enter__()
    dump(cfg, out_dir / "config.json")
    print(describe(cfg), flush=True)

    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    picked = args.targets if args.targets is not None else range(len(targets))
    schedule = Schedule("anneal", cfg.planner.stages, cfg.planner.num_samples,
                        cfg.planner.sigma_start, cfg.planner.sigma_end)

    backend = MjxOracleBackend(plant, cfg.planner,
                               num_candidates=cfg.planner.num_samples,
                               horizon=cfg.planner.horizon)
    compile_s = backend.warmup()
    print(f"MJX rollout compiled in {compile_s:.1f} s "
          f"(N={cfg.planner.num_samples}, T={cfg.planner.horizon}, "
          f"nsub={backend.nsub}, {cfg.planner.stages * cfg.planner.horizon * backend.nsub}"
          f" sequential substeps per control step)", flush=True)

    report: dict = {"compile_s": compile_s}
    if not args.skip_agreement:
        rng = np.random.default_rng(0)
        probe = np.clip(rng.normal(0, 0.6, (64, cfg.planner.horizon, plant.act_dim)),
                        -plant.limit, plant.limit)
        report["agreement"] = agreement(plant, backend, probe)
        print(f"  GPU vs CPU rollout: {report['agreement']}", flush=True)

    trials_path = out_dir / "trials.jsonl"
    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                done.add(json.loads(line)["target_idx"])

    with open(trials_path, "a") as fh:
        for ti in picked:
            if ti in done:
                continue
            rec = run_trial(cfg, plant, backend, schedule, targets[ti],
                            condition="oracle_gpu", model_seed=0, target_idx=ti,
                            rng_seed=cfg.run.rng_base + ti, record_curve=True)
            fh.write(json.dumps(rec.to_json()) + "\n")
            fh.flush()
            print(f"t{ti} final {rec.final_err:.4f} min {rec.min_err:.4f} "
                  f"strict {rec.reached_strict} {rec.ms_per_step:.1f} ms "
                  f"worst {rec.worst_ms:.1f} misses {rec.deadline_misses}",
                  flush=True)

    (out_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"\ndone -> {trials_path}", flush=True)
    lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
