"""Narrow-stall probe: is fixed-narrow sampling out of budget, or stuck on a
surrogate minimum?

Runs the trained bilinear model (no injected error, lam=0) under three
schedules over model_seeds x targets, recording the per-step tracking-error
curve and, at the settled state, the terminal error the model predicts against
the one the plant delivers.

    (method 1)  err_curve + plateau_step  -> did the run stop improving early?
    (method 3)  plan_end_pred vs true      -> did it stop where the model thinks
                                              it arrived but the plant says not?

    python -m exper.run_probe --config exp_c --set run.tag=probe
"""

from __future__ import annotations

import json
from pathlib import Path

import torch

from .backends import build_backend
from .config import RunLock, build_parser, describe, dump, load_config
from .planner import schedules_at_equal_budget, schedule_by_name
from .plant import FrankaPlant
from .trial import run_trial
from .training import get_model

SCHEDULES = ("staged_narrow", "staged_wide", "anneal")


def main() -> None:
    ap = build_parser(__doc__)
    cfg = load_config(**vars(ap.parse_args()))
    torch.set_num_threads(cfg.run.torch_threads)

    out_dir = Path(__file__).resolve().parent / cfg.run.out_dir / f"probe_{cfg.run.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    RunLock(out_dir).__enter__()
    dump(cfg, out_dir / "config.json")
    trials_path = out_dir / "trials.jsonl"
    print(describe(cfg), flush=True)

    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    all_sched = schedules_at_equal_budget(
        cfg.planner.num_samples, cfg.planner.stages,
        cfg.planner.sigma_start, cfg.planner.sigma_end)
    schedules = [schedule_by_name(all_sched, n) for n in SCHEDULES]

    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["schedule"], r["model_seed"], r["target_idx"]))

    total = len(schedules) * len(cfg.run.model_seeds) * len(targets)
    print(f"schedules {len(schedules)} x seeds {len(cfg.run.model_seeds)} "
          f"x targets {len(targets)} = {total} trials -> {trials_path}", flush=True)

    idx = 0
    with open(trials_path, "a") as fh:
        for seed in cfg.run.model_seeds:
            model = get_model(cfg, plant, "bilinear", seed)  # lam=0: trained as-is
            backend = build_backend("bilinear", plant, cfg.planner, model)
            for sched in schedules:
                for ti, goal in enumerate(targets):
                    idx += 1
                    if (sched.name, seed, ti) in done:
                        continue
                    rec = run_trial(cfg, plant, backend, sched, goal,
                                    condition="probe", model_seed=seed,
                                    target_idx=ti, lam=0.0, rng_seed=cfg.run.rng_base + ti,
                                    record_curve=True, probe_plan_end=True)
                    fh.write(json.dumps(rec.to_json()) + "\n")
                    fh.flush()
                    pe = (f" plateau@{rec.plateau_step}" if rec.plateau_step is not None
                          else " no-plateau")
                    pg = (f" pred={rec.plan_end_pred:.3f}/true={rec.plan_end_true:.3f}"
                          if rec.plan_end_pred is not None else "")
                    print(f"[{idx}/{total}] {sched.name:14s} s{seed} t{ti} "
                          f"final={rec.final_err:.4f}{pe}{pg} "
                          f"-> {'STRICT' if rec.reached_strict else 'miss'}", flush=True)

    print(f"\ndone -> {trials_path}", flush=True)


if __name__ == "__main__":
    main()
