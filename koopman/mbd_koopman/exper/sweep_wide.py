"""Does a wider wide stage separate the annealed schedule from fixed-wide?

At sigma_start = 1.2 the two are indistinguishable on success (McNemar p = 0.25)
and differ only in how fast they enter the tolerance band. Reach is bounded by
sigma sqrt(2 log(N/rho)) before projection, so a wider stage should buy more of
it -- except that the proposals are clipped to the action box, and at sigma=1.2
against a limit of 1.5 a fifth of the population already saturates. This sweep
separates the two knobs:

    sigma_start   how far the proposals are drawn
    action_limit  how much of that draw survives the projection

Each action limit gets its own excitation and its own identified model, tagged by
the limit, because plant.excite() draws its training inputs within the same box.
Reusing a model fitted at +-1.5 outside that box would confound reach with
extrapolation error.

    python -m exper.sweep_wide --config exp_b
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .backends import build_backend
from .config import RunLock, build_parser, dump, load_config
from .planner import Schedule, schedules_at_equal_budget, schedule_by_name
from .plant import FrankaPlant
from .trial import run_trial
from .training import get_model

LIMITS = (1.5, 3.0)
SIGMA_STARTS = (1.2, 3.0)
SCHEDULES = ("single_wide", "staged_wide", "staged_narrow", "anneal")


def clip_fraction(plant, cfg, sigma: float, rng, repeats: int = 200) -> float:
    """Share of proposal entries the action box rejects, from a zero nominal."""
    eps = rng.standard_normal((repeats, cfg.planner.horizon, plant.act_dim))
    return float(np.mean(np.abs(sigma * eps) > plant.limit))


def main() -> None:
    cfg0 = load_config(**vars(build_parser(__doc__).parse_args()))
    torch.set_num_threads(cfg0.run.torch_threads)

    out_dir = Path(__file__).resolve().parent / cfg0.run.out_dir / "wide"
    out_dir.mkdir(parents=True, exist_ok=True)
    RunLock(out_dir).__enter__()
    dump(cfg0, out_dir / "config.json")
    trials_path = out_dir / "trials.jsonl"

    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["schedule"], r["model_seed"],
                          r["target_idx"]))

    clips: dict = {}
    total = (len(LIMITS) * len(SIGMA_STARTS) * len(SCHEDULES)
             * len(cfg0.run.model_seeds) * cfg0.task.num_targets)
    print(f"limits {LIMITS} x sigma_start {SIGMA_STARTS} x schedules "
          f"{len(SCHEDULES)} x seeds {len(cfg0.run.model_seeds)} x targets "
          f"{cfg0.task.num_targets} = {total} trials -> {trials_path}", flush=True)

    idx = 0
    with open(trials_path, "a") as fh:
        for limit in LIMITS:
            cfg = cfg0.replace(plant=replace(cfg0.plant, action_limit=limit))
            plant = FrankaPlant(cfg)
            targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
            models = {s: get_model(cfg, plant, "bilinear", s,
                                   tag=f"lim{limit:g}")
                      for s in cfg.run.model_seeds}

            for sigma_start in SIGMA_STARTS:
                pl = replace(cfg.planner, sigma_start=sigma_start)
                sub = cfg.replace(planner=pl)
                allsc = schedules_at_equal_budget(pl.num_samples, pl.stages,
                                                  sigma_start, pl.sigma_end)
                scheds = [schedule_by_name(allsc, n) for n in SCHEDULES]
                cond = f"lim{limit:g}_sig{sigma_start:g}"

                rng = np.random.default_rng(0)
                clips[cond] = {s.name: clip_fraction(plant, sub, s.sigma_start, rng)
                               for s in scheds}
                print(f"\n--- {cond}  clip fractions "
                      f"{ {k: round(v, 3) for k, v in clips[cond].items()} }",
                      flush=True)

                for seed in cfg.run.model_seeds:
                    backend = build_backend("bilinear", plant, pl, models[seed])
                    for sched in scheds:
                        for ti, goal in enumerate(targets):
                            idx += 1
                            key = (cond, sched.name, seed, ti)
                            if key in done:
                                continue
                            rec = run_trial(sub, plant, backend, sched, goal,
                                            condition=cond, model_seed=seed,
                                            target_idx=ti, rng_seed=cfg.run.rng_base + ti)
                            fh.write(json.dumps(rec.to_json()) + "\n")
                            fh.flush()
                            print(f"[{idx}/{total}] {cond} {sched.name:14s} "
                                  f"s{seed} t{ti} err={rec.final_err:.4f} "
                                  f"steps={rec.steps} "
                                  f"-> {'STRICT' if rec.reached_strict else 'miss'}",
                                  flush=True)

    (out_dir / "clip_fractions.json").write_text(json.dumps(clips, indent=2))
    print(f"\ndone -> {trials_path}", flush=True)


if __name__ == "__main__":
    main()
