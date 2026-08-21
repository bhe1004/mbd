"""Experiment D: does the temperature separate the two accounts of the wide end?

Sec. V-C attributes the delay of a fixed wide level to the precision floor of
Remark 2: the executed command inherits a dispersion of order sigma/sqrt(N_eff),
so a level that is never narrowed keeps the plan in motion once it arrives. The
closed-loop record does not show that motion -- after entering the band, a wide
schedule is as quiet as the annealed one -- while the approach is uniformly
slower, which is what a smaller effective step would produce instead.

sigma carries both roles at once, so it cannot separate them. The temperature
can, because the two accounts move in opposite directions under it:

    precision floor   alpha down -> weights sharpen -> N_eff down
                      -> sigma/sqrt(N_eff) up -> wide gets WORSE
    small step        alpha down -> the mean commits to the low-cost candidates
                      instead of regressing to the nominal -> wide gets BETTER

The grid is a fixed-sigma ladder crossed with alpha, with the annealed schedule
carried along at every alpha as the reference. N_eff is recorded per trial as
the manipulation check: if alpha does not move it, the test is void.

    python -m exper.run_d --config exp_c --set run.tag=alpha
    python -m exper.run_d --config exp_c --set run.model_seeds=[0]
"""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import torch

from .backends import build_backend
from .config import RunLock, build_parser, describe, dump, load_config
from .plant import FrankaPlant
from .planner import Schedule
from .trial import run_trial
from .training import get_model

# the ladder, in units of the planner's own wide and narrow levels
SIGMAS = (0.3, 0.5, 0.8, 1.2)
ALPHAS = (0.4, 0.1, 0.025)


def grid(cfg):
    """Fixed-sigma ladder plus the annealed reference, at every temperature."""
    S, N = cfg.planner.stages, cfg.planner.num_samples
    cells = []
    for alpha in ALPHAS:
        for sigma in SIGMAS:
            cells.append((alpha, Schedule(f"fixed{sigma:g}", S, N, sigma, sigma)))
        cells.append((alpha, Schedule("anneal", S, N,
                                      cfg.planner.sigma_start,
                                      cfg.planner.sigma_end)))
    return cells


def main() -> None:
    ap = build_parser(__doc__)
    cfg = load_config(**vars(ap.parse_args()))
    torch.set_num_threads(cfg.run.torch_threads)

    out_dir = Path(__file__).resolve().parent / cfg.run.out_dir / f"d_{cfg.run.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out_dir).__enter__()
    dump(cfg, out_dir / "config.json")
    trials_path = out_dir / "trials.jsonl"
    print(describe(cfg), flush=True)

    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    cells = grid(cfg)

    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["schedule"],
                          r["model_seed"], r["target_idx"]))

    total = len(cells) * len(cfg.run.model_seeds) * len(targets)
    print(f"cells {len(cells)} x seeds {len(cfg.run.model_seeds)} x targets "
          f"{len(targets)} = {total} trials -> {trials_path}", flush=True)

    idx = 0
    with open(trials_path, "a") as fh:
        for seed in cfg.run.model_seeds:
            model = get_model(cfg, plant, "bilinear", seed)
            for alpha, sched in cells:
                # only the temperature changes across cells; the rollout, the
                # sample budget and the target stream are held fixed
                planner_cfg = dataclasses.replace(cfg.planner, alpha=alpha)
                cell_cfg = cfg.replace(planner=planner_cfg)
                backend = build_backend("bilinear", plant, planner_cfg, model)
                cond = f"a{alpha:g}"
                for ti, goal in enumerate(targets):
                    idx += 1
                    if (cond, sched.name, seed, ti) in done:
                        continue
                    rec = run_trial(cell_cfg, plant, backend, sched, goal,
                                    condition=cond, model_seed=seed,
                                    target_idx=ti,
                                    rng_seed=cfg.run.rng_base + ti,
                                    record_curve=True, record_trace=True)
                    fh.write(json.dumps(rec.to_json()) + "\n")
                    fh.flush()
                    print(f"[{idx}/{total}] a={alpha:<6g} {sched.name:9s} "
                          f"s{seed} t{ti} err={rec.final_err:.4f} "
                          f"steps={rec.steps:3d} N_eff={rec.n_eff:6.1f} "
                          f"-> {'STRICT' if rec.reached_strict else 'miss'}",
                          flush=True)

    lock.__exit__(None, None, None)
    print(f"\ndone -> {trials_path}", flush=True)


if __name__ == "__main__":
    main()
