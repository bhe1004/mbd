"""Experiment C: annealing under controlled surrogate error.

The bilinear input matrices of the planner's model are scaled to (1-lambda)B_i
while the plant is left untouched, so lambda sets how much of the
configuration-dependent input gain the planner still sees. Five schedules run at
one total sample budget on each lambda, and two diagnostics locate where each
fixed family stops.

    python -m exper.run_c --config exp_c
    python -m exper.run_c --config exp_c --set run.model_seeds=[0]
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from .backends import build_backend
from .config import RunLock, build_parser, describe, dump, load_config
from .koopman import freeze_gain, visited_mean_lift
from .planner import (MBDPlanner, Schedule, first_action_dispersion,
                      schedule_by_name, schedules_at_equal_budget)
from .plant import FrankaPlant
from .trial import plan_end_gap, run_trial
from .training import get_model

# how much of the state-dependent input gain the planner keeps
LAMBDAS = (0.0, 0.5, 1.0)
DISPERSION_REPEATS = 40


def dispersion_diagnostic(cfg, plant, backend, goal, rng_seed=7) -> dict:
    """Spread of the first action over repeated plans, wide against narrow.

    Measured twice: from a settled state, where the spread is the floor the
    executed command inherits, and from the home state, where the same spread is
    how far the population reaches.
    """
    wide, narrow = cfg.planner.sigma_start, cfg.planner.sigma_end
    out = {}
    for label, sigma in (("wide", wide), ("narrow", narrow)):
        sched = Schedule("staged", cfg.planner.stages,
                         cfg.planner.num_samples, sigma, sigma)
        planner = MBDPlanner(cfg.planner.horizon, plant.act_dim, plant.limit,
                             sched, cfg.planner.alpha, cfg.planner.eta,
                             cfg.planner.alpha_adaptive, cfg.planner.alpha_scale)
        rng = np.random.default_rng(rng_seed)

        state = plant.reset()
        U = np.zeros((cfg.planner.horizon, plant.act_dim))
        out[f"cold_{label}"] = first_action_dispersion(
            planner, U, backend.cost_fn(state, goal), DISPERSION_REPEATS, rng)

        for _ in range(cfg.task.steps // 2):
            U = planner.plan(U, backend.cost_fn(state, goal), rng)
            state = plant.step(state, plant.clip(U[0]))
            U = MBDPlanner.warm_start(U)
        out[f"settled_{label}"] = first_action_dispersion(
            planner, U, backend.cost_fn(state, goal), DISPERSION_REPEATS, rng)
    return out


def main() -> None:
    ap = build_parser(__doc__)
    cfg = load_config(**vars(ap.parse_args()))
    torch.set_num_threads(cfg.run.torch_threads)

    out_dir = Path(__file__).resolve().parent / cfg.run.out_dir / f"c_{cfg.run.tag}"
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

    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["lam"], r["schedule"], r["model_seed"], r["target_idx"]))

    total = len(LAMBDAS) * len(schedules) * len(cfg.run.model_seeds) * len(targets)
    print(f"lambdas {len(LAMBDAS)} x schedules {len(schedules)} x seeds "
          f"{len(cfg.run.model_seeds)} x targets {len(targets)} = {total} trials",
          flush=True)

    idx = 0
    diagnostics: dict = {}
    with open(trials_path, "a") as fh:
        for seed in cfg.run.model_seeds:
            trained = get_model(cfg, plant, "bilinear", seed)
            # freeze point: the mean lifted state over the training excitation,
            # so lam=1 freezes the input gain at a representative configuration
            exc = plant.excite(seed=100 + seed, white=False)
            obs = torch.as_tensor(exc.obs.reshape(-1, plant.obs_dim),
                                  dtype=torch.float32)
            z_bar = visited_mean_lift(trained, obs)
            for lam in LAMBDAS:
                model = freeze_gain(trained, z_bar, lam)
                backend = build_backend("bilinear", plant, cfg.planner, model)
                for sched in schedules:
                    for ti, goal in enumerate(targets):
                        idx += 1
                        key = (lam, sched.name, seed, ti)
                        if key in done:
                            continue
                        rec = run_trial(cfg, plant, backend, sched, goal,
                                        condition=f"lam{lam}", model_seed=seed,
                                        target_idx=ti, lam=lam,
                                        rng_seed=cfg.run.rng_base + ti)
                        fh.write(json.dumps(rec.to_json()) + "\n")
                        fh.flush()
                        print(f"[{idx}/{total}] lam={lam:.2f} {sched.name:14s} "
                              f"s{seed} t{ti} err={rec.final_err:.4f} "
                              f"steps={rec.steps} "
                              f"-> {'STRICT' if rec.reached_strict else 'miss'}",
                              flush=True)

                if seed == cfg.run.model_seeds[0]:
                    goal = targets[0]
                    disp = dispersion_diagnostic(cfg, plant, backend, goal)
                    narrow_sched = schedule_by_name(schedules, "staged_narrow")
                    pred, true = plan_end_gap(cfg, plant, backend,
                                              narrow_sched, goal)
                    diagnostics[f"lam{lam}"] = {
                        **disp, "narrow_plan_end_pred": pred,
                        "narrow_plan_end_true": true,
                    }
                    print(f"  diagnostics lam={lam}: {diagnostics[f'lam{lam}']}",
                          flush=True)

    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2))
    print(f"\ndone -> {trials_path}", flush=True)


if __name__ == "__main__":
    main()
