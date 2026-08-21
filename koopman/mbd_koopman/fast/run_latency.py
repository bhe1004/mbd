"""Closed-loop latency and accuracy under the fast path.

Every learned condition of Sec. V-A goes through ``fast.rollout``, so the rows of
the latency table stay comparable with each other. The oracle is not here: it
rolls through MuJoCo and shares none of the items ``fast/`` applies.

Two modes, because reproducibility and speed pull against each other:

``exact`` (default)
    The reference planner and the reference trial loop, with only the rollout
    replaced by the fused one. The candidate stream, the softmax and the weighted
    mean stay the float64 NumPy ones, so a trial is comparable with the
    reference run goal by goal and the accuracy tables must not move.

``torch``
    The all-torch planner as well. Faster, but the candidates come from a torch
    generator, so trials are not paired with the reference and only aggregates
    are comparable.

    PYTHONPATH=. python -m fast.run_latency --config exp_b            # exact
    PYTHONPATH=. python -m fast.run_latency --config exp_b --mode torch
"""

from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from exper.config import RunLock, build_parser, describe, dump, load_config
from exper.planner import Schedule
from exper.plant import FrankaPlant
from exper.trial import run_trial as reference_trial
from exper.training import get_model

from .planner import FastNumpyBackend, FusedBilinearBackend, TorchMBDPlanner
from .rollout import FastLifted, build_fast_rollout


def run_trial_torch(cfg, plant, backend, schedule, goal, *, seed: int,
                    target: int, rng_seed: int) -> dict:
    """One reach under the all-torch planner.

    Mirrors ``exper.trial.run_trial``: full budget, no early stop, warm start by
    a one-step shift. The ``exact`` mode calls that function directly instead.
    """
    p = cfg.planner
    planner = TorchMBDPlanner(
        p.horizon, plant.act_dim, plant.limit, schedule, p.alpha, p.eta,
        p.alpha_adaptive, p.alpha_scale,
        generator=torch.Generator().manual_seed(rng_seed))

    state = plant.reset()
    U = torch.zeros(p.horizon, plant.act_dim)
    times: list[float] = []
    curve: list[float] = []
    min_err = float("inf")
    strict_at = None
    period_ms = cfg.plant.control_dt * 1e3

    for step in range(cfg.task.steps):
        cost_fn = backend.cost_fn(state, goal)
        t0 = time.perf_counter()
        U = planner.plan(U, cost_fn)
        times.append((time.perf_counter() - t0) * 1e3)

        state = plant.step(state, plant.clip(U[0].numpy().astype(np.float64)))
        err = float(np.linalg.norm(plant.tcp(plant.observe(state)) - goal))
        curve.append(err)
        min_err = min(min_err, err)
        if strict_at is None and err <= cfg.task.strict:
            strict_at = step + 1
        U = TorchMBDPlanner.warm_start(U)

    t = np.asarray(times)
    return dict(condition="bilinear_fast", model_seed=seed, target_idx=target,
                final_err=curve[-1], min_err=min_err,
                steps=strict_at or cfg.task.steps,
                reached=min_err <= cfg.task.reach,
                reached_strict=min_err <= cfg.task.strict,
                ms_per_step=float(np.median(t)), worst_ms=float(t.max()),
                deadline_misses=int((t > period_ms).sum()))


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--targets", nargs="+", type=int, default=None,
                    metavar="IDX")
    ap.add_argument("--mode", choices=("exact", "torch"), default="exact",
                    help="exact: reference planner + fast rollout (paired with "
                         "the reference run). torch: all-torch planner too.")
    ap.add_argument("--conditions", nargs="+", default=["bilinear"],
                    metavar="NAME",
                    help="any of linear, linear_large, mlp, split, bilinear")
    args = ap.parse_args()
    cfg = load_config(**vars(args))
    torch.set_num_threads(cfg.run.torch_threads)

    out_dir = Path(__file__).resolve().parent.parent / "exper" / cfg.run.out_dir \
        / f"fast_{args.mode}_{cfg.run.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out_dir).__enter__()
    dump(cfg, out_dir / "config.json")
    print(describe(cfg), flush=True)

    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    picked = args.targets if args.targets is not None else range(len(targets))
    p = cfg.planner
    schedule = Schedule("anneal", p.stages, p.num_samples,
                        p.sigma_start, p.sigma_end)

    trials_path = out_dir / "trials.jsonl"
    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["model_seed"], r["target_idx"]))

    # condition -> (model variant, rollout kind, lift override)
    SPEC = {"linear": ("linear", "linear", None),
            "linear_large": ("linear", "linear", "large"),
            "mlp": ("mlp", "mlp", None),
            "split": ("linear", "split", None),
            "bilinear": ("bilinear", "bilinear", None)}
    unknown = [c for c in args.conditions if c not in SPEC]
    if unknown:
        raise SystemExit(f"unknown conditions {unknown}, pick from {sorted(SPEC)}")

    rows = []
    with open(trials_path, "a") as fh:
        for name in args.conditions:
            variant, kind, lift = SPEC[name]
            sub = cfg if lift != "large" else cfg.replace(
                model=replace(cfg.model, lift_dim=cfg.model.lift_dim_large))
            for seed in cfg.run.model_seeds:
                model = get_model(sub, plant, variant, seed,
                                  tag="large" if lift else "")
                rollout = build_fast_rollout(kind, plant, model)
                backend = (FastNumpyBackend(rollout, plant, sub.planner, name)
                           if args.mode == "exact"
                           else FusedBilinearBackend(rollout, plant, sub.planner))
                for ti in picked:
                    if (name, seed, ti) in done:
                        continue
                    if args.mode == "exact":
                        rec = reference_trial(
                            cfg, plant, backend, schedule, targets[ti],
                            condition=name, model_seed=seed, target_idx=ti,
                            rng_seed=cfg.run.rng_base + ti).to_json()
                    else:
                        rec = run_trial_torch(cfg, plant, backend, schedule,
                                              targets[ti], seed=seed, target=ti,
                                              rng_seed=cfg.run.rng_base + ti)
                        rec["condition"] = name
                    fh.write(json.dumps(rec) + "\n")
                    fh.flush()
                    rows.append(rec)
                    print(f"{name:13s} s{seed} t{ti} "
                          f"final {rec['final_err']:.4f} "
                          f"strict {rec['reached_strict']} "
                          f"{rec['ms_per_step']:.1f} ms "
                          f"worst {rec['worst_ms']:.1f} "
                          f"misses {rec['deadline_misses']}", flush=True)

    for name in args.conditions:
        got = [r for r in rows if r["condition"] == name]
        if not got:
            continue
        print(f"\n{name:13s} {len(got)} trials | reach 5cm "
              f"{sum(r['reached'] for r in got)}/{len(got)} | reach 1cm "
              f"{sum(r['reached_strict'] for r in got)}/{len(got)} | "
              f"median {np.median([r['ms_per_step'] for r in got]):.1f} ms | "
              f"worst {max(r['worst_ms'] for r in got):.1f} ms | "
              f"misses {sum(r['deadline_misses'] for r in got)}", flush=True)
    lock.__exit__(None, None, None)


if __name__ == "__main__":
    main()
