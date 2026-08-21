"""Experiment B: rollout class and planning latency.

Six conditions run the same reaching task inside one planner with only the
rollout backend exchanged, over ``model_seeds x targets`` trials. The run also
measures held-out prediction error for the three training choices, which is the
open-loop evidence behind the multi-step loss and the coherent excitation.

    python -m exper.run_b --config exp_b
    python -m exper.run_b --config exp_b --set run.model_seeds=[0,1]
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

from .backends import build_backend
from .config import Config, RunLock, build_parser, describe, dump, load_config
from .planner import Schedule
from .plant import FrankaPlant
from .trial import run_trial
from .training import get_model, horizon_error

# condition -> (model variant, backend kind, lift-dim override)
CONDITIONS = [
    ("oracle", None, "oracle", None),
    ("linear", "linear", "linear", None),
    ("linear_large", "linear", "linear", "large"),
    ("mlp", "mlp", "mlp", None),
    ("split", "linear", "split", None),
    ("bilinear", "bilinear", "bilinear", None),
]


def _cfg_for(cfg: Config, lift_override: str | None) -> Config:
    if lift_override != "large":
        return cfg
    big = replace(cfg.model, lift_dim=cfg.model.lift_dim_large)
    return cfg.replace(model=big)


def open_loop_table(cfg: Config, plant: FrankaPlant, seed: int) -> dict:
    """Held-out horizon error for the three training choices."""
    out = {}
    variants = {
        "multi_step": cfg,
        "one_step": cfg.replace(train=replace(cfg.train, one_step=True)),
        "white_excitation": cfg.replace(data=replace(cfg.data, white=True)),
    }
    held = plant.excite(seed=9_000 + seed, white=False)
    b = torch.as_tensor(held.obs[:, : cfg.train.horizon + 1], dtype=torch.float32)
    u = torch.as_tensor(held.controls[:, : cfg.train.horizon], dtype=torch.float32)
    for name, sub in variants.items():
        model = get_model(sub, plant, "bilinear", seed, tag=name)
        out[name] = horizon_error(model, b, u, plant.num_joints)
    return out


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--conditions", nargs="+", default=None,
                    metavar="NAME",
                    help="restrict to these conditions (default: all six)")
    ap.add_argument("--skip-open-loop", action="store_true",
                    help="do not measure the training-choice prediction errors")
    args = ap.parse_args()
    cfg = load_config(**vars(args))
    picked = set(args.conditions) if args.conditions else None
    conditions = [c for c in CONDITIONS if picked is None or c[0] in picked]
    torch.set_num_threads(cfg.run.torch_threads)

    out_dir = Path(__file__).resolve().parent / cfg.run.out_dir / f"b_{cfg.run.tag}"
    out_dir.mkdir(parents=True, exist_ok=True)
    lock = RunLock(out_dir).__enter__()
    dump(cfg, out_dir / "config.json")
    trials_path = out_dir / "trials.jsonl"
    print(describe(cfg), flush=True)

    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    schedule = Schedule("anneal", cfg.planner.stages, cfg.planner.num_samples,
                        cfg.planner.sigma_start, cfg.planner.sigma_end)

    done = set()
    if trials_path.exists():
        for line in trials_path.read_text().splitlines():
            if line.strip():
                r = json.loads(line)
                done.add((r["condition"], r["model_seed"], r["target_idx"]))

    total = len(conditions) * len(cfg.run.model_seeds) * len(targets)
    print(f"conditions {len(conditions)} x seeds {len(cfg.run.model_seeds)} "
          f"x targets {len(targets)} = {total} trials -> {trials_path}", flush=True)

    idx = 0
    with open(trials_path, "a") as fh:
        for seed in cfg.run.model_seeds:
            models = {}
            for name, variant, kind, lift in conditions:
                if variant is not None:
                    sub = _cfg_for(cfg, lift)
                    models[name] = (sub, get_model(sub, plant, variant, seed,
                                                   tag="large" if lift else ""))
            for name, variant, kind, lift in conditions:
                sub, model = models.get(name, (cfg, None))
                backend = build_backend(kind, plant, sub.planner, model)
                for ti, goal in enumerate(targets):
                    idx += 1
                    if (name, seed, ti) in done:
                        continue
                    rec = run_trial(cfg, plant, backend, schedule, goal,
                                    condition=name, model_seed=seed,
                                    target_idx=ti, rng_seed=cfg.run.rng_base + ti)
                    fh.write(json.dumps(rec.to_json()) + "\n")
                    fh.flush()
                    print(f"[{idx}/{total}] {name:13s} s{seed} t{ti} "
                          f"err={rec.final_err:.4f} steps={rec.steps} "
                          f"ms={rec.ms_per_step:.0f} "
                          f"-> {'STRICT' if rec.reached_strict else ('reach' if rec.reached else 'miss')}",
                          flush=True)

    if not args.skip_open_loop:
        ol = open_loop_table(cfg, plant, cfg.run.model_seeds[0])
        (out_dir / "open_loop.json").write_text(json.dumps(ol, indent=2))
        print("\nheld-out horizon error [m]:", flush=True)
        for k, v in ol.items():
            print(f"  {k:18s} {v:.4f}", flush=True)


if __name__ == "__main__":
    main()
