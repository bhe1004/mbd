"""What one batched physics substep costs on the GPU, and what that buys.

The deadline question of Sec. III-B is arithmetic, not a closed loop: one
control step issues ``S * T * nsub`` substeps that run in order, and the GPU
parallelizes only the ``N`` candidates within each of them. This probe times a
single rollout call and reads the control step off it, so the answer arrives in
seconds rather than after a full trial.

    exper/mjx.sh -m exper.probe_gpu --config exp_b
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .backends_mjx import MjxOracleBackend, agreement
from .config import build_parser, describe, load_config
from .plant import FrankaPlant


def time_rollout(backend, plant, num, horizon, reps: int) -> float:
    """Seconds per rollout call, after the kernel is built."""
    rng = np.random.default_rng(0)
    u = np.clip(rng.normal(0, 0.6, (num, horizon, plant.act_dim)),
                -plant.limit, plant.limit)
    state = plant.reset()
    backend.rollout(state, u)                      # warm
    backend._jax.block_until_ready(backend.rollout(state, u))
    t0 = time.perf_counter()
    for _ in range(reps):
        out = backend.rollout(state, u)
    backend._jax.block_until_ready(out)
    return (time.perf_counter() - t0) / reps


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--candidates", nargs="+", type=int, default=None,
                    metavar="N", help="batch widths to time")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--contact", choices=("off", "on", "both"), default="off",
                    help="the model has no active contact pairs, so 'off' is "
                         "the like-for-like setting; 'both' also prices the "
                         "collision pipeline MJX builds anyway")
    args = ap.parse_args()
    cfg = load_config(**vars(args))

    plant = FrankaPlant(cfg)
    T, S = cfg.planner.horizon, cfg.planner.stages
    widths = args.candidates or [cfg.planner.num_samples]
    period_ms = cfg.plant.control_dt * 1e3
    print(describe(cfg), flush=True)

    rows = []
    for num in widths:
        modes = {"off": (True,), "on": (False,),
                 "both": (True, False)}[args.contact]
        for no_contact in modes:
            backend = MjxOracleBackend(plant, cfg.planner, num, T,
                                       disable_contact=no_contact)
            nsub = backend.nsub
            compile_s = backend.warmup()
            sec = time_rollout(backend, plant, num, T, args.reps)
            sub_us = sec / (T * nsub) * 1e6
            step_ms = sec * S * 1e3
            rows.append(dict(num_candidates=num, disable_contact=no_contact,
                             compile_s=compile_s, rollout_ms=sec * 1e3,
                             substep_us=sub_us, control_step_ms=step_ms,
                             over_budget=step_ms / period_ms))
            print(f"  N={num:5d} contact={'off' if no_contact else 'on ':>3s}"
                  f"  compile {compile_s:6.1f}s"
                  f"  rollout {sec*1e3:8.1f} ms ({T*nsub} substeps)"
                  f"  substep {sub_us:7.1f} us"
                  f"  control step {step_ms:8.1f} ms"
                  f"  = {step_ms/period_ms:6.1f}x the {period_ms:.0f} ms period",
                  flush=True)

    # the plant must be the same plant, or the baseline is not a baseline
    backend = MjxOracleBackend(plant, cfg.planner, 64, T)
    backend.warmup()
    rng = np.random.default_rng(0)
    probe = np.clip(rng.normal(0, 0.6, (64, T, plant.act_dim)),
                    -plant.limit, plant.limit)
    gap = agreement(plant, backend, probe)
    print(f"\n  GPU(float32) vs CPU(float64) on identical commands: {gap}",
          flush=True)

    budget_us = period_ms * 1e3 / (S * T * backend.nsub)
    print(f"\n  meeting the period needs {budget_us:.1f} us per batched substep",
          flush=True)

    out = Path(__file__).resolve().parent / cfg.run.out_dir / "gpu_probe.json"
    out.write_text(json.dumps(
        {"rows": rows, "agreement": gap, "substep_budget_us": budget_us,
         "sequential_substeps_per_control_step": S * T * backend.nsub},
        indent=2))
    print(f"\nwrote {out}", flush=True)


if __name__ == "__main__":
    main()
