"""Where the GPU rollout's time goes: the batch, or the launches.

One batched substep at the planner's ``N`` costs far more than 800 copies of a
seven-joint physics step should. This probe sweeps the batch width and reads off
which regime the rollout sits in.

    flat wall time as N grows   -> launch-bound; the GPU is idle at this width
    time proportional to N      -> compute-bound; the measurement stands

The per-candidate cost is the quantity to compare against the CPU, and the
crossing point is the batch width at which hardware would start to pay.

    exper/mjx.sh -m exper.probe_gpu_scaling --config exp_b
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .backends_mjx import MjxOracleBackend
from .config import build_parser, describe, load_config
from .plant import FrankaPlant

# the CPU oracle of Table III, per scalar substep: 2598 ms for N*S*T*nsub
CPU_MS_PER_CONTROL_STEP = 2598.0


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--candidates", nargs="+", type=int,
                    default=[1, 64, 800, 3200, 8192],
                    metavar="N", help="batch widths to time")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    cfg = load_config(**vars(args))

    plant = FrankaPlant(cfg)
    T, S, N0 = cfg.planner.horizon, cfg.planner.stages, cfg.planner.num_samples
    period_ms = cfg.plant.control_dt * 1e3
    cpu_scalar_us = CPU_MS_PER_CONTROL_STEP * 1e3 / (N0 * S * T * 25)
    print(describe(cfg), flush=True)
    print(f"CPU reference: {CPU_MS_PER_CONTROL_STEP:.0f} ms per control step "
          f"= {cpu_scalar_us:.2f} us per scalar substep (16 threads)\n", flush=True)

    rows = []
    for num in args.candidates:
        backend = MjxOracleBackend(plant, cfg.planner, num, T,
                                   disable_contact=True)
        nsub = backend.nsub
        compile_s = backend.warmup()

        rng = np.random.default_rng(0)
        u = np.clip(rng.normal(0, 0.6, (num, T, plant.act_dim)),
                    -plant.limit, plant.limit)
        state = plant.reset()
        backend._jax.block_until_ready(backend.rollout(state, u))
        t0 = time.perf_counter()
        for _ in range(args.reps):
            out = backend.rollout(state, u)
        backend._jax.block_until_ready(out)
        sec = (time.perf_counter() - t0) / args.reps

        substep_us = sec / (T * nsub) * 1e6           # one batched substep
        scalar_us = substep_us / num                  # per candidate in it
        rows.append(dict(num_candidates=num, compile_s=compile_s,
                         rollout_ms=sec * 1e3, batched_substep_us=substep_us,
                         scalar_substep_us=scalar_us,
                         vs_cpu=scalar_us / cpu_scalar_us))
        print(f"  N={num:6d}  rollout {sec*1e3:8.1f} ms"
              f"  batched substep {substep_us:8.1f} us"
              f"  per candidate {scalar_us:7.3f} us"
              f"  = {scalar_us/cpu_scalar_us:6.2f}x the CPU", flush=True)

    # What a control step would cost if the planner could use the wider batch:
    # the stages stay serial, so only the candidates within one stage widen.
    print("\n  at the planner's N=800 the control step is "
          f"{rows[[r['num_candidates'] for r in rows].index(800)]['rollout_ms']*S:.0f} ms"
          if 800 in args.candidates else "", flush=True)
    print(f"  the 50 ms period allows {period_ms*1e3/(S*T*25):.1f} us "
          f"per batched substep", flush=True)

    out_path = Path(__file__).resolve().parent / cfg.run.out_dir / "gpu_scaling.json"
    out_path.write_text(json.dumps(
        {"rows": rows, "cpu_scalar_substep_us": cpu_scalar_us,
         "period_ms": period_ms}, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
