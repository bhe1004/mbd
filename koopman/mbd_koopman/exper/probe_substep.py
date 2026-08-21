"""What integrating at the control rate buys, and what it costs in fidelity.

DIAL-MPC~\\cite{xue2025full} sets the physics timestep equal to the control
period, so one planner step is one integration step and the serial chain of
Sec. III-B shrinks by the substep count. This probe applies the same choice to
the FR3 plant and prices both sides of it: the wall-clock time per control step,
and how far the coarse rollout departs from the 2 ms reference the study treats
as the truth.

A cheaper rollout that predicts a different plant is not an oracle, so the two
columns have to be read together.

    exper/mjx.sh -m exper.probe_substep --config exp_b
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from .config import build_parser, describe, load_config
from .plant import FrankaPlant

# substeps per control period; 25 is the study's 2 ms integrator, 1 is DIAL-MPC's
LADDER = (25, 10, 5, 2, 1)


def set_substeps(plant: FrankaPlant, nsub: int) -> float:
    """Retune the plant's integrator to take ``nsub`` steps per control period."""
    dt = plant.dt / nsub
    plant.task.model.opt.timestep = dt
    plant.task._nsub = nsub
    return dt


def tcp_paths(plant: FrankaPlant, controls: np.ndarray) -> np.ndarray:
    return plant.rollout_true(plant.reset(), controls)[..., plant.num_joints:]


def gap(reference: np.ndarray, other: np.ndarray) -> dict:
    err = np.linalg.norm(reference - other, axis=-1)
    return {"median_mm": float(np.median(err) * 1e3),
            "max_mm": float(err.max() * 1e3),
            "terminal_median_mm": float(np.median(err[:, -1]) * 1e3),
            "finite": bool(np.isfinite(other).all())}


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--gpu", action="store_true",
                    help="also time the MJX rollout at each substep count")
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    cfg = load_config(**vars(args))

    plant = FrankaPlant(cfg)
    N, S, T = cfg.planner.num_samples, cfg.planner.stages, cfg.planner.horizon
    period_ms = cfg.plant.control_dt * 1e3
    print(describe(cfg), flush=True)
    print(f"control period {period_ms:.0f} ms, N={N}, S={S}, T={T}\n", flush=True)

    rng = np.random.default_rng(0)
    probe = np.clip(rng.normal(0, 0.6, (64, T, plant.act_dim)),
                    -plant.limit, plant.limit)
    batch = np.clip(rng.normal(0, 0.6, (N, T, plant.act_dim)),
                    -plant.limit, plant.limit)

    set_substeps(plant, LADDER[0])
    reference = tcp_paths(plant, probe)
    path_m = float(np.median(
        np.linalg.norm(np.diff(reference, axis=1), axis=-1).sum(axis=1)))
    print(f"reference: {LADDER[0]} substeps of "
          f"{plant.dt / LADDER[0] * 1e3:.0f} ms, path length {path_m*1e3:.0f} mm\n",
          flush=True)

    rows = []
    for nsub in LADDER:
        dt = set_substeps(plant, nsub)
        fidelity = gap(reference, tcp_paths(plant, probe))

        state = plant.reset()
        plant.rollout_true(state, batch[:8])            # warm the thread pool
        t0 = time.perf_counter()
        for _ in range(args.reps):
            plant.rollout_true(state, batch)
        cpu_stage_ms = (time.perf_counter() - t0) / args.reps * 1e3

        row = dict(nsub=nsub, timestep_ms=dt * 1e3,
                   sequential_steps=S * T * nsub,
                   cpu_control_step_ms=cpu_stage_ms * S, **fidelity)
        print(f"  nsub={nsub:3d} ({dt*1e3:5.1f} ms)  chain {S*T*nsub:5d}"
              f"  CPU {cpu_stage_ms*S:8.1f} ms"
              f"  gap median {fidelity['median_mm']:8.2f} mm"
              f"  max {fidelity['max_mm']:8.2f} mm"
              f"{'' if fidelity['finite'] else '  DIVERGED'}", flush=True)
        rows.append(row)

    if args.gpu:
        print("", flush=True)
        from .backends_mjx import MjxOracleBackend
        for row in rows:
            set_substeps(plant, row["nsub"])
            backend = MjxOracleBackend(plant, cfg.planner, N, T,
                                       disable_contact=True)
            backend.warmup()
            state = plant.reset()
            backend._jax.block_until_ready(backend.rollout(state, batch))
            t0 = time.perf_counter()
            for _ in range(args.reps):
                out = backend.rollout(state, batch)
            backend._jax.block_until_ready(out)
            sec = (time.perf_counter() - t0) / args.reps
            row["gpu_control_step_ms"] = sec * S * 1e3
            print(f"  nsub={row['nsub']:3d}  GPU {sec*S*1e3:8.1f} ms"
                  f"  ({'meets' if sec*S*1e3 <= period_ms else 'misses'}"
                  f" the {period_ms:.0f} ms period)", flush=True)

    out_path = Path(__file__).resolve().parent / cfg.run.out_dir / "substep_ladder.json"
    out_path.write_text(json.dumps(
        {"rows": rows, "reference_nsub": LADDER[0], "path_length_mm": path_m * 1e3,
         "period_ms": period_ms}, indent=2))
    print(f"\nwrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
