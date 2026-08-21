"""Does the fast path compute the same thing, and how much time does it save?

The equivalence check comes first, because a faster implementation that quietly
disagrees with the reference is not a result. Both paths receive the same trained
checkpoint and the same candidate batch, and the largest disagreement is
reported next to float32 resolution.

    PYTHONPATH=. python -m fast.bench --config exp_b
"""

from __future__ import annotations

import time

import numpy as np
import torch

from exper.backends import build_backend
from exper.config import build_parser, describe, load_config
from exper.planner import MBDPlanner, Schedule
from exper.plant import FrankaPlant
from exper.training import get_model

from .planner import FastNumpyBackend, FusedBilinearBackend, TorchMBDPlanner
from .rollout import FastLifted


def timed(fn, reps: int) -> float:
    fn()
    t0 = time.perf_counter()
    for _ in range(reps):
        fn()
    return (time.perf_counter() - t0) / reps * 1e3


def main() -> None:
    ap = build_parser(__doc__)
    ap.add_argument("--model-seed", type=int, default=0)
    ap.add_argument("--reps", type=int, default=20)
    args = ap.parse_args()
    cfg = load_config(**vars(args))
    torch.set_num_threads(cfg.run.torch_threads)

    plant = FrankaPlant(cfg)
    goal = plant.targets(cfg.task.num_targets, cfg.task.target_seed)[0]
    model = get_model(cfg, plant, "bilinear", args.model_seed)
    fused = FastLifted(model, plant.num_joints)

    p = cfg.planner
    N, S, T, m = p.num_samples, p.stages, p.horizon, plant.act_dim
    nj = plant.num_joints
    print(describe(cfg), flush=True)

    # ------------------------------------------------------------ equivalence
    rng = np.random.default_rng(0)
    batch = np.clip(rng.normal(0, 0.6, (N, T, m)), -plant.limit, plant.limit)
    batch_t = torch.as_tensor(batch, dtype=torch.float32)
    b0 = torch.as_tensor(plant.observe(plant.reset()), dtype=torch.float32)

    with torch.no_grad():
        ref_obs = model.rollout(b0, batch_t)
    fast_obs = fused.rollout(b0, batch_t)
    fast_tcp = fused.rollout_tracked(b0, batch_t)

    obs_gap = (ref_obs - fast_obs).abs().max().item()
    tcp_gap = (ref_obs[..., nj:] - fast_tcp).abs().max().item()
    scale = ref_obs.abs().max().item()

    ref_back = build_backend("bilinear", plant, p, model)
    fast_back = FusedBilinearBackend(fused, plant, p)
    state = plant.reset()
    ref_cost = ref_back.cost_fn(state, goal)(batch)
    fast_cost = fast_back.cost_fn(state, goal)(batch_t).numpy()
    cost_gap = float(np.abs(ref_cost - fast_cost).max())
    order = int((np.argsort(ref_cost) != np.argsort(fast_cost)).sum())

    print("\nequivalence on an identical batch")
    print(f"  observation   max abs diff {obs_gap:.3e}  (values up to {scale:.3f})")
    print(f"  tracked only  max abs diff {tcp_gap:.3e}")
    print(f"  cost          max abs diff {cost_gap:.3e}"
          f"  (relative {cost_gap / max(abs(ref_cost).max(), 1e-12):.2e})")
    print(f"  candidates whose rank moved: {order}/{N}")
    print(f"  float32 resolution at this scale: {np.spacing(np.float32(scale)):.3e}")

    # ----------------------------------------------------------------- timings
    print("\ntimings")
    ms_ref_roll = timed(lambda: model.rollout(b0, batch_t), args.reps)
    ms_fast_roll = timed(lambda: fused.rollout_tracked(b0, batch_t), args.reps)
    print(f"  rollout, one stage    reference {ms_ref_roll:6.2f} ms"
          f"   fast {ms_fast_roll:6.2f} ms   {ms_ref_roll/ms_fast_roll:5.2f}x")

    sched = Schedule("anneal", S, N, p.sigma_start, p.sigma_end)
    ref_plan = MBDPlanner(T, m, plant.limit, sched, p.alpha, p.eta,
                          p.alpha_adaptive, p.alpha_scale)
    fast_plan = TorchMBDPlanner(T, m, plant.limit, sched, p.alpha, p.eta,
                                p.alpha_adaptive, p.alpha_scale,
                                generator=torch.Generator().manual_seed(0))
    U_np = np.zeros((T, m))
    U_t = torch.zeros(T, m)
    ref_fn = ref_back.cost_fn(state, goal)
    fast_fn = fast_back.cost_fn(state, goal)
    np_rng = np.random.default_rng(0)

    ms_ref = timed(lambda: ref_plan.plan(U_np, ref_fn, np_rng), args.reps)
    ms_fast = timed(lambda: fast_plan.plan(U_t, fast_fn), args.reps)
    period = cfg.plant.control_dt * 1e3
    print(f"  control step          reference {ms_ref:6.2f} ms"
          f"   fast {ms_fast:6.2f} ms   {ms_ref/ms_fast:5.2f}x")
    print(f"  share of the {period:.0f} ms period   "
          f"{ms_ref/period*100:5.1f}%          {ms_fast/period*100:5.1f}%")


if __name__ == "__main__":
    main()
