"""Why does the settled error stop where it does?

Drives one trial to its settled state, then dissects the update the planner
performs there. Reports, per condition and per terminal noise level:

    err           the tracking error the run settled at
    dJ            spread of the candidate costs at that state (std and p95-p05)
    N_eff         1 / sum w^2, the effective sample size of the softmax
    |dU|          norm of the nominal displacement the update produces
    step_gain     how far the executed first action moves the TCP toward the goal

A flat softmax (N_eff close to N) means the candidate costs differ by much less
than alpha, so the update carries no direction and the run stops regardless of
sigma. That is hypothesis (A). A small |dU| with a sharp softmax instead points
at the control penalty, hypothesis (B).

    python -m exper.probe_floor --config exp_b --set plant.kinematic=true
"""

from __future__ import annotations

import numpy as np
import torch

from .backends import build_backend
from .config import build_parser, load_config
from .planner import MBDPlanner, Schedule, effective_sample_size
from .plant import FrankaPlant
from .training import get_model

SIGMAS = (0.30, 0.15, 0.05)
CONDITIONS = [("oracle", None, "oracle"),
              ("split", "linear", "split"),
              ("bilinear", "bilinear", "bilinear")]
NUM_TARGETS = 5


def settle(cfg, plant, backend, sched, goal, rng):
    """Run the closed loop to the end of the budget and return (state, plan)."""
    state = plant.reset()
    U = np.zeros((cfg.planner.horizon, plant.act_dim))
    planner = MBDPlanner(cfg.planner.horizon, plant.act_dim, plant.limit,
                         sched, cfg.planner.alpha, cfg.planner.eta,
                         cfg.planner.alpha_adaptive, cfg.planner.alpha_scale)
    for _ in range(cfg.task.steps):
        U = planner.plan(U, backend.cost_fn(state, goal), rng)
        state = plant.step(state, plant.clip(U[0]))
        U = MBDPlanner.warm_start(U)
    return planner, state, U


def dissect(planner, cfg, plant, backend, state, U, goal, sigma, rng):
    """One stage at `sigma` from the settled nominal, opened up."""
    cost_fn = backend.cost_fn(state, goal)
    eps = rng.standard_normal((cfg.planner.num_samples,
                               cfg.planner.horizon, plant.act_dim))
    cand = np.clip(U[None] + sigma * eps, -plant.limit, plant.limit)
    J = np.asarray(cost_fn(cand), dtype=np.float64)
    alpha = planner.temperature(J)
    w = np.exp(-(J - J.min()) / alpha)
    w = w / w.sum()
    dU = np.tensordot(w, cand, axes=(0, 0)) - U

    tcp = plant.tcp(plant.observe(state))
    err = float(np.linalg.norm(tcp - goal))
    nxt = plant.step(state, plant.clip((U + dU)[0]))
    err_next = float(np.linalg.norm(plant.tcp(plant.observe(nxt)) - goal))
    return dict(err=err, alpha=alpha, J_std=float(J.std()),
                J_spread=float(np.percentile(J, 95) - np.percentile(J, 5)),
                n_eff=effective_sample_size(w), dU=float(np.linalg.norm(dU)),
                closes=err - err_next)


def main() -> None:
    cfg = load_config(**vars(build_parser(__doc__).parse_args()))
    torch.set_num_threads(cfg.run.torch_threads)
    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)[:NUM_TARGETS]
    seed = cfg.run.model_seeds[0]

    print(f"alpha={cfg.planner.alpha} adaptive={cfg.planner.alpha_adaptive}  "
          f"w_ctrl={cfg.planner.w_ctrl}  "
          f"N={cfg.planner.num_samples}  kinematic={cfg.plant.kinematic}\n")
    print(f"{'cond':<9}{'sigma':>6}{'err[m]':>9}{'alpha':>9}{'J_std':>10}"
          f"{'J_p95-p05':>11}{'N_eff':>8}{'|dU|':>9}{'closes[m]':>11}")

    for name, variant, kind in CONDITIONS:
        model = get_model(cfg, plant, variant, seed) if variant else None
        backend = build_backend(kind, plant, cfg.planner, model)
        sched = Schedule("anneal", cfg.planner.stages, cfg.planner.num_samples,
                         cfg.planner.sigma_start, cfg.planner.sigma_end)
        for sigma in SIGMAS:
            acc = []
            for ti, goal in enumerate(targets):
                rng = np.random.default_rng(1000 + ti)
                planner, state, U = settle(cfg, plant, backend, sched, goal, rng)
                acc.append(dissect(planner, cfg, plant, backend,
                                   state, U, goal, sigma, rng))
            m = {k: float(np.mean([a[k] for a in acc])) for k in acc[0]}
            print(f"{name:<9}{sigma:>6.2f}{m['err']:>9.4f}{m['alpha']:>9.4f}"
                  f"{m['J_std']:>10.5f}{m['J_spread']:>11.5f}{m['n_eff']:>8.0f}"
                  f"{m['dU']:>9.4f}{m['closes']:>11.5f}")


if __name__ == "__main__":
    main()
