"""Where inside a control step does each schedule move the plan?

The update statistics reported elsewhere are read off the terminal stage, since
that is the stage whose output is executed. That leaves the reach argument
unmeasured: it is the opening stages, not the terminal one, that are supposed to
carry the plan out of a basin. This opens up all S stages of one control step.

Per schedule it reports, as medians over the run:

    stage|dU|     displacement of each stage on its own
    sum|dU|       sum of the stage magnitudes, the path length within a step
    net|dU|       norm of the whole step's displacement, opening to terminal
    net/sum       1.0 if the stages pull the same way, 0 if they cancel

A wide opening stage that supplies reach must show a large stage|dU| early and
a net that keeps it. If net/sum collapses for the annealed schedule, the wide
stages are cancelling and reach has to be argued from the outcome instead.

    python -m exper.probe_stages --config exp_c --set run.model_seeds=[0]
"""

from __future__ import annotations

import numpy as np
import torch

from .backends import build_backend
from .config import build_parser, load_config
from .planner import MBDPlanner, Schedule
from .plant import FrankaPlant
from .training import get_model


def run(cfg, plant, backend, sched, goal, rng_seed):
    rng = np.random.default_rng(rng_seed)
    planner = MBDPlanner(cfg.planner.horizon, plant.act_dim, plant.limit,
                         sched, cfg.planner.alpha, cfg.planner.eta,
                         cfg.planner.alpha_adaptive, cfg.planner.alpha_scale)
    state = plant.reset()
    U = np.zeros((cfg.planner.horizon, plant.act_dim))
    per_stage, sums, nets = [], [], []
    for _ in range(cfg.task.steps):
        trace: list[dict] = []
        before = U.copy()
        U = planner.plan(U, backend.cost_fn(state, goal), rng, trace=trace)
        d = [t["dU"] for t in trace]
        per_stage.append(d)
        sums.append(float(np.sum(d)))
        nets.append(float(np.linalg.norm(U - before)))
        state = plant.step(state, plant.clip(U[0]))
        U = MBDPlanner.warm_start(U)
    return np.asarray(per_stage), np.asarray(sums), np.asarray(nets)


def main() -> None:
    cfg = load_config(**vars(build_parser(__doc__).parse_args()))
    torch.set_num_threads(cfg.run.torch_threads)
    plant = FrankaPlant(cfg)
    targets = plant.targets(cfg.task.num_targets, cfg.task.target_seed)
    seed = cfg.run.model_seeds[0]
    model = get_model(cfg, plant, "bilinear", seed)
    backend = build_backend("bilinear", plant, cfg.planner, model)

    S, N = cfg.planner.stages, cfg.planner.num_samples
    scheds = [Schedule("fixed0.3", S, N, 0.3, 0.3),
              Schedule("fixed1.2", S, N, 1.2, 1.2),
              Schedule("anneal", S, N, cfg.planner.sigma_start,
                       cfg.planner.sigma_end)]

    print(f"alpha={cfg.planner.alpha}  N={N}  S={S}  "
          f"targets={len(targets)}  seed={seed}\n")
    head = "".join(f"  stage{i+1}" for i in range(S))
    print(f"{'schedule':<10}{head}{'sum':>9}{'net':>9}{'net/sum':>9}")

    for sched in scheds:
        ps, sm, nt = [], [], []
        for ti, goal in enumerate(targets):
            a, b, c = run(cfg, plant, backend, sched, goal,
                          cfg.run.rng_base + ti)
            ps.append(a); sm.append(b); nt.append(c)
        ps = np.concatenate(ps); sm = np.concatenate(sm); nt = np.concatenate(nt)
        cells = "".join(f"{v:>8.3f}" for v in np.median(ps, axis=0))
        print(f"{sched.name:<10}{cells}{np.median(sm):>9.3f}"
              f"{np.median(nt):>9.3f}{np.median(nt / sm):>9.3f}")


if __name__ == "__main__":
    main()
