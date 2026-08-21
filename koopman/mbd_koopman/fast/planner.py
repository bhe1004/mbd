"""The annealed sampling update of ``exper.planner``, kept in torch.

The reference planner draws its candidates with NumPy, hands them to a torch
rollout, copies the decoded horizon back to NumPy and scores it there. This one
never leaves torch, so a stage costs one noise draw, one rollout and one
weighted mean with no conversions in between.

Every quantity that decides the trajectory is copied from the reference: the
stage sigmas, the temperature rule including the adaptive variant, the clip to
the input bound applied to the proposals and to the iterate, the self-normalised
softmax shifted by the population minimum, and the preconditioned step
``U <- U + eta (mean - U)``.
"""

from __future__ import annotations

from typing import Callable

import torch

from exper.config import PlannerCfg
from exper.planner import Schedule

CostFn = Callable[[torch.Tensor], torch.Tensor]   # (K, T, m) -> (K,)


def trajectory_cost(tcp: torch.Tensor, controls: torch.Tensor,
                    goal: torch.Tensor, cfg: PlannerCfg) -> torch.Tensor:
    """Cost of a batch of predicted paths, the torch twin of the reference."""
    err = tcp - goal
    running = cfg.w_ee * (err ** 2).sum(dim=(1, 2))
    effort = cfg.w_ctrl * (controls ** 2).sum(dim=(1, 2))
    terminal = cfg.w_term * (err[:, -1] ** 2).sum(dim=-1)
    return running + effort + terminal


class TorchMBDPlanner:
    """Cost-weighted annealed sampling over a fixed horizon, in torch.

    Args:
        horizon: T, the planned sequence length.
        act_dim: m, the number of input channels.
        limit: the symmetric input bound.
        schedule: the stage count and the noise endpoints.
        alpha, eta: temperature and step size.
        alpha_adaptive, alpha_scale: rescale the temperature to the spread of the
            candidate costs, alpha = alpha_scale * std(costs).
        generator: torch RNG, so a run is reproducible from a seed.
    """

    def __init__(self, horizon: int, act_dim: int, limit: float,
                 schedule: Schedule, alpha: float, eta: float = 1.0,
                 alpha_adaptive: bool = False, alpha_scale: float = 0.5,
                 generator: torch.Generator | None = None) -> None:
        self.horizon = horizon
        self.act_dim = act_dim
        self.limit = float(limit)
        self.schedule = schedule
        self.alpha = alpha
        self.eta = eta
        self.alpha_adaptive = alpha_adaptive
        self.alpha_scale = alpha_scale
        self.generator = generator
        self.sigmas = schedule.sigmas()

    # ------------------------------------------------------------------ pieces
    def temperature(self, costs: torch.Tensor) -> float:
        if not self.alpha_adaptive:
            return self.alpha
        spread = float(costs.std(unbiased=False))
        return self.alpha_scale * spread if spread > 1e-12 else self.alpha

    def _weights(self, costs: torch.Tensor) -> torch.Tensor:
        shifted = costs - costs.min()
        w = torch.exp(-shifted / self.temperature(costs))
        total = w.sum()
        if not torch.isfinite(total) or total <= 0:
            return torch.full_like(w, 1.0 / w.numel())
        return w / total

    def _propose(self, nominal: torch.Tensor, sigma: float,
                 num: int) -> torch.Tensor:
        eps = torch.randn(num, self.horizon, self.act_dim,
                          generator=self.generator)
        return (nominal.unsqueeze(0) + sigma * eps).clamp_(-self.limit,
                                                          self.limit)

    # -------------------------------------------------------------------- plan
    @torch.no_grad()
    def plan(self, nominal: torch.Tensor, cost_fn: CostFn,
             trace: list | None = None) -> torch.Tensor:
        """One control step: descend the stages and return the updated plan."""
        U = nominal.clone()
        for sigma in self.sigmas:
            candidates = self._propose(U, sigma, self.schedule.num_samples)
            costs = cost_fn(candidates)
            weights = self._weights(costs)
            mean = torch.einsum("k,ktm->tm", weights, candidates)
            step = self.eta * (mean - U)
            if trace is not None:
                trace.append(dict(
                    sigma=float(sigma),
                    alpha=float(self.temperature(costs)),
                    n_eff=float(1.0 / (weights ** 2).sum()),
                    dU=float(step.norm()),
                ))
            U = (U + step).clamp_(-self.limit, self.limit)
        return U

    @staticmethod
    def warm_start(U: torch.Tensor) -> torch.Tensor:
        """Shift the plan by one step and repeat the tail."""
        shifted = torch.roll(U, -1, dims=0)
        shifted[-1] = shifted[-2]
        return shifted


class FusedBilinearBackend:
    """Cost of a candidate batch under the fused rollout, all in torch.

    Pairs with :class:`TorchMBDPlanner`. Fastest, but the candidates come from a
    torch generator, so a run is not comparable trial by trial with the
    reference; use :class:`FastNumpyBackend` when that matters.
    """

    name = "bilinear_fast"

    def __init__(self, rollout, plant, cfg: PlannerCfg) -> None:
        self.rollout = rollout
        self.plant = plant
        self.cfg = cfg

    def cost_fn(self, state, goal) -> CostFn:
        b0 = torch.as_tensor(self.plant.observe(state), dtype=torch.float32)
        g = torch.as_tensor(goal, dtype=torch.float32)

        def evaluate(candidates: torch.Tensor) -> torch.Tensor:
            tcp = self.rollout.rollout_tracked(b0, candidates)
            return trajectory_cost(tcp, candidates, g, self.cfg)
        return evaluate


class FastNumpyBackend:
    """A fast rollout behind the reference planner's NumPy interface.

    Drops into ``exper.planner.MBDPlanner`` and ``exper.trial.run_trial``
    unchanged, so the candidate stream, the softmax and the weighted mean stay
    the float64 NumPy ones the paper's numbers were produced with. The only
    departure from the reference is the order in which the lifted step
    accumulates, which ``fast.bench`` measures.

    Everything that survives the NumPy planner survives here: the preallocated
    buffer, the precomputed transposes, the narrowed return, and for the bilinear
    class the channel fusion. The noise draw stays in NumPy on purpose.

    Takes any rollout from :mod:`fast.rollout`, so every learned condition of
    Sec. V-A is measured through the same path.
    """

    def __init__(self, rollout, plant, cfg: PlannerCfg,
                 name: str = "fast") -> None:
        self.rollout = rollout
        self.plant = plant
        self.cfg = cfg
        self.name = name
        self.model = rollout.model      # for probes that reach for the model

    def cost_fn(self, state, goal):
        import numpy as np

        from exper.backends import trajectory_cost as np_cost

        b0 = torch.as_tensor(self.plant.observe(state), dtype=torch.float32)

        def evaluate(candidates):
            u = torch.as_tensor(candidates, dtype=torch.float32)
            tcp = self.rollout.rollout_tracked(b0, u).numpy()
            return np_cost(tcp, np.asarray(candidates), goal, self.cfg)
        return evaluate
