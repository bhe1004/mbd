"""Annealed sampling planner.

One implementation of the Model-Based Diffusion optimizer serves every
condition. What differs across conditions is the rollout backend that maps a
candidate control sequence to a cost, and the noise schedule that governs how
the candidates are drawn.

A schedule is (stages, sigma_start, sigma_end):

    anneal   S stages, sigma_start > sigma_end
    wide     S stages, sigma_start = sigma_end = wide
    narrow   S stages, sigma_start = sigma_end = narrow
    single   1 stage at N*S samples, so the sample budget is held fixed
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

Array = np.ndarray
CostFn = Callable[[Array], Array]   # (K, T, m) -> (K,)


@dataclass(frozen=True)
class Schedule:
    """How the noise is drawn across the stages of one control step."""

    name: str
    stages: int
    num_samples: int
    sigma_start: float
    sigma_end: float

    def sigmas(self) -> list[float]:
        if self.stages == 1:
            return [self.sigma_start]
        step = (self.sigma_end - self.sigma_start) / (self.stages - 1)
        return [self.sigma_start + i * step for i in range(self.stages)]

    @property
    def total_samples(self) -> int:
        return self.stages * self.num_samples


def schedules_at_equal_budget(num_samples: int, stages: int,
                              wide: float, narrow: float) -> list[Schedule]:
    """The five conditions of the schedule study, at one total sample budget."""
    total = num_samples * stages
    return [
        Schedule("single_wide", 1, total, wide, wide),
        Schedule("single_narrow", 1, total, narrow, narrow),
        Schedule("staged_wide", stages, num_samples, wide, wide),
        Schedule("staged_narrow", stages, num_samples, narrow, narrow),
        Schedule("anneal", stages, num_samples, wide, narrow),
    ]


def schedule_by_name(schedules: Sequence[Schedule], name: str) -> Schedule:
    """Look up a schedule by name so callers never depend on list order."""
    for s in schedules:
        if s.name == name:
            return s
    raise KeyError(f"no schedule named '{name}' in {[s.name for s in schedules]}")


class MBDPlanner:
    """Cost-weighted annealed sampling over a fixed horizon."""

    def __init__(self, horizon: int, act_dim: int, limit: float,
                 schedule: Schedule, alpha: float, eta: float = 1.0,
                 alpha_adaptive: bool = False, alpha_scale: float = 0.5) -> None:
        self.horizon = horizon
        self.act_dim = act_dim
        self.limit = limit
        self.schedule = schedule
        self.alpha = alpha
        self.eta = eta
        self.alpha_adaptive = alpha_adaptive
        self.alpha_scale = alpha_scale

    # ------------------------------------------------------------------ pieces
    def temperature(self, costs: Array) -> float:
        """The temperature this stage weighs its candidates with.

        Fixed by default. Under ``alpha_adaptive`` it is rescaled to the spread
        of the candidate costs, alpha = alpha_scale * std(costs), which is the
        adaptive rule MPPI-DK reports. Near the goal the spread collapses, so a
        fixed alpha flattens the softmax while the adaptive one does not.
        """
        if not self.alpha_adaptive:
            return self.alpha
        spread = float(np.std(costs))
        # a degenerate population carries no information either way; fall back
        # to the fixed scale rather than dividing by zero
        return self.alpha_scale * spread if spread > 1e-12 else self.alpha

    def _weights(self, costs: Array) -> Array:
        shifted = costs - costs.min()
        w = np.exp(-shifted / self.temperature(costs))
        total = w.sum()
        return np.full_like(w, 1.0 / len(w)) if total <= 0 else w / total

    def _propose(self, nominal: Array, sigma: float, num: int,
                 rng: np.random.Generator) -> Array:
        eps = rng.standard_normal((num, self.horizon, self.act_dim))
        return np.clip(nominal[None] + sigma * eps, -self.limit, self.limit)

    # -------------------------------------------------------------------- plan
    def plan(self, nominal: Array, cost_fn: CostFn,
             rng: np.random.Generator, trace: list | None = None) -> Array:
        """One control step: descend the stages and return the updated plan.

        ``trace``, when given, receives one record per stage holding the
        quantities the precision floor is written in: the effective sample size
        the softmax realizes, the displacement the stage produces, and how much
        of the candidate cloud sits on the input bound. Purely observational.
        """
        U = np.array(nominal, dtype=np.float64, copy=True)
        for sigma in self.schedule.sigmas():
            candidates = self._propose(U, sigma, self.schedule.num_samples, rng)
            costs = np.asarray(cost_fn(candidates), dtype=np.float64)
            weights = self._weights(costs)
            mean = np.tensordot(weights, candidates, axes=(0, 0))
            step = self.eta * (mean - U)
            if trace is not None:
                trace.append(dict(
                    sigma=float(sigma),
                    alpha=float(self.temperature(costs)),
                    n_eff=effective_sample_size(weights),
                    dU=float(np.linalg.norm(step)),
                    dU0=step[0].copy(),
                    clip_frac=float(np.mean(
                        np.abs(candidates) >= self.limit - 1e-9)),
                ))
            U = np.clip(U + step, -self.limit, self.limit)
        return U

    @staticmethod
    def warm_start(U: Array) -> Array:
        """Shift the plan by one step and repeat the tail."""
        shifted = np.roll(U, -1, axis=0)
        shifted[-1] = shifted[-2]
        return shifted


# ------------------------------------------------------------------ diagnostics
def first_action_dispersion(planner: MBDPlanner, nominal: Array, cost_fn: CostFn,
                            repeats: int, rng: np.random.Generator) -> float:
    """Per-axis spread of the first action over repeated plans from one state.

    This is the quantity the precision floor bounds: at a fixed noise level the
    executed command inherits it at every control step.
    """
    firsts = [planner.plan(nominal, cost_fn, rng)[0] for _ in range(repeats)]
    return float(np.mean(np.std(np.stack(firsts), axis=0)))


def effective_sample_size(weights: Array) -> float:
    return float(1.0 / np.sum(np.asarray(weights) ** 2))
