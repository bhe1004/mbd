"""Common Model-Based Diffusion optimizer over control sequences."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

import numpy as np

from .config import MBDConfig, UpdateRule
from .costs import stable_softmax_from_cost
from .types import Array


EvaluateCandidates = Callable[[Array], Array]


@dataclass
class MBDOptimizeResult:
    """Result from optimizing one control sequence."""

    controls: Array
    best_candidate: Array
    best_cost: float
    history: Dict[str, Array] = field(default_factory=dict)


class MBDOptimizer:
    """Diffusion-style optimizer shared by all rollout backends."""

    def __init__(
        self,
        config: MBDConfig,
        action_low: Array,
        action_high: Array,
    ) -> None:
        self.config = config
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)

    def sigma_schedule(self) -> Array:
        """Linear high-to-low noise schedule."""

        return np.linspace(
            self.config.sigma_start,
            self.config.sigma_end,
            self.config.num_diffusion_steps,
            dtype=np.float64,
        )

    def optimize(
        self,
        initial_controls: Array,
        evaluate_candidates: EvaluateCandidates,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> MBDOptimizeResult:
        """Optimize U by repeatedly sampling, scoring, and denoising."""

        rng = np.random.default_rng(self.config.seed) if rng is None else rng
        U = np.asarray(initial_controls, dtype=np.float64).copy()
        U = np.clip(U, self.action_low, self.action_high)

        cost_history = []
        best_cost_history = []
        sigma_history = []
        best_candidate = U.copy()
        best_cost = np.inf

        for sigma in self.sigma_schedule():
            eps = rng.normal(
                loc=0.0,
                scale=1.0,
                size=(self.config.num_samples,) + U.shape,
            )
            candidates = U[None, ...] + sigma * eps
            candidates = np.clip(candidates, self.action_low, self.action_high)

            costs = np.asarray(evaluate_candidates(candidates), dtype=np.float64)
            if costs.shape != (self.config.num_samples,):
                raise ValueError("evaluate_candidates must return shape (num_samples,)")

            weights = stable_softmax_from_cost(costs, self.config.alpha)
            weighted_mean = np.einsum("k,kij->ij", weights, candidates)

            idx = int(np.argmin(costs))
            if float(costs[idx]) < best_cost:
                best_cost = float(costs[idx])
                best_candidate = candidates[idx].copy()

            if self.config.update_rule == UpdateRule.WEIGHTED_MEAN:
                U = weighted_mean
            elif self.config.update_rule == UpdateRule.SCORE_LANGEVIN:
                sigma_sq = max(float(sigma**2), 1e-12)
                score = (weighted_mean - U) / sigma_sq
                # eta_s = eta * sigma_s^2 (relative, default) or eta (absolute).
                eta_s = (
                    self.config.eta * sigma_sq
                    if self.config.eta_relative
                    else self.config.eta
                )
                U = U + eta_s * score
                if self.config.add_langevin_noise:
                    U = U + np.sqrt(2.0 * eta_s) * rng.normal(size=U.shape)
            else:
                raise ValueError(f"unknown update rule: {self.config.update_rule}")

            U = np.clip(U, self.action_low, self.action_high)
            cost_history.append(float(np.average(costs, weights=weights)))
            best_cost_history.append(best_cost)
            sigma_history.append(float(sigma))

        return MBDOptimizeResult(
            controls=U,
            best_candidate=best_candidate,
            best_cost=best_cost,
            history={
                "weighted_cost": np.asarray(cost_history),
                "best_cost": np.asarray(best_cost_history),
                "sigma": np.asarray(sigma_history),
            },
        )

