"""Model-Based Diffusion optimizer over control sequences.

One plan = ``num_diffusion_steps`` rounds of: perturb the current sequence with
noise sigma_s, score every candidate, and denoise toward the softmax-weighted
mean. sigma anneals high -> low *within* the plan (explore -> refine).

With ``eta_relative`` (default) the score step is ``eta_s = eta * sigma_s^2``,
so ``eta = 1`` without Langevin noise reproduces the weighted-mean update
exactly -- the algebraic reduction of the original MBD reverse step.

:class:`AdaptiveNoise` adds a SECOND, across-plans shrink: near the goal the
whole schedule is scaled down, so a converged plan is not re-blasted with the
full sigma_start every replan (which otherwise leaves the tool jittering tens
of mm short).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Dict, Optional

import numpy as np

from .types import Array

EvaluateCandidates = Callable[[Array], Array]

UPDATE_RULES = ("score_langevin", "weighted_mean")


@dataclass(frozen=True)
class MBDSettings:
    """Sampler settings (mirrors the ``mbd`` block of the config file)."""

    num_samples: int = 512
    num_diffusion_steps: int = 8
    sigma_start: float = 3.0
    sigma_end: float = 0.1
    alpha: float = 0.03
    eta: float = 1.0
    update_rule: str = "score_langevin"
    langevin_noise: bool = False
    eta_relative: bool = True
    seed: int = 0

    def __post_init__(self) -> None:
        if self.update_rule not in UPDATE_RULES:
            raise ValueError(f"unknown update_rule {self.update_rule!r}")
        if self.sigma_start <= 0.0 or self.sigma_end <= 0.0:
            raise ValueError("sigma_start / sigma_end must be positive")
        if self.num_samples < 2:
            raise ValueError("num_samples must be >= 2")


def softmax_from_cost(costs: Array, alpha: float) -> Array:
    """softmax(-cost / alpha), stabilized by subtracting the minimum cost."""

    c = np.asarray(costs, dtype=np.float64)
    weights = np.exp(-(c - np.min(c)) / max(alpha, 1e-12))
    denom = float(np.sum(weights))
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(c, 1.0 / c.size)
    return weights / denom


@dataclass
class OptimizeResult:
    controls: Array
    best_candidate: Array
    best_cost: float
    history: Dict[str, Array]


class MBDOptimizer:
    """Diffusion-style optimizer over a control sequence U of shape (T, m)."""

    def __init__(self, settings: MBDSettings, action_low: Array, action_high: Array) -> None:
        self.settings = settings
        self.action_low = np.asarray(action_low, dtype=np.float64)
        self.action_high = np.asarray(action_high, dtype=np.float64)

    def sigma_schedule(self) -> Array:
        return np.linspace(
            self.settings.sigma_start,
            self.settings.sigma_end,
            self.settings.num_diffusion_steps,
            dtype=np.float64,
        )

    def optimize(
        self,
        initial_controls: Array,
        evaluate_candidates: EvaluateCandidates,
        *,
        rng: Optional[np.random.Generator] = None,
    ) -> OptimizeResult:
        cfg = self.settings
        rng = np.random.default_rng(cfg.seed) if rng is None else rng
        U = np.clip(np.asarray(initial_controls, dtype=np.float64).copy(),
                    self.action_low, self.action_high)

        weighted_cost_history, best_cost_history, sigma_history = [], [], []
        best_candidate = U.copy()
        best_cost = np.inf

        for sigma in self.sigma_schedule():
            eps = rng.normal(size=(cfg.num_samples,) + U.shape)
            candidates = np.clip(U[None, ...] + sigma * eps,
                                 self.action_low, self.action_high)

            costs = np.asarray(evaluate_candidates(candidates), dtype=np.float64)
            if costs.shape != (cfg.num_samples,):
                raise ValueError("evaluate_candidates must return shape (num_samples,)")

            weights = softmax_from_cost(costs, cfg.alpha)
            weighted_mean = np.einsum("k,kij->ij", weights, candidates)

            idx = int(np.argmin(costs))
            if float(costs[idx]) < best_cost:
                best_cost = float(costs[idx])
                best_candidate = candidates[idx].copy()

            if cfg.update_rule == "weighted_mean":
                U = weighted_mean
            else:  # score_langevin
                sigma_sq = max(float(sigma) ** 2, 1e-12)
                score = (weighted_mean - U) / sigma_sq
                eta_s = cfg.eta * sigma_sq if cfg.eta_relative else cfg.eta
                U = U + eta_s * score
                if cfg.langevin_noise:
                    U = U + np.sqrt(2.0 * eta_s) * rng.normal(size=U.shape)

            U = np.clip(U, self.action_low, self.action_high)
            weighted_cost_history.append(float(np.average(costs, weights=weights)))
            best_cost_history.append(best_cost)
            sigma_history.append(float(sigma))

        return OptimizeResult(
            controls=U,
            best_candidate=best_candidate,
            best_cost=best_cost,
            history={
                "weighted_cost": np.asarray(weighted_cost_history),
                "best_cost": np.asarray(best_cost_history),
                "sigma": np.asarray(sigma_history),
            },
        )


class AdaptiveNoise:
    """Scale both ends of the sigma schedule by ``clip(err/err_full, floor, 1)``.

    Optimizers are cached by rounded scale, so a new one is not built per plan.
    Disabled -> the base optimizer is used verbatim (the fixed-schedule
    protocol).
    """

    def __init__(self, optimizer: MBDOptimizer, *, enabled: bool,
                 err_full: float, floor: float) -> None:
        self.optimizer = optimizer
        self.enabled = bool(enabled)
        self.err_full = float(err_full)
        self.floor = float(floor)
        self._cache: Dict[float, MBDOptimizer] = {}

    def optimize(self, U: Array, evaluate: EvaluateCandidates,
                 rng: np.random.Generator, err: float) -> OptimizeResult:
        if not self.enabled:
            return self.optimizer.optimize(U, evaluate, rng=rng)
        scale = min(1.0, max(err / max(self.err_full, 1e-12), self.floor))
        key = round(scale, 2)
        opt = self._cache.get(key)
        if opt is None:
            base = self.optimizer.settings
            opt = MBDOptimizer(
                replace(base,
                        sigma_start=base.sigma_start * scale,
                        sigma_end=base.sigma_end * scale),
                self.optimizer.action_low,
                self.optimizer.action_high,
            )
            self._cache[key] = opt
        return opt.optimize(U, evaluate, rng=rng)
