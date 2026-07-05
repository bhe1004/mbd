"""Shared cost and weighting utilities."""

from __future__ import annotations

import numpy as np

from .types import Array


def wrap_angle(theta: Array) -> Array:
    """Wrap angles to [-pi, pi]."""

    return (theta + np.pi) % (2.0 * np.pi) - np.pi


def control_effort(U: Array, weight: float) -> float:
    """Quadratic control effort."""

    return float(weight * np.sum(np.asarray(U) ** 2))


def stable_softmax_from_cost(costs: Array, alpha: float) -> Array:
    """Compute softmax(-cost / alpha) with min-cost stabilization."""

    costs_arr = np.asarray(costs, dtype=np.float64)
    shifted = costs_arr - np.min(costs_arr)
    logits = -shifted / max(alpha, 1e-12)
    weights = np.exp(logits)
    denom = np.sum(weights)
    if denom <= 0.0 or not np.isfinite(denom):
        return np.full_like(costs_arr, 1.0 / costs_arr.size)
    return weights / denom

