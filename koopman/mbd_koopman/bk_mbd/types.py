"""Common data containers and protocols."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Protocol, Tuple

import numpy as np


Array = np.ndarray


@dataclass
class RolloutResult:
    """Rollout output for one or more candidate control sequences."""

    states: Array
    controls: Array
    costs: Array | None = None
    tubes: Array | None = None
    info: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LinearKoopmanParams:
    """Parameters for z_next = A z + B u."""

    A: Array
    B: Array


@dataclass(frozen=True)
class BilinearKoopmanParams:
    """Parameters for z_next = A z + B0 u + sum_i u_i B_i z."""

    A: Array
    B0: Array
    Bs: Array


@dataclass(frozen=True)
class TubeConstants:
    """Scalar proportional residual bound constants."""

    c_x: float
    c_u: float


class TaskEnv(Protocol):
    """Minimal task interface expected by experiment scripts."""

    state_dim: int
    action_dim: int
    action_bounds: Tuple[Array, Array]

    def true_step(self, x: Array, u: Array) -> Array:
        ...

    def true_rollout(self, x0: Array, U: Array) -> Array:
        ...

    def cost(self, xs: Array, U: Array, goal: Array) -> float:
        ...

    def success(self, x_final: Array, goal: Array) -> bool:
        ...

    def sample_dataset(self, seed: int) -> Dict[str, Array]:
        ...

