"""Data containers and the two interfaces the MBD planner talks to.

Nothing in :mod:`mbd` imports MuJoCo, the FR3, or anything robot-specific.
The planner only ever sees:

``PredictiveModel``
    something that maps a batch of candidate control sequences to a batch of
    predicted *feature* trajectories (here features = [q, ee], but the
    algorithm never inspects them),
``CandidateCost``
    something that scores those feature trajectories against a goal.

Both are supplied by :mod:`env`. Swapping the simulator for a real robot, or
the reaching task for another one, means writing new implementations of these
two protocols -- no file under :mod:`mbd` changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Protocol

import numpy as np
import torch

Array = np.ndarray


@dataclass
class ModelRollout:
    """Output of one batched predictive rollout.

    Attributes:
        features: Predicted feature trajectories, shape (K, T + 1, feature_dim).
        latent: Optional lifted states, shape (K, T + 1, lift_dim), for models
            that have a lifted space; the planner does not require them.
    """

    features: torch.Tensor
    latent: Optional[torch.Tensor] = None


@dataclass
class PlanResult:
    """One finished plan."""

    controls: Array                      # (T, action_dim)
    predicted_features: Array            # (T + 1, feature_dim)
    best_cost: float = float("nan")
    info: Dict[str, Any] = field(default_factory=dict)


class PredictiveModel(Protocol):
    """Rolls candidate control sequences forward from the current features."""

    def rollout(self, features0: Array, controls: torch.Tensor) -> ModelRollout:
        """Args: features0 (feature_dim,), controls (K, T, action_dim)."""


class CandidateCost(Protocol):
    """Scores batched feature trajectories against a goal."""

    def __call__(
        self,
        features: torch.Tensor,      # (K, T + 1, feature_dim)
        controls: torch.Tensor,      # (K, T, action_dim)
        goal: Array,
        *,
        features_now: Array,         # the measured features the rollout started from
    ) -> torch.Tensor:               # (K,)
        ...
