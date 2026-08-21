"""Rollout backends.

Each backend turns a batch of candidate control sequences into a batch of costs.
The cost itself is shared, so conditions differ only in how the predicted output
is produced:

    oracle    the plant, batched through MuJoCo
    model     a lifted or unstructured learned model
    split     a linear lifted model whose configuration is decoded and then
              passed through the analytic forward kinematics, which hands the
              linear class the configuration-dependent coupling it cannot learn
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import torch

from .config import PlannerCfg
from .plant import FrankaPlant

Array = np.ndarray


def trajectory_cost(tcp: Array, controls: Array, goal: Array,
                    cfg: PlannerCfg) -> Array:
    """Cost of a batch of predicted paths, (K, T, 3) and (K, T, m) -> (K,)."""
    err = tcp - goal[None, None, :]
    running = cfg.w_ee * np.sum(err ** 2, axis=(1, 2))
    effort = cfg.w_ctrl * np.sum(controls ** 2, axis=(1, 2))
    terminal = cfg.w_term * np.sum(err[:, -1] ** 2, axis=-1)
    return running + effort + terminal


class OracleBackend:
    """Roll the true dynamics; the upper reference for accuracy and the cost."""

    name = "oracle"

    def __init__(self, plant: FrankaPlant, cfg: PlannerCfg) -> None:
        self.plant = plant
        self.cfg = cfg

    def cost_fn(self, state: Array, goal: Array) -> Callable[[Array], Array]:
        def evaluate(candidates: Array) -> Array:
            obs = self.plant.rollout_true(state, candidates)
            return trajectory_cost(obs[..., self.plant.num_joints :],
                                   candidates, goal, self.cfg)
        return evaluate


class ModelBackend:
    """Roll a learned model entirely in its own space."""

    def __init__(self, plant: FrankaPlant, cfg: PlannerCfg, model, name: str) -> None:
        self.plant = plant
        self.cfg = cfg
        self.model = model
        self.name = name

    def cost_fn(self, state: Array, goal: Array) -> Callable[[Array], Array]:
        b0 = torch.as_tensor(self.plant.observe(state), dtype=torch.float32)

        def evaluate(candidates: Array) -> Array:
            u = torch.as_tensor(candidates, dtype=torch.float32)
            obs = self.model.rollout(b0, u).numpy()
            return trajectory_cost(obs[..., self.plant.num_joints :],
                                   candidates, goal, self.cfg)
        return evaluate


class SplitBackend:
    """Linear lifted joints plus analytic forward kinematics.

    The learned part predicts only the configuration, into which the input
    enters state-independently, and the tracked output is recovered by the
    analytic map. This isolates the configuration-dependent coupling as the one
    thing the linear class is missing.
    """

    name = "split"

    def __init__(self, plant: FrankaPlant, cfg: PlannerCfg, model) -> None:
        self.plant = plant
        self.cfg = cfg
        self.model = model

    def cost_fn(self, state: Array, goal: Array) -> Callable[[Array], Array]:
        b0 = torch.as_tensor(self.plant.observe(state), dtype=torch.float32)
        nj = self.plant.num_joints

        def evaluate(candidates: Array) -> Array:
            u = torch.as_tensor(candidates, dtype=torch.float32)
            with torch.no_grad():
                obs = self.model.rollout(b0, u)
                tcp = self.plant.task.forward_kinematics_torch(obs[..., :nj])
            return trajectory_cost(tcp.numpy(), candidates, goal, self.cfg)
        return evaluate


def build_backend(kind: str, plant: FrankaPlant, cfg: PlannerCfg, model=None):
    """Dispatch by condition name."""
    if kind == "oracle":
        return OracleBackend(plant, cfg)
    if kind == "split":
        if model is None:
            raise SystemExit("split backend needs a linear model")
        return SplitBackend(plant, cfg, model)
    if model is None:
        raise SystemExit(f"backend '{kind}' needs a model")
    return ModelBackend(plant, cfg, model, kind)
