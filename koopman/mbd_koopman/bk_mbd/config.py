"""Shared configuration objects for BK-MBD experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional, Tuple


class MethodName(str, Enum):
    """Planner variants used in the comparison.

    DK_MBD_SPLIT reproduces the MPPI-DK paper's structure-informed setup:
    the linear Koopman model learns only the part where the input enters
    state-independently, and the first-order coupling (heading rotation /
    forward kinematics) is handled analytically outside the learned model.
    """

    VANILLA_MBD_TRUE = "vanilla_mbd_true"
    DK_MBD = "dk_mbd"
    DK_MBD_SPLIT = "dk_mbd_split"
    BK_MBD = "bk_mbd"


class UpdateRule(str, Enum):
    """MBD control-sequence update variants."""

    SCORE_LANGEVIN = "score_langevin"
    WEIGHTED_MEAN = "weighted_mean"


@dataclass(frozen=True)
class TaskConfig:
    """Task-level settings shared by all methods."""

    name: str
    state_dim: int
    action_dim: int
    horizon: int
    dt: float
    action_low: Tuple[float, ...]
    action_high: Tuple[float, ...]
    closed_loop_steps: int

    def __post_init__(self) -> None:
        if len(self.action_low) != self.action_dim:
            raise ValueError("action_low length must match action_dim")
        if len(self.action_high) != self.action_dim:
            raise ValueError("action_high length must match action_dim")


@dataclass(frozen=True)
class MBDConfig:
    """Diffusion-style optimizer settings.

    With `eta_relative=True` (default) the score step size is
    eta_s = eta * sigma_s^2, so eta = 1.0 without Langevin noise reproduces
    the weighted-mean update exactly (the algebraic reduction of the original
    MBD reverse step). `eta_relative=False` keeps a fixed absolute step for
    debugging.
    """

    num_samples: int
    num_diffusion_steps: int
    sigma_start: float
    sigma_end: float
    alpha: float
    eta: float
    update_rule: UpdateRule = UpdateRule.SCORE_LANGEVIN
    seed: int = 0
    add_langevin_noise: bool = True
    eta_relative: bool = True


@dataclass(frozen=True)
class KoopmanModelConfig:
    """Deep Koopman model shape."""

    state_dim: int
    action_dim: int
    lift_dim: int
    hidden_width: int
    hidden_depth: int = 2

    @property
    def feature_dim(self) -> int:
        return self.lift_dim - self.state_dim

    def __post_init__(self) -> None:
        if self.lift_dim < self.state_dim:
            raise ValueError("lift_dim must be at least state_dim")
        if self.hidden_depth < 0:
            raise ValueError("hidden_depth must be nonnegative")


@dataclass(frozen=True)
class KoopmanTrainConfig:
    """Training settings shared by linear DK and bilinear BK."""

    rollout_horizon: int
    gamma_latent: float
    batch_size: int
    num_epochs: int
    learning_rate: float
    weight_decay: float = 0.0
    val_fraction: float = 0.2
    seed: int = 0
    grad_clip: float = 0.0
    cosine_lr: bool = False


@dataclass(frozen=True)
class TubeConfig:
    """Empirical error-tube settings for BK-MBD."""

    beta_e: float
    quantile: float = 0.99
    min_constant: float = 1e-8

    def __post_init__(self) -> None:
        if not 0.0 < self.quantile <= 1.0:
            raise ValueError("quantile must be in (0, 1]")


@dataclass(frozen=True)
class ExperimentConfig:
    """Top-level experiment wiring."""

    task: TaskConfig
    mbd: MBDConfig
    koopman_model: Optional[KoopmanModelConfig]
    koopman_train: Optional[KoopmanTrainConfig]
    tube: Optional[TubeConfig]
    output_dir: Path = field(
        default_factory=lambda: Path(__file__).resolve().parents[1] / "out"
    )

