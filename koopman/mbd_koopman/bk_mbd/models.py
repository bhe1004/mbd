"""Koopman models and parameter utilities.

Contains the shared deep lift `Psi_theta`, the linear and bilinear deep
Koopman dynamics, and NumPy-side shape checks used by rollout code.

The model follows `koopman/mppi_koopman/verify/unicycle_multiseed_v2.py`:

- lift: z = [x, encoder(x)] with a tanh MLP encoder,
- linear DK: z_next = A z + B0 u,
- bilinear BK: z_next = A z + B0 u + sum_i u_i B_i z,
- decode: C z = z[:state_dim].

One deviation from the reference is intentional: each bilinear `B_i` is
zero-initialized (the reference uses 0.01 * randn), so BK training starts from
the linear model and adds bilinear terms only when useful (see plan.md).
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn

from .config import KoopmanModelConfig
from .types import Array, BilinearKoopmanParams, LinearKoopmanParams


class DeepKoopmanModel(nn.Module):
    """Shared deep lift with linear or bilinear lifted dynamics.

    `x` denotes the Koopman training state (for the unicycle this is the
    4-dim base feature vector, not the raw 3-dim pose).

    Subclasses may override `encoder_inputs` / `encoder_input_dim` to feed a
    transformed version of the state to the encoder (see
    `ArmDeepKoopmanModel`). `kind` identifies the subclass in checkpoints.
    """

    kind = "default"

    def encoder_input_dim(self, config: KoopmanModelConfig) -> int:
        return config.state_dim

    def encoder_inputs(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def __init__(self, config: KoopmanModelConfig, *, bilinear: bool) -> None:
        super().__init__()
        self.config = config
        self.bilinear = bilinear

        layers: list[nn.Module] = []
        in_dim = self.encoder_input_dim(config)
        for _ in range(config.hidden_depth):
            layers.append(nn.Linear(in_dim, config.hidden_width))
            layers.append(nn.Tanh())
            in_dim = config.hidden_width
        layers.append(nn.Linear(in_dim, config.feature_dim))
        self.encoder = nn.Sequential(*layers)

        lift_dim = config.lift_dim
        action_dim = config.action_dim
        self.A = nn.Parameter(
            torch.eye(lift_dim) + 0.01 * torch.randn(lift_dim, lift_dim)
        )
        self.B0 = nn.Parameter(0.01 * torch.randn(lift_dim, action_dim))
        if bilinear:
            # Zero init: training starts exactly at the linear model.
            self.Bs = nn.Parameter(torch.zeros(action_dim, lift_dim, lift_dim))

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        """Psi_theta(x) = [x, encoder(encoder_inputs(x))]."""

        return torch.cat([x, self.encoder(self.encoder_inputs(x))], dim=-1)

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One lifted dynamics step for arbitrary batch shapes."""

        z_next = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            z_next = z_next + torch.einsum("...i,ijk,...k->...j", u, self.Bs, z)
        return z_next

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """C z = z[:state_dim]."""

        return z[..., : self.config.state_dim]

    def rollout(self, z0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """Roll lifted dynamics; U has shape (..., T, action_dim).

        Returns lifted states with shape (..., T + 1, lift_dim).
        """

        z = z0
        zs = [z]
        for t in range(U.shape[-2]):
            z = self.step(z, U[..., t, :])
            zs.append(z)
        return torch.stack(zs, dim=-2)

    def linear_params(self) -> LinearKoopmanParams:
        """Export NumPy parameters for the linear rollout backend."""

        if self.bilinear:
            raise ValueError("model is bilinear; use bilinear_params()")
        return LinearKoopmanParams(
            A=self.A.detach().cpu().numpy().astype(np.float64),
            B=self.B0.detach().cpu().numpy().astype(np.float64),
        )

    def bilinear_params(self) -> BilinearKoopmanParams:
        """Export NumPy parameters for the bilinear rollout backend."""

        if not self.bilinear:
            raise ValueError("model is linear; use linear_params()")
        return BilinearKoopmanParams(
            A=self.A.detach().cpu().numpy().astype(np.float64),
            B0=self.B0.detach().cpu().numpy().astype(np.float64),
            Bs=self.Bs.detach().cpu().numpy().astype(np.float64),
        )


class ArmDeepKoopmanModel(DeepKoopmanModel):
    """Arm variant: encoder consumes [sin q, cos q] of the 7 joint angles.

    Follows `koopman/mppi_koopman/verify/arm_multiseed_v2.py`, where the
    Koopman state is b = [q (7), ee (3)] and the encoder input is the
    14-dim [sin q, cos q].
    """

    kind = "arm_sincos"
    num_joints = 7

    def encoder_input_dim(self, config: KoopmanModelConfig) -> int:
        return 2 * self.num_joints

    def encoder_inputs(self, x: torch.Tensor) -> torch.Tensor:
        q = x[..., : self.num_joints]
        return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


MODEL_KINDS = {
    cls.kind: cls for cls in (DeepKoopmanModel, ArmDeepKoopmanModel)
}


def check_linear_params(params: LinearKoopmanParams) -> None:
    """Validate linear Koopman parameter shapes."""

    if params.A.ndim != 2 or params.A.shape[0] != params.A.shape[1]:
        raise ValueError("A must be square with shape (lift_dim, lift_dim)")
    if params.B.ndim != 2 or params.B.shape[0] != params.A.shape[0]:
        raise ValueError("B must have shape (lift_dim, action_dim)")


def check_bilinear_params(params: BilinearKoopmanParams) -> None:
    """Validate bilinear Koopman parameter shapes."""

    if params.A.ndim != 2 or params.A.shape[0] != params.A.shape[1]:
        raise ValueError("A must be square with shape (lift_dim, lift_dim)")
    if params.B0.ndim != 2 or params.B0.shape[0] != params.A.shape[0]:
        raise ValueError("B0 must have shape (lift_dim, action_dim)")
    if params.Bs.ndim != 3:
        raise ValueError("Bs must have shape (action_dim, lift_dim, lift_dim)")
    if params.Bs.shape[1:] != params.A.shape:
        raise ValueError("each B_i must have shape (lift_dim, lift_dim)")
    if params.Bs.shape[0] != params.B0.shape[1]:
        raise ValueError("number of B_i matrices must match action_dim")


def decode_state(z: Array, state_dim: int) -> Array:
    """Decode state by slicing z = [x, learned_features]."""

    return np.asarray(z)[..., :state_dim]


def zero_bilinear_matrices(action_dim: int, lift_dim: int) -> Array:
    """Create zero-initialized bilinear matrices B_i."""

    return np.zeros((action_dim, lift_dim, lift_dim), dtype=np.float64)

