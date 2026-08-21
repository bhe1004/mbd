"""Bilinear deep Koopman model: the learned dynamics BK-MBD plans through.

    lift    z = Psi(x) = [x, encoder(phi(x))]
    step    z' = A z + B0 u + sum_i u_i B_i z
    decode  x = C z = z[:feature_dim]

``phi`` is the encoder input transform, selected from the config:

``identity``
    the features themselves,
``sincos_prefix``
    ``[sin x[:n], cos x[:n]]`` -- for a manipulator whose first ``n`` features
    are joint angles, this removes the 2*pi wrap the encoder would otherwise
    have to learn (the arm setup used in the paper).

Each ``B_i`` is zero-initialized, so training starts exactly at the linear
model and adds bilinear terms only where they pay for themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .types import Array, ModelRollout

ENCODER_INPUTS = ("identity", "sincos_prefix")


@dataclass(frozen=True)
class KoopmanArchitecture:
    """Model shape (the ``koopman`` block of the config file)."""

    feature_dim: int
    action_dim: int
    lift_extra: int              # learned features appended to the raw state
    hidden_width: int = 96
    hidden_depth: int = 2
    encoder_input: str = "sincos_prefix"
    angle_dim: int = 7           # only used by sincos_prefix

    def __post_init__(self) -> None:
        if self.encoder_input not in ENCODER_INPUTS:
            raise ValueError(f"unknown encoder_input {self.encoder_input!r}")
        if self.lift_extra < 0:
            raise ValueError("lift_extra must be nonnegative")
        if self.hidden_depth < 0:
            raise ValueError("hidden_depth must be nonnegative")
        if self.encoder_input == "sincos_prefix" and not 0 < self.angle_dim <= self.feature_dim:
            raise ValueError("angle_dim must be in (0, feature_dim]")

    @property
    def lift_dim(self) -> int:
        return self.feature_dim + self.lift_extra

    @property
    def encoder_input_dim(self) -> int:
        if self.encoder_input == "sincos_prefix":
            return 2 * self.angle_dim
        return self.feature_dim

    def as_dict(self) -> dict:
        return {
            "feature_dim": self.feature_dim,
            "action_dim": self.action_dim,
            "lift_extra": self.lift_extra,
            "hidden_width": self.hidden_width,
            "hidden_depth": self.hidden_depth,
            "encoder_input": self.encoder_input,
            "angle_dim": self.angle_dim,
        }


class BilinearKoopman(nn.Module):
    """Deep lift + bilinear lifted dynamics."""

    def __init__(self, arch: KoopmanArchitecture) -> None:
        super().__init__()
        self.arch = arch

        layers: list[nn.Module] = []
        in_dim = arch.encoder_input_dim
        for _ in range(arch.hidden_depth):
            layers += [nn.Linear(in_dim, arch.hidden_width), nn.Tanh()]
            in_dim = arch.hidden_width
        layers.append(nn.Linear(in_dim, arch.lift_extra))
        self.encoder = nn.Sequential(*layers)

        n, m = arch.lift_dim, arch.action_dim
        self.A = nn.Parameter(torch.eye(n) + 0.01 * torch.randn(n, n))
        self.B0 = nn.Parameter(0.01 * torch.randn(n, m))
        self.Bs = nn.Parameter(torch.zeros(m, n, n))

    # ------------------------------------------------------------- lifted maps
    def encoder_inputs(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch.encoder_input == "sincos_prefix":
            q = x[..., : self.arch.angle_dim]
            return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)
        return x

    def lift(self, x: torch.Tensor) -> torch.Tensor:
        if self.arch.lift_extra == 0:
            return x
        return torch.cat([x, self.encoder(self.encoder_inputs(x))], dim=-1)

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        z_next = z @ self.A.T + u @ self.B0.T
        return z_next + torch.einsum("...i,ijk,...k->...j", u, self.Bs, z)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.arch.feature_dim]

    def rollout(self, z0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """z0 (..., lift_dim), U (..., T, action_dim) -> (..., T + 1, lift_dim)."""

        z = z0
        zs = [z]
        for t in range(U.shape[-2]):
            z = self.step(z, U[..., t, :])
            zs.append(z)
        return torch.stack(zs, dim=-2)

    # ---------------------------------------------------------- numpy exports
    def matrices(self) -> Tuple[Array, Array, Array]:
        """(A, B0, Bs) as float64 NumPy arrays, for analysis outside torch."""

        to_np = lambda p: p.detach().cpu().numpy().astype(np.float64)  # noqa: E731
        return to_np(self.A), to_np(self.B0), to_np(self.Bs)


class KoopmanPredictor:
    """Adapts :class:`BilinearKoopman` to the ``PredictiveModel`` protocol.

    The planner hands it the measured features; it lifts once, broadcasts the
    lifted state across the candidate batch, and rolls the whole batch in the
    lifted space with a single matmul chain per step.
    """

    def __init__(self, model: BilinearKoopman, device: torch.device | str = "cpu") -> None:
        self.model = model.to(device).eval()
        self.device = torch.device(device)

    def rollout(self, features0: Array, controls: torch.Tensor) -> ModelRollout:
        b0 = torch.as_tensor(np.asarray(features0, dtype=np.float32),
                             dtype=torch.float32, device=self.device)
        z0 = self.model.lift(b0).expand(controls.shape[0], -1)
        zs = self.model.rollout(z0, controls)
        return ModelRollout(features=self.model.decode(zs), latent=zs)
