"""Lifted rollout models.

One module holds the three rollout classes the experiments compare. They share
the exact-decoder convention: the observation occupies the leading coordinates
of the lifted state, so decoding is a slice and only the trailing coordinates
are learned.

    z = [b ; psi(tau(b))],   C z = z[:d] = b

``bilinear`` lets the input multiply the lifted state, ``linear`` keeps a
constant input matrix, and ``mlp`` is an unstructured baseline that predicts the
next observation directly.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn

VARIANTS = ("bilinear", "linear", "mlp")


def tau(b: torch.Tensor, num_joints: int) -> torch.Tensor:
    """Fixed task input map: sines and cosines of the configuration angles."""
    q = b[..., :num_joints]
    return torch.cat([torch.sin(q), torch.cos(q)], dim=-1)


def _mlp(in_dim: int, hidden: int, layers: int, out_dim: int) -> nn.Sequential:
    mods: list[nn.Module] = []
    d = in_dim
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.Tanh()]
        d = hidden
    mods.append(nn.Linear(d, out_dim))
    return nn.Sequential(*mods)


class LiftedModel(nn.Module):
    """Deep Koopman lift with a linear or bilinear lifted transition.

    Args:
        obs_dim: d, the width of the observation b.
        act_dim: m, the number of input channels.
        num_joints: how many leading entries of b are configuration angles.
        lift_dim: r, the width of the lifted state (must exceed obs_dim).
        hidden, layers: encoder shape.
        bilinear: attach the per-channel matrices {B_i}.
    """

    def __init__(self, obs_dim: int, act_dim: int, num_joints: int,
                 lift_dim: int, hidden: int = 96, layers: int = 2,
                 bilinear: bool = True) -> None:
        super().__init__()
        if lift_dim <= obs_dim:
            raise ValueError(f"lift_dim {lift_dim} must exceed obs_dim {obs_dim}")
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_joints = num_joints
        self.lift_dim = lift_dim
        self.bilinear = bilinear

        extra = lift_dim - obs_dim
        self.encoder = _mlp(2 * num_joints, hidden, layers, extra)
        self.A = nn.Parameter(torch.eye(lift_dim) + 0.01 * torch.randn(lift_dim, lift_dim))
        self.B0 = nn.Parameter(0.01 * torch.randn(lift_dim, act_dim))
        # zero-initialized, so training starts exactly at the linear model
        self.B = nn.Parameter(torch.zeros(act_dim, lift_dim, lift_dim)) \
            if bilinear else None

    # ------------------------------------------------------------------ pieces
    def lift(self, b: torch.Tensor) -> torch.Tensor:
        return torch.cat([b, self.encoder(tau(b, self.num_joints))], dim=-1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z[..., : self.obs_dim]

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One lifted step for a batch of states and inputs."""
        zn = z @ self.A.T + u @ self.B0.T
        if self.bilinear:
            # sum_i u_i (B_i z); one matmul per input channel
            for i in range(self.act_dim):
                zn = zn + u[..., i : i + 1] * (z @ self.B[i].T)
        return zn

    # ---------------------------------------------------------------- rollouts
    @torch.no_grad()
    def rollout(self, b0: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        """Propagate a batch of control sequences entirely in the lifted space.

        Args:
            b0: current observation, (d,) or (K, d).
            controls: (K, T, m).

        Returns:
            Decoded observations along the horizon, (K, T, d).
        """
        k = controls.shape[0]
        z = self.lift(b0.expand(k, -1) if b0.dim() == 1 else b0)
        out = []
        for t in range(controls.shape[1]):
            z = self.step(z, controls[:, t])
            out.append(self.decode(z))
        return torch.stack(out, dim=1)

    def unroll_train(self, b0: torch.Tensor, controls: torch.Tensor):
        """Differentiable unroll used by the multi-step loss.

        Yields (decoded, latent) at every step so the caller can score both the
        output error and the latent consistency term.
        """
        z = self.lift(b0)
        for t in range(controls.shape[1]):
            z = self.step(z, controls[:, t])
            yield self.decode(z), z


class MLPModel(nn.Module):
    """Unstructured next-observation predictor, as a generic nonlinear control."""

    def __init__(self, obs_dim: int, act_dim: int, num_joints: int,
                 hidden: int = 96, layers: int = 2) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.num_joints = num_joints
        self.lift_dim = obs_dim
        self.bilinear = False
        self.net = _mlp(2 * num_joints + obs_dim + act_dim, hidden, layers, obs_dim)

    def _features(self, b: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return torch.cat([b, tau(b, self.num_joints), u], dim=-1)

    def lift(self, b: torch.Tensor) -> torch.Tensor:
        return b

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        return z + self.net(self._features(z, u))

    @torch.no_grad()
    def rollout(self, b0: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        k = controls.shape[0]
        b = b0.expand(k, -1) if b0.dim() == 1 else b0
        out = []
        for t in range(controls.shape[1]):
            b = self.step(b, controls[:, t])
            out.append(b)
        return torch.stack(out, dim=1)

    def unroll_train(self, b0: torch.Tensor, controls: torch.Tensor):
        b = b0
        for t in range(controls.shape[1]):
            b = self.step(b, controls[:, t])
            yield b, b


def build_model(variant: str, obs_dim: int, act_dim: int, num_joints: int,
                lift_dim: int, hidden: int, layers: int) -> nn.Module:
    if variant not in VARIANTS:
        raise SystemExit(f"unknown model variant '{variant}', pick from {VARIANTS}")
    if variant == "mlp":
        return MLPModel(obs_dim, act_dim, num_joints, hidden, layers)
    return LiftedModel(obs_dim, act_dim, num_joints, lift_dim, hidden, layers,
                       bilinear=(variant == "bilinear"))


# ------------------------------------------------------------------ perturbation
class FrozenGainModel(nn.Module):
    """A bilinear model whose input gain is partly frozen at a fixed lifted state.

    The bilinear step's input response, ``sum_i u_i (B_i z)``, is state-dependent
    through ``B_i z``. Replacing ``z`` there by a fixed reference ``z_bar`` removes
    that state dependence while preserving the response magnitude, so ``lam`` mixes
    the two:

        input response  <-  sum_i u_i B_i ( (1-lam) z + lam z_bar )

    ``lam = 0`` is the trained model; ``lam = 1`` freezes the input gain at
    ``z_bar``, leaving a response that no longer varies with the configuration but
    keeps the size of the trained one. This is the coupling error the paper's
    argument is about, injected without shrinking the response the way a plain
    scaling of ``B_i`` would.
    """

    def __init__(self, base: "LiftedModel", z_bar: torch.Tensor, lam: float) -> None:
        super().__init__()
        self.base = base
        self.lam = float(lam)
        self.register_buffer("z_bar", z_bar.detach().clone())
        # mirror the attributes the backends and rollout expect
        self.obs_dim = base.obs_dim
        self.act_dim = base.act_dim
        self.num_joints = base.num_joints
        self.lift_dim = base.lift_dim
        self.bilinear = True

    def lift(self, b):
        return self.base.lift(b)

    def decode(self, z):
        return self.base.decode(z)

    def step(self, z, u):
        z_eff = (1.0 - self.lam) * z + self.lam * self.z_bar
        zn = z @ self.base.A.T + u @ self.base.B0.T
        for i in range(self.act_dim):
            zn = zn + u[..., i : i + 1] * (z_eff @ self.base.B[i].T)
        return zn

    @torch.no_grad()
    def rollout(self, b0, controls):
        k = controls.shape[0]
        z = self.lift(b0.expand(k, -1) if b0.dim() == 1 else b0)
        out = []
        for t in range(controls.shape[1]):
            z = self.step(z, controls[:, t])
            out.append(self.decode(z))
        return torch.stack(out, dim=1)


@torch.no_grad()
def visited_mean_lift(model: "LiftedModel", obs: torch.Tensor) -> torch.Tensor:
    """Mean lifted state over a batch of visited observations, the freeze point."""
    return model.lift(obs).mean(dim=0)


@torch.no_grad()
def freeze_gain(model: nn.Module, z_bar: torch.Tensor, lam: float) -> nn.Module:
    """Inject configuration-coupling error by freezing the input gain (size-preserving).

    ``lam = 0`` returns the trained model unchanged; ``lam > 0`` returns a wrapper
    that mixes the state-dependent gain with one frozen at ``z_bar``.
    """
    if lam == 0.0:
        return model
    if not getattr(model, "bilinear", False):
        raise SystemExit("freeze_gain needs a bilinear model")
    wrapped = FrozenGainModel(model, z_bar, lam)
    wrapped.eval()
    return wrapped


def scale_bilinear(model: nn.Module, lam: float) -> nn.Module:
    """Deprecated size-changing perturbation; kept for reference.

    Shrinks the state-dependent part of the input gain toward zero, which also
    shrinks the response magnitude. Prefer :func:`freeze_gain`, which preserves
    the magnitude and isolates the configuration dependence.
    """
    if lam == 0.0:
        return model
    if not getattr(model, "bilinear", False):
        raise SystemExit("scale_bilinear needs a bilinear model")
    import copy

    perturbed = copy.deepcopy(model)
    perturbed.B.mul_(1.0 - lam)
    perturbed.eval()
    return perturbed
