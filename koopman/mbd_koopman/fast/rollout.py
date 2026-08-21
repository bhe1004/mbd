"""Rollouts with the layout tidied, one class per rollout class of Sec. V-A.

Three things are done everywhere they apply, so the conditions stay comparable
with each other:

* the output buffer is preallocated instead of built as a list and stacked,
* the transposes the step needs are precomputed once instead of per step,
* only what the cost reads leaves torch, rather than the whole observation.

One thing applies to the bilinear class alone, because only that class has the
term to fuse: ``LiftedModel.step`` adds ``sum_i u_i (B_i z)`` channel by channel,
so a lifted step issues ``m`` products of shape ``(N, r) @ (r, r)``. Stacking
``{B_i}`` sideways into one ``(r, m*r)`` matrix computes every channel in a
single product. The linear class has no such term and the unstructured network
has no lifted step at all, so there is nothing there to fuse; those two get the
three shared items and nothing else.

Every class here wraps a model that ``exper.training.get_model`` already
returned. No checkpoint is retrained or rewritten.
"""

from __future__ import annotations

import torch

from exper.koopman import LiftedModel, MLPModel, tau


class FastLifted:
    """Linear or bilinear lifted rollout.

    Args:
        model: a trained :class:`~exper.koopman.LiftedModel`. The bilinear case
            additionally gets its input channels fused.
        num_joints: how many leading observation entries are joint angles, which
            fixes where the tracked output starts.
    """

    def __init__(self, model: LiftedModel, num_joints: int) -> None:
        self.model = model
        self.obs_dim = model.obs_dim
        self.act_dim = model.act_dim
        self.lift_dim = model.lift_dim
        self.num_joints = num_joints
        self.bilinear = bool(getattr(model, "bilinear", False))

        with torch.no_grad():
            self.A_T = model.A.T.contiguous()
            self.B0_T = model.B0.T.contiguous()
            # (r, m*r): column block i holds B_i^T, so one product yields every
            # channel's B_i z at once. Absent in the linear class.
            self.B_cat = torch.cat(
                [model.B[i].T.contiguous() for i in range(self.act_dim)],
                dim=1).contiguous() if self.bilinear else None

    @torch.no_grad()
    def step(self, z: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """One lifted step for a batch, (K, r) and (K, m) -> (K, r)."""
        zn = z @ self.A_T + u @ self.B0_T
        if self.B_cat is None:
            return zn
        channels = (z @ self.B_cat).view(z.shape[0], self.act_dim, self.lift_dim)
        return zn + torch.einsum("km,kmr->kr", u, channels)

    @torch.no_grad()
    def _run(self, b0: torch.Tensor, controls: torch.Tensor,
             lo: int, hi: int) -> torch.Tensor:
        k = controls.shape[0]
        z = self.model.lift(b0.expand(k, -1) if b0.dim() == 1 else b0)
        out = torch.empty(k, controls.shape[1], hi - lo)
        for t in range(controls.shape[1]):
            z = self.step(z, controls[:, t])
            out[:, t] = z[:, lo:hi]
        return out

    def rollout(self, b0, controls):
        """Decoded observations along the horizon, (K, T, d)."""
        return self._run(b0, controls, 0, self.obs_dim)

    def rollout_tracked(self, b0, controls):
        """Only the tracked output, (K, T, d_y)."""
        return self._run(b0, controls, self.num_joints, self.obs_dim)

    def rollout_joints(self, b0, controls):
        """Only the configuration block, (K, T, n_j), for the split condition."""
        return self._run(b0, controls, 0, self.num_joints)


class FastMLP:
    """Unstructured next-observation rollout.

    The network predicts the whole observation in one pass, so there is no
    per-channel term to fuse and no way to avoid computing the joint block. The
    preallocated buffer and the narrowed return still apply.
    """

    def __init__(self, model: MLPModel, num_joints: int) -> None:
        self.model = model
        self.obs_dim = model.obs_dim
        self.act_dim = model.act_dim
        self.num_joints = num_joints

    @torch.no_grad()
    def rollout_tracked(self, b0: torch.Tensor,
                        controls: torch.Tensor) -> torch.Tensor:
        k = controls.shape[0]
        b = b0.expand(k, -1) if b0.dim() == 1 else b0
        net, nj = self.model.net, self.num_joints
        out = torch.empty(k, controls.shape[1], self.obs_dim - nj)
        for t in range(controls.shape[1]):
            u = controls[:, t]
            b = b + net(torch.cat([b, tau(b, nj), u], dim=-1))
            out[:, t] = b[:, nj:]
        return out

    def rollout(self, b0, controls):
        return self.model.rollout(b0, controls)


class FastSplit:
    """Linear lifted joints plus the analytic forward kinematics.

    The kinematics need the whole configuration, so the rollout cannot be
    narrowed here; it is narrowed to the joint block instead, and the tool
    position it returns is already the only thing the cost reads.
    """

    def __init__(self, model: LiftedModel, plant) -> None:
        self.lifted = FastLifted(model, plant.num_joints)
        self.model = model
        self.plant = plant

    @torch.no_grad()
    def rollout_tracked(self, b0: torch.Tensor,
                        controls: torch.Tensor) -> torch.Tensor:
        q = self.lifted.rollout_joints(b0, controls)
        return self.plant.task.forward_kinematics_torch(q)


def build_fast_rollout(kind: str, plant, model):
    """Dispatch by condition name, mirroring ``exper.backends.build_backend``."""
    if kind in ("linear", "bilinear"):
        return FastLifted(model, plant.num_joints)
    if kind == "mlp":
        return FastMLP(model, plant.num_joints)
    if kind == "split":
        return FastSplit(model, plant)
    raise SystemExit(f"no fast rollout for condition '{kind}'; the oracle rolls "
                     f"through MuJoCo and shares none of these items")
