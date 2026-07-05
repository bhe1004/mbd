"""Rollout backends for true, linear Koopman, and bilinear Koopman models.

The NumPy functions below are useful for small smoke tests and single-sequence
debugging. The planner path should use the Torch batched functions so all MBD
candidate sequences share one vectorized rollout over the sample dimension.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
import torch

from .models import check_bilinear_params, check_linear_params, decode_state
from .types import (
    Array,
    BilinearKoopmanParams,
    LinearKoopmanParams,
    RolloutResult,
    TubeConstants,
)


@dataclass
class TorchRolloutResult:
    """Torch rollout output for batched candidate control sequences."""

    states: torch.Tensor
    controls: torch.Tensor
    costs: torch.Tensor | None = None
    tubes: torch.Tensor | None = None
    info: Dict[str, Any] = field(default_factory=dict)


def _as_tensor(
    x: Array | torch.Tensor,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    if torch.is_tensor(x):
        return x.to(device=device if device is not None else x.device, dtype=dtype)
    return torch.as_tensor(x, dtype=dtype, device=device)


def _batched_controls(
    U: Array | torch.Tensor,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    U_t = _as_tensor(U, device=device, dtype=dtype)
    if U_t.ndim == 2:
        U_t = U_t.unsqueeze(0)
    if U_t.ndim != 3:
        raise ValueError("U must have shape (T, action_dim) or (K, T, action_dim)")
    return U_t


def _batched_initial_lift(
    z0: Array | torch.Tensor,
    batch_size: int,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> torch.Tensor:
    z = _as_tensor(z0, device=device, dtype=dtype)
    if z.ndim == 1:
        z = z.unsqueeze(0).expand(batch_size, -1).clone()
    if z.ndim != 2:
        raise ValueError("z0 must have shape (lift_dim,) or (K, lift_dim)")
    if z.shape[0] != batch_size:
        raise ValueError("batched z0 must have the same K as U")
    return z


def _linear_params_torch(
    params: LinearKoopmanParams,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    A = _as_tensor(params.A, device=device, dtype=dtype)
    B = _as_tensor(params.B, device=device, dtype=dtype)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (lift_dim, lift_dim)")
    if B.ndim != 2 or B.shape[0] != A.shape[0]:
        raise ValueError("B must have shape (lift_dim, action_dim)")
    return A, B


def _bilinear_params_torch(
    params: BilinearKoopmanParams,
    *,
    device: torch.device | str | None,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    A = _as_tensor(params.A, device=device, dtype=dtype)
    B0 = _as_tensor(params.B0, device=device, dtype=dtype)
    Bs = _as_tensor(params.Bs, device=device, dtype=dtype)
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square with shape (lift_dim, lift_dim)")
    if B0.ndim != 2 or B0.shape[0] != A.shape[0]:
        raise ValueError("B0 must have shape (lift_dim, action_dim)")
    if Bs.ndim != 3 or Bs.shape[1:] != A.shape:
        raise ValueError("Bs must have shape (action_dim, lift_dim, lift_dim)")
    if Bs.shape[0] != B0.shape[1]:
        raise ValueError("number of B_i matrices must match action_dim")
    return A, B0, Bs


def linear_koopman_step(z: Array, u: Array, params: LinearKoopmanParams) -> Array:
    """Compute z_next = A z + B u."""

    return params.A @ z + params.B @ u


def bilinear_matrix(u: Array, params: BilinearKoopmanParams) -> Array:
    """Compute M(u) = A + sum_i u_i B_i."""

    return params.A + np.einsum("i,ijk->jk", u, params.Bs)


def bilinear_koopman_step(
    z: Array, u: Array, params: BilinearKoopmanParams
) -> Array:
    """Compute z_next = M(u) z + B0 u."""

    M_u = bilinear_matrix(u, params)
    return M_u @ z + params.B0 @ u


def linear_koopman_step_batch_torch(
    z: torch.Tensor,
    u: torch.Tensor,
    A: torch.Tensor,
    B: torch.Tensor,
) -> torch.Tensor:
    """Compute z_next = A z + B u for a batch of samples."""

    return z @ A.T + u @ B.T


def bilinear_matrix_batch_torch(
    u: torch.Tensor,
    A: torch.Tensor,
    Bs: torch.Tensor,
) -> torch.Tensor:
    """Compute batched M(u) = A + sum_i u_i B_i."""

    return A.unsqueeze(0) + torch.einsum("ki,inj->knj", u, Bs)


def bilinear_koopman_step_batch_torch(
    z: torch.Tensor,
    u: torch.Tensor,
    A: torch.Tensor,
    B0: torch.Tensor,
    Bs: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute a batched bilinear Koopman step and return z_next, M(u)."""

    M_u = bilinear_matrix_batch_torch(u, A, Bs)
    z_next = torch.bmm(M_u, z.unsqueeze(-1)).squeeze(-1) + u @ B0.T
    return z_next, M_u


def linear_koopman_rollout_batch_torch(
    z0: Array | torch.Tensor,
    U: Array | torch.Tensor,
    params: LinearKoopmanParams,
    state_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchRolloutResult:
    """Roll K control sequences through a linear Koopman model.

    Args:
        z0: Initial lifted state, shape (lift_dim,) or (K, lift_dim).
        U: Controls, shape (T, action_dim) or (K, T, action_dim).
        params: Linear Koopman parameters.
        state_dim: Number of physical state coordinates stored first in z.
    """

    U_t = _batched_controls(U, device=device, dtype=dtype)
    z = _batched_initial_lift(
        z0, U_t.shape[0], device=U_t.device, dtype=U_t.dtype
    )
    A, B = _linear_params_torch(params, device=U_t.device, dtype=U_t.dtype)

    zs = [z]
    for t in range(U_t.shape[1]):
        z = linear_koopman_step_batch_torch(z, U_t[:, t], A, B)
        zs.append(z)
    zs_t = torch.stack(zs, dim=1)
    xs_t = zs_t[..., :state_dim]
    return TorchRolloutResult(states=xs_t, controls=U_t, info={"z": zs_t})


def bilinear_koopman_rollout_batch_torch(
    z0: Array | torch.Tensor,
    U: Array | torch.Tensor,
    params: BilinearKoopmanParams,
    state_dim: int,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchRolloutResult:
    """Roll K control sequences through a bilinear Koopman model."""

    U_t = _batched_controls(U, device=device, dtype=dtype)
    z = _batched_initial_lift(
        z0, U_t.shape[0], device=U_t.device, dtype=U_t.dtype
    )
    A, B0, Bs = _bilinear_params_torch(params, device=U_t.device, dtype=U_t.dtype)

    zs = [z]
    for t in range(U_t.shape[1]):
        z, _ = bilinear_koopman_step_batch_torch(z, U_t[:, t], A, B0, Bs)
        zs.append(z)
    zs_t = torch.stack(zs, dim=1)
    xs_t = zs_t[..., :state_dim]
    return TorchRolloutResult(states=xs_t, controls=U_t, info={"z": zs_t})


def bilinear_koopman_rollout_with_tube_batch_torch(
    z0: Array | torch.Tensor,
    U: Array | torch.Tensor,
    params: BilinearKoopmanParams,
    state_dim: int,
    tube_constants: TubeConstants,
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> TorchRolloutResult:
    """Roll K bilinear Koopman sequences while propagating scalar tubes."""

    U_t = _batched_controls(U, device=device, dtype=dtype)
    z = _batched_initial_lift(
        z0, U_t.shape[0], device=U_t.device, dtype=U_t.dtype
    )
    A, B0, Bs = _bilinear_params_torch(params, device=U_t.device, dtype=U_t.dtype)

    e = torch.zeros(U_t.shape[0], dtype=U_t.dtype, device=U_t.device)
    zs = [z]
    tubes = [e]
    c_x = torch.as_tensor(tube_constants.c_x, dtype=U_t.dtype, device=U_t.device)
    c_u = torch.as_tensor(tube_constants.c_u, dtype=U_t.dtype, device=U_t.device)

    for t in range(U_t.shape[1]):
        u = U_t[:, t]
        z_next, M_u = bilinear_koopman_step_batch_torch(z, u, A, B0, Bs)
        M_norm = torch.linalg.matrix_norm(M_u, ord=2)
        e = (
            (M_norm + c_x) * e
            + c_x * torch.linalg.vector_norm(z, dim=-1)
            + c_u * torch.linalg.vector_norm(u, dim=-1)
        )
        z = z_next
        zs.append(z)
        tubes.append(e)

    zs_t = torch.stack(zs, dim=1)
    tubes_t = torch.stack(tubes, dim=1)
    xs_t = zs_t[..., :state_dim]
    return TorchRolloutResult(
        states=xs_t,
        controls=U_t,
        tubes=tubes_t,
        info={"z": zs_t},
    )


def linear_koopman_rollout(
    z0: Array,
    U: Array,
    params: LinearKoopmanParams,
    state_dim: int,
) -> RolloutResult:
    """Roll one control sequence through a linear Koopman model."""

    check_linear_params(params)
    z = np.asarray(z0, dtype=np.float64)
    zs = [z.copy()]
    for u in np.asarray(U, dtype=np.float64):
        z = linear_koopman_step(z, u, params)
        zs.append(z.copy())
    zs_arr = np.stack(zs, axis=0)
    xs = decode_state(zs_arr, state_dim)
    return RolloutResult(states=xs, controls=np.asarray(U), info={"z": zs_arr})


def bilinear_koopman_rollout(
    z0: Array,
    U: Array,
    params: BilinearKoopmanParams,
    state_dim: int,
) -> RolloutResult:
    """Roll one control sequence through a bilinear Koopman model."""

    check_bilinear_params(params)
    z = np.asarray(z0, dtype=np.float64)
    zs = [z.copy()]
    for u in np.asarray(U, dtype=np.float64):
        z = bilinear_koopman_step(z, u, params)
        zs.append(z.copy())
    zs_arr = np.stack(zs, axis=0)
    xs = decode_state(zs_arr, state_dim)
    return RolloutResult(states=xs, controls=np.asarray(U), info={"z": zs_arr})


def bilinear_koopman_rollout_with_tube(
    z0: Array,
    U: Array,
    params: BilinearKoopmanParams,
    state_dim: int,
    tube_constants: TubeConstants,
) -> RolloutResult:
    """Roll a bilinear Koopman model while propagating the scalar error tube."""

    check_bilinear_params(params)
    z = np.asarray(z0, dtype=np.float64)
    zs = [z.copy()]
    tubes = [0.0]
    e = 0.0
    for u in np.asarray(U, dtype=np.float64):
        M_u = bilinear_matrix(u, params)
        e = (
            (np.linalg.norm(M_u, ord=2) + tube_constants.c_x) * e
            + tube_constants.c_x * np.linalg.norm(z)
            + tube_constants.c_u * np.linalg.norm(u)
        )
        z = M_u @ z + params.B0 @ u
        zs.append(z.copy())
        tubes.append(float(e))
    zs_arr = np.stack(zs, axis=0)
    xs = decode_state(zs_arr, state_dim)
    return RolloutResult(
        states=xs,
        controls=np.asarray(U),
        tubes=np.asarray(tubes),
        info={"z": zs_arr},
    )
