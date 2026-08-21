"""Empirical error-tube fitting and propagation helpers.

Follows `koopman/mppi_koopman/verify/tube_deep.py`:

- one-step lifted residuals r_k = Psi(x_{k+1}) - F(Psi(x_k), u_k),
- proportional bound ||r|| <= c_x ||z|| + c_u ||u|| fitted either by linprog
  (covers every training residual; sensitive to rare transient outliers) or
  by quantile-coverage scaling (`fit_tube_constants`, the default in the
  experiment runners: lstsq fit rescaled to cover a 1-delta fraction of
  residuals, robust to e.g. rest-to-motion transients the velocity-free
  lifted state cannot capture),
- propagation with the precomputed norm bound
  mbar(u) = ||A||_2 + sum_i |u_i| ||B_i||_2 >= ||M(u)||_2:

  e_{k+1} = (mbar(u_k) + c_x) e_k + c_x ||zhat_k|| + c_u ||u_k||.
"""

from __future__ import annotations

from typing import Callable, Tuple

import numpy as np
import torch
from scipy.optimize import linprog

from .models import DeepKoopmanModel
from .types import Array, BilinearKoopmanParams, TubeConstants


def fit_tube_constants(
    zs: Array,
    us: Array,
    residuals: Array,
    *,
    quantile: float = 0.99,
    min_constant: float = 1e-8,
) -> TubeConstants:
    """Fit ||r|| <= c_x ||z|| + c_u ||u|| with empirical coverage scaling."""

    z_norm = np.linalg.norm(np.asarray(zs), axis=-1).reshape(-1)
    u_norm = np.linalg.norm(np.asarray(us), axis=-1).reshape(-1)
    r_norm = np.linalg.norm(np.asarray(residuals), axis=-1).reshape(-1)
    Phi = np.stack([z_norm, u_norm], axis=-1)

    coef, *_ = np.linalg.lstsq(Phi, r_norm, rcond=None)
    coef = np.maximum(coef, min_constant)
    pred = Phi @ coef
    ratio = r_norm / np.maximum(pred, min_constant)
    scale = float(np.quantile(ratio, quantile))
    coef = np.maximum(coef * max(scale, 1.0), min_constant)
    return TubeConstants(c_x=float(coef[0]), c_u=float(coef[1]))


def tube_penalty(tubes: Array, beta_e: float) -> float:
    """Return beta_e * sum_t e_t."""

    return float(beta_e * np.sum(np.asarray(tubes)))


def cost_sensitivity_torch(
    bs: torch.Tensor,
    cost_fn: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    """Per-step cost sensitivities L[k, t] = ||d cost_fn / d bs[k, t]||_2.

    One backward pass over the whole candidate batch: trajectory costs are
    additive over steps, so the gradient of the batch-summed cost w.r.t.
    bs[k, t] is exactly the stage-cost gradient at that (candidate, step).

    Used by the "cost-sens" tube mode: the penalty beta_e * sum_t L_t e_t is
    a first-order bound on the candidate cost error
    |J_true - J_hat| <= sum_t L_t ||C|| e_t (decode is a projection here, so
    ||C|| = 1). Model error is charged only where it can actually move the
    cost, i.e. where it can distort the softmax weights of the MBD update.
    """

    with torch.enable_grad():
        bs_g = bs.detach().requires_grad_(True)
        total = cost_fn(bs_g).sum()
        (grad,) = torch.autograd.grad(total, bs_g)
    return grad.norm(dim=-1)


@torch.no_grad()
def compute_one_step_residuals(
    model: DeepKoopmanModel,
    states: Array,
    controls: Array,
    *,
    device: torch.device | str = "cpu",
    batch_size: int = 2048,
) -> Tuple[Array, Array, Array]:
    """One-step lifted residuals over trajectory windows.

    Args:
        states: Koopman states with shape (num_traj, T + 1, state_dim).
        controls: Controls with shape (num_traj, T, action_dim).

    Returns:
        (zs, us, residuals) flattened over all (traj, step) pairs, where
        residuals[i] = Psi(x_{k+1}) - F(Psi(x_k), u_k).
    """

    device = torch.device(device)
    model = model.to(device)
    zs_out, us_out, res_out = [], [], []
    x_all = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    u_all = torch.as_tensor(np.asarray(controls), dtype=torch.float32, device=device)
    for i in range(0, x_all.shape[0], batch_size):
        xb = x_all[i : i + batch_size]
        ub = u_all[i : i + batch_size]
        z = model.lift(xb)
        z_from, z_to = z[:, :-1], z[:, 1:]
        z_pred = model.step(z_from, ub)
        lift_dim = z.shape[-1]
        zs_out.append(z_from.reshape(-1, lift_dim).cpu().numpy())
        us_out.append(ub.reshape(-1, ub.shape[-1]).cpu().numpy())
        res_out.append((z_to - z_pred).reshape(-1, lift_dim).cpu().numpy())
    return (
        np.concatenate(zs_out, axis=0),
        np.concatenate(us_out, axis=0),
        np.concatenate(res_out, axis=0),
    )


def fit_tube_constants_linprog(
    zs: Array,
    us: Array,
    residuals: Array,
    *,
    min_constant: float = 1e-8,
) -> TubeConstants:
    """Fit the smallest c_x, c_u covering every residual (reference method).

    Solves: min c_x + c_u  s.t.  c_x ||z_i|| + c_u ||u_i|| >= ||r_i||.
    """

    z_norm = np.linalg.norm(np.asarray(zs), axis=-1).reshape(-1)
    u_norm = np.linalg.norm(np.asarray(us), axis=-1).reshape(-1)
    r_norm = np.linalg.norm(np.asarray(residuals), axis=-1).reshape(-1)
    sol = linprog(
        c=[1.0, 1.0],
        A_ub=-np.stack([z_norm, u_norm], axis=-1),
        b_ub=-r_norm,
        bounds=[(0.0, None), (0.0, None)],
        method="highs",
    )
    if not sol.success:
        raise RuntimeError(f"tube linprog failed: {sol.message}")
    return TubeConstants(
        c_x=float(max(sol.x[0], min_constant)),
        c_u=float(max(sol.x[1], min_constant)),
    )


def bilinear_norm_bounds(params: BilinearKoopmanParams) -> Tuple[float, Array]:
    """Spectral norms (||A||_2, [||B_i||_2]) for the mbar(u) bound."""

    norm_a = float(np.linalg.norm(params.A, ord=2))
    norm_bs = np.array(
        [np.linalg.norm(params.Bs[i], ord=2) for i in range(params.Bs.shape[0])],
        dtype=np.float64,
    )
    return norm_a, norm_bs


def propagate_tube_batch_torch(
    lifted_states: torch.Tensor,
    controls: torch.Tensor,
    *,
    norm_a: float,
    norm_bs: torch.Tensor,
    constants: TubeConstants,
) -> torch.Tensor:
    """Propagate the scalar tube for batched candidate rollouts.

    Args:
        lifted_states: Predicted lifted states, shape (K, T + 1, lift_dim).
        controls: Controls, shape (K, T, action_dim).
        norm_a / norm_bs: Precomputed spectral norms for mbar(u).

    Returns:
        Tubes with shape (K, T + 1); tubes[:, 0] = 0.
    """

    mbar = norm_a + (controls.abs() * norm_bs).sum(dim=-1)
    z_norm = lifted_states.norm(dim=-1)
    u_norm = controls.norm(dim=-1)
    horizon = controls.shape[1]
    tubes = [torch.zeros(controls.shape[0], dtype=controls.dtype, device=controls.device)]
    e = tubes[0]
    for k in range(horizon):
        e = (mbar[:, k] + constants.c_x) * e + constants.c_x * z_norm[:, k] + constants.c_u * u_norm[:, k]
        tubes.append(e)
    return torch.stack(tubes, dim=1)

