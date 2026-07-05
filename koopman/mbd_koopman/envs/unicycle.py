"""Unicycle parking task used by the BK-MBD experiments.

This ports the task structure from
`koopman/mppi_koopman/verify/unicycle_multiseed_v2.py`:

- state is (x, y, theta),
- action is (forward_velocity, yaw_rate),
- dynamics use RK4 with dt=0.05,
- task cost is computed on base features
  [x, y, sin(theta), cos(theta) - 1],
- reported parking success follows the original distance threshold.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np
import torch

from bk_mbd.costs import wrap_angle
from bk_mbd.types import Array
from envs.base import MBDClosedLoopMixin


@dataclass(frozen=True)
class UnicycleCostWeights:
    """Cost weights matched to the v2 MPPI unicycle experiment."""

    pos: float = 1.0
    heading: float = 0.3
    control: float = 0.05
    delta_control: float = 0.4
    terminal_pos: float = 30.0
    terminal_heading: float = 8.0


@dataclass(frozen=True)
class UnicycleDatasetConfig:
    """Offline rollout data settings for Koopman training."""

    num_snippets: int = 8000
    snippet_horizon: int = 15
    state_low_xy: float = -2.0
    state_high_xy: float = 2.0
    action_low: float = -1.5
    action_high: float = 1.5


@dataclass(frozen=True)
class UnicycleTaskConfig:
    """Unicycle task settings."""

    dt: float = 0.05
    horizon: int = 40
    closed_loop_steps: int = 180
    action_limit: float = 1.5
    park_radius: float = 0.3
    strict_pos_radius: float = 0.08
    strict_angle_radius: float = 0.2


def default_start_goal_pairs() -> List[Array]:
    """Start states used by the original multi-seed unicycle experiment."""

    return [
        np.array([1.5, 1.5, -np.pi / 2.0], dtype=np.float64),
        np.array([-1.5, 1.0, 0.0], dtype=np.float64),
        np.array([1.0, -1.5, np.pi / 2.0], dtype=np.float64),
        np.array([-1.2, -1.2, np.pi], dtype=np.float64),
    ]


def default_goal() -> Array:
    """Default parking goal."""

    return np.array([0.0, 0.0, 0.0], dtype=np.float64)


def base_features_np(states: Array) -> Array:
    """Return [x, y, sin(theta), cos(theta) - 1] for NumPy states."""

    states_arr = np.asarray(states)
    return np.stack(
        [
            states_arr[..., 0],
            states_arr[..., 1],
            np.sin(states_arr[..., 2]),
            np.cos(states_arr[..., 2]) - 1.0,
        ],
        axis=-1,
    )


def base_features_torch(states: torch.Tensor) -> torch.Tensor:
    """Return [x, y, sin(theta), cos(theta) - 1] for Torch states."""

    return torch.stack(
        [
            states[..., 0],
            states[..., 1],
            torch.sin(states[..., 2]),
            torch.cos(states[..., 2]) - 1.0,
        ],
        dim=-1,
    )


def _derivative_np(states: Array, controls: Array) -> Array:
    return np.stack(
        [
            controls[..., 0] * np.cos(states[..., 2]),
            controls[..., 0] * np.sin(states[..., 2]),
            controls[..., 1],
        ],
        axis=-1,
    )


def _derivative_torch(states: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
    return torch.stack(
        [
            controls[..., 0] * torch.cos(states[..., 2]),
            controls[..., 0] * torch.sin(states[..., 2]),
            controls[..., 1],
        ],
        dim=-1,
    )


class UnicycleTask(MBDClosedLoopMixin):
    """Unicycle parking environment with NumPy and Torch-batched rollouts."""

    task_name = "unicycle"
    state_dim = 3
    action_dim = 2
    base_dim = 4

    def __init__(
        self,
        config: UnicycleTaskConfig | None = None,
        cost_weights: UnicycleCostWeights | None = None,
        dataset_config: UnicycleDatasetConfig | None = None,
    ) -> None:
        self.config = config or UnicycleTaskConfig()
        self.cost_weights = cost_weights or UnicycleCostWeights()
        self.dataset_config = dataset_config or UnicycleDatasetConfig()
        low = -self.config.action_limit * np.ones(self.action_dim, dtype=np.float64)
        high = self.config.action_limit * np.ones(self.action_dim, dtype=np.float64)
        self.action_bounds: Tuple[Array, Array] = (low, high)
        self.starts = default_start_goal_pairs()
        self.goal = default_goal()

    def true_step(self, x: Array, u: Array) -> Array:
        """RK4 step for one or more NumPy states."""

        dt = self.config.dt
        x_arr = np.asarray(x, dtype=np.float64)
        u_arr = np.asarray(u, dtype=np.float64)
        k1 = _derivative_np(x_arr, u_arr)
        k2 = _derivative_np(x_arr + 0.5 * dt * k1, u_arr)
        k3 = _derivative_np(x_arr + 0.5 * dt * k2, u_arr)
        k4 = _derivative_np(x_arr + dt * k3, u_arr)
        return x_arr + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def true_step_torch(
        self, states: torch.Tensor, controls: torch.Tensor
    ) -> torch.Tensor:
        """RK4 step for Torch states with arbitrary batch dimensions."""

        dt = self.config.dt
        k1 = _derivative_torch(states, controls)
        k2 = _derivative_torch(states + 0.5 * dt * k1, controls)
        k3 = _derivative_torch(states + 0.5 * dt * k2, controls)
        k4 = _derivative_torch(states + dt * k3, controls)
        return states + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    def true_rollout(self, x0: Array, U: Array) -> Array:
        """Roll one NumPy control sequence."""

        states = [np.asarray(x0, dtype=np.float64).copy()]
        x = states[0]
        for u in np.asarray(U, dtype=np.float64):
            clipped = np.clip(u, self.action_bounds[0], self.action_bounds[1])
            x = self.true_step(x, clipped)
            states.append(x.copy())
        return np.stack(states, axis=0)

    def true_rollout_batch_torch(
        self,
        x0: Array | torch.Tensor,
        U: Array | torch.Tensor,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        """Roll K control sequences with Torch.

        Args:
            x0: Shape (state_dim,) or (K, state_dim).
            U: Shape (T, action_dim) or (K, T, action_dim).

        Returns:
            States with shape (K, T + 1, state_dim).
        """

        if torch.is_tensor(U):
            controls = U.to(device=device if device is not None else U.device, dtype=dtype)
        else:
            controls = torch.as_tensor(U, device=device, dtype=dtype)
        if controls.ndim == 2:
            controls = controls.unsqueeze(0)
        if controls.ndim != 3:
            raise ValueError("U must have shape (T, action_dim) or (K, T, action_dim)")
        low = torch.as_tensor(self.action_bounds[0], device=controls.device, dtype=dtype)
        high = torch.as_tensor(self.action_bounds[1], device=controls.device, dtype=dtype)
        controls = torch.clamp(controls, low, high)

        if torch.is_tensor(x0):
            state = x0.to(device=controls.device, dtype=dtype)
        else:
            state = torch.as_tensor(x0, device=controls.device, dtype=dtype)
        if state.ndim == 1:
            state = state.unsqueeze(0).expand(controls.shape[0], -1).clone()
        if state.ndim != 2 or state.shape[0] != controls.shape[0]:
            raise ValueError("x0 must have shape (state_dim,) or (K, state_dim)")

        states = [state]
        for t in range(controls.shape[1]):
            state = self.true_step_torch(state, controls[:, t])
            states.append(state)
        return torch.stack(states, dim=1)

    def _goal_relative_heading_feat(
        self, base_states: torch.Tensor, goal_t: torch.Tensor
    ) -> torch.Tensor:
        """[sin(theta - g), cos(theta - g) - 1] from base features via rotation."""

        sin_th = base_states[..., 2]
        cos_th = base_states[..., 3] + 1.0
        sin_g = torch.sin(goal_t[2])
        cos_g = torch.cos(goal_t[2])
        return torch.stack(
            [
                sin_th * cos_g - cos_th * sin_g,
                cos_th * cos_g + sin_th * sin_g - 1.0,
            ],
            dim=-1,
        )

    def trajectory_cost_base_torch(
        self,
        base_states: torch.Tensor,
        controls: torch.Tensor,
        goal: Array | torch.Tensor | None = None,
        *,
        u_prev: Array | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Original unicycle MPPI cost on batched base-feature trajectories.

        `base_states` has shape (K, T + 1, 4) with rows [x, y, sin, cos - 1];
        this is what both true rollouts (via `base_features_torch`) and decoded
        Koopman rollouts provide, so all methods share this cost exactly.
        """

        if base_states.ndim != 3 or controls.ndim != 3:
            raise ValueError("base_states must be (K,T+1,4), controls must be (K,T,m)")
        if (
            base_states.shape[0] != controls.shape[0]
            or base_states.shape[1] != controls.shape[1] + 1
        ):
            raise ValueError("base_states and controls have incompatible shapes")

        dtype = base_states.dtype
        device = base_states.device
        goal_t = (
            torch.as_tensor(self.goal, dtype=dtype, device=device)
            if goal is None
            else torch.as_tensor(goal, dtype=dtype, device=device)
        )
        prev = (
            torch.zeros(self.action_dim, dtype=dtype, device=device)
            if u_prev is None
            else torch.as_tensor(u_prev, dtype=dtype, device=device)
        )

        rolled = base_states[:, 1:]
        pos_err = rolled[..., :2] - goal_t[:2]
        heading_feat = self._goal_relative_heading_feat(rolled, goal_t)
        du = torch.empty_like(controls)
        du[:, 0] = controls[:, 0] - prev
        du[:, 1:] = controls[:, 1:] - controls[:, :-1]

        w = self.cost_weights
        running = (
            w.pos * torch.sum(pos_err**2, dim=-1)
            + w.heading * torch.sum(heading_feat**2, dim=-1)
            + w.control * torch.sum(controls**2, dim=-1)
            + w.delta_control * torch.sum(du**2, dim=-1)
        )

        final = base_states[:, -1]
        final_pos = final[:, :2] - goal_t[:2]
        final_heading = self._goal_relative_heading_feat(final, goal_t)
        terminal = (
            w.terminal_pos * torch.sum(final_pos**2, dim=-1)
            + w.terminal_heading * torch.sum(final_heading**2, dim=-1)
        )
        return torch.sum(running, dim=-1) + terminal

    def trajectory_cost_torch(
        self,
        states: torch.Tensor,
        controls: torch.Tensor,
        goal: Array | torch.Tensor | None = None,
        *,
        u_prev: Array | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the original unicycle MPPI cost for batched raw-state trajectories."""

        if states.ndim != 3 or controls.ndim != 3:
            raise ValueError("states must be (K,T+1,n), controls must be (K,T,m)")
        return self.trajectory_cost_base_torch(
            base_features_torch(states), controls, goal, u_prev=u_prev
        )

    def cost(
        self,
        xs: Array,
        U: Array,
        goal: Array | None = None,
        *,
        u_prev: Array | None = None,
    ) -> float:
        """NumPy wrapper around the batched Torch cost."""

        states = torch.as_tensor(xs[None], dtype=torch.float64)
        controls = torch.as_tensor(U[None], dtype=torch.float64)
        return float(
            self.trajectory_cost_torch(
                states,
                controls,
                self.goal if goal is None else goal,
                u_prev=np.zeros(self.action_dim) if u_prev is None else u_prev,
            )[0].item()
        )

    def final_error(self, x_final: Array, goal: Array | None = None) -> float:
        """Position-only final error used by the original reported park rate."""

        g = self.goal if goal is None else np.asarray(goal, dtype=np.float64)
        return float(np.linalg.norm(np.asarray(x_final, dtype=np.float64)[:2] - g[:2]))

    def angle_error(self, x_final: Array, goal: Array | None = None) -> float:
        """Absolute wrapped heading error."""

        g = self.goal if goal is None else np.asarray(goal, dtype=np.float64)
        return float(abs(wrap_angle(np.asarray(x_final)[2] - g[2])))

    def success(self, x_final: Array, goal: Array | None = None) -> bool:
        """Parking success metric matched to the original v2 final report."""

        return self.final_error(x_final, goal) < self.config.park_radius

    def strict_success(self, x_final: Array, goal: Array | None = None) -> bool:
        """Strict pose condition used for early stopping."""

        return (
            self.final_error(x_final, goal) < self.config.strict_pos_radius
            and self.angle_error(x_final, goal) < self.config.strict_angle_radius
        )

    def sample_dataset(self, seed: int) -> Dict[str, Array]:
        """Generate coherent random-control snippets for Koopman training."""

        cfg = self.dataset_config
        rng = np.random.default_rng(seed)
        states = np.zeros(
            (cfg.num_snippets, cfg.snippet_horizon + 1, self.state_dim),
            dtype=np.float64,
        )
        states[:, 0, 0] = rng.uniform(
            cfg.state_low_xy, cfg.state_high_xy, cfg.num_snippets
        )
        states[:, 0, 1] = rng.uniform(
            cfg.state_low_xy, cfg.state_high_xy, cfg.num_snippets
        )
        states[:, 0, 2] = rng.uniform(-np.pi, np.pi, cfg.num_snippets)
        controls = rng.uniform(
            cfg.action_low,
            cfg.action_high,
            (cfg.num_snippets, cfg.snippet_horizon, self.action_dim),
        )
        for t in range(cfg.snippet_horizon):
            states[:, t + 1] = self.true_step(states[:, t], controls[:, t])
        return {
            "states": states,
            "base_states": base_features_np(states),
            "controls": controls,
        }

    def case(self, case_id: int) -> Tuple[Array, Array, str]:
        """Return (start, goal, case_name)."""

        if case_id < 0 or case_id >= len(self.starts):
            raise ValueError(f"case_id must be in [0, {len(self.starts) - 1}]")
        return self.starts[case_id].copy(), self.goal.copy(), f"case_{case_id}"

    def extra_final_metrics(self, x: Array, goal: Array) -> Dict[str, float]:
        return {"angle_error": self.angle_error(x, goal)}

    def state_to_base_torch(
        self, x: Array, device: torch.device | str | None
    ) -> torch.Tensor:
        return base_features_torch(
            torch.as_tensor(x, dtype=torch.float32, device=device)
        )

    # ------------------------------------------------------------ split baseline
    def fit_body_frame_linear(self, dataset: Dict[str, Array]) -> Array:
        """Fit the MPPI-DK-conditions linear model [dp_body; dtheta] = G u.

        Transitions are mapped to the body frame (dp_body = R(-theta_k)
        (p_{k+1} - p_k)), where the kinematic unicycle's input enters
        state-independently, and a least-squares linear map G (3 x 2) is fitted.
        The heading-dependent rotation is composed analytically at rollout
        time, mirroring the surface-vehicle setup of the MPPI-DK paper.
        """

        states = np.asarray(dataset["states"], dtype=np.float64)
        controls = np.asarray(dataset["controls"], dtype=np.float64)
        theta = states[:, :-1, 2]
        dp = states[:, 1:, :2] - states[:, :-1, :2]
        cos_t, sin_t = np.cos(theta), np.sin(theta)
        dp_body = np.stack(
            [
                cos_t * dp[..., 0] + sin_t * dp[..., 1],
                -sin_t * dp[..., 0] + cos_t * dp[..., 1],
            ],
            axis=-1,
        )
        dtheta = wrap_angle(states[:, 1:, 2] - states[:, :-1, 2])
        targets = np.concatenate([dp_body, dtheta[..., None]], axis=-1).reshape(-1, 3)
        inputs = controls.reshape(-1, self.action_dim)
        G, *_ = np.linalg.lstsq(inputs, targets, rcond=None)
        return G.T  # (3, action_dim)

    def closed_loop_koopman_split_mbd(
        self,
        optimizer,
        G: Array,
        *,
        case_id: int = 0,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> Dict:
        """MPPI-DK-conditions baseline: linear body-frame model + analytic SE(2).

        Candidates are rolled by predicting the body-frame displacement with
        the linear map G and composing it analytically with the current
        heading, so the first-order coupling never has to be learned.
        """

        G_t = torch.as_tensor(np.asarray(G), dtype=torch.float32, device=device)

        def make_evaluate(x: Array, goal: Array, u_prev: Array):
            def evaluate(candidates: Array) -> Array:
                with torch.no_grad():
                    U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                    num = U_t.shape[0]
                    state = (
                        torch.as_tensor(x, dtype=torch.float32, device=device)
                        .unsqueeze(0)
                        .expand(num, -1)
                        .clone()
                    )
                    xs = [state]
                    deltas = U_t @ G_t.T  # (K, T, 3) body-frame displacements
                    for k in range(U_t.shape[1]):
                        cos_t = torch.cos(state[:, 2])
                        sin_t = torch.sin(state[:, 2])
                        dx = cos_t * deltas[:, k, 0] - sin_t * deltas[:, k, 1]
                        dy = sin_t * deltas[:, k, 0] + cos_t * deltas[:, k, 1]
                        state = torch.stack(
                            [
                                state[:, 0] + dx,
                                state[:, 1] + dy,
                                state[:, 2] + deltas[:, k, 2],
                            ],
                            dim=-1,
                        )
                        xs.append(state)
                    xs_t = torch.stack(xs, dim=1)
                    costs_t = self.trajectory_cost_torch(xs_t, U_t, goal, u_prev=u_prev)
                    return costs_t.cpu().numpy()

            return evaluate

        return self._closed_loop_mbd(
            optimizer,
            make_evaluate,
            method="dk_mbd_split",
            case_id=case_id,
            seed=seed,
        )

    def make_true_evaluate(
        self,
        x: Array,
        goal: Array,
        u_prev: Array,
        device: torch.device | str | None,
    ):
        def evaluate(candidates: Array) -> Array:
            with torch.no_grad():
                U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                xs_t = self.true_rollout_batch_torch(
                    x, U_t, device=device, dtype=torch.float32
                )
                costs_t = self.trajectory_cost_torch(xs_t, U_t, goal, u_prev=u_prev)
                return costs_t.cpu().numpy()

        return evaluate

