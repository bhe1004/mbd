"""7-DOF arm end-effector reaching task for the BK-MBD experiments.

Ports the task structure from
`koopman/mppi_koopman/verify/arm_multiseed_v2.py`:

- state is the joint vector q (7), action is joint velocity qd (7),
- true dynamics are kinematic: q_next = clip(q + qd * dt, -q_lim, q_lim),
- the Koopman state is b = [q, ee(q)] (10-dim) with an analytic forward
  kinematics of the same capsule chain used by the reference MuJoCo model,
- task cost is sum_k ||ee_k - target||^2 + 0.002 ||u_k||^2 with terminal
  10 ||ee_T - target||^2,
- reach success is ||ee - target|| < 0.05 m, early stop at < 0.025 m.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import torch

from bk_mbd.types import Array
from envs.base import MBDClosedLoopMixin

# Kinematic chain of the reference MuJoCo XML: alternating z/y joint axes,
# capsule offsets along local z, end-effector site 0.12 above the last joint.
_OFFSETS = (
    (0.0, 0.0, 0.10),
    (0.0, 0.0, 0.18),
    (0.0, 0.0, 0.18),
    (0.0, 0.0, 0.16),
    (0.0, 0.0, 0.16),
    (0.0, 0.0, 0.14),
    (0.0, 0.0, 0.12),
)
_AXES = "zyzyzyz"
_EE_OFFSET = (0.0, 0.0, 0.12)
NUM_JOINTS = 7


def _rot(axis: str, q: torch.Tensor) -> torch.Tensor:
    """Homogeneous rotation about local z or y for a batch of angles."""

    c, s = torch.cos(q), torch.sin(q)
    o, l = torch.zeros_like(q), torch.ones_like(q)
    if axis == "z":
        rows = [[c, -s, o, o], [s, c, o, o], [o, o, l, o], [o, o, o, l]]
    else:
        rows = [[c, o, s, o], [o, l, o, o], [-s, o, c, o], [o, o, o, l]]
    return torch.stack([torch.stack(r, -1) for r in rows], -2)


def forward_kinematics_torch(q: torch.Tensor) -> torch.Tensor:
    """End-effector position for joint angles q with shape (..., 7)."""

    batch_shape = q.shape[:-1]
    qf = q.reshape(-1, NUM_JOINTS)
    dtype, device = qf.dtype, qf.device
    T = torch.eye(4, dtype=dtype, device=device).expand(qf.shape[0], 4, 4)
    for i in range(NUM_JOINTS):
        trans = torch.eye(4, dtype=dtype, device=device).clone()
        trans[:3, 3] = torch.tensor(_OFFSETS[i], dtype=dtype, device=device)
        T = T @ (trans.expand(qf.shape[0], 4, 4) @ _rot(_AXES[i], qf[:, i]))
    ee_point = torch.tensor([*_EE_OFFSET, 1.0], dtype=dtype, device=device)
    ee = (T @ ee_point)[:, :3]
    return ee.reshape(*batch_shape, 3)


def forward_kinematics_np(q: Array) -> Array:
    """NumPy wrapper around the torch forward kinematics."""

    return forward_kinematics_torch(torch.as_tensor(q, dtype=torch.float64)).numpy()


def link_positions_np(q: Array) -> Array:
    """Positions of the base, each joint origin, and the EE for one q (7,).

    Returns an array with shape (9, 3): [base, joint_1..joint_7, ee], useful
    for drawing the kinematic chain at a given configuration.
    """

    qt = torch.as_tensor(q, dtype=torch.float64).reshape(NUM_JOINTS)
    T = torch.eye(4, dtype=torch.float64)
    points = [T[:3, 3].clone()]
    for i in range(NUM_JOINTS):
        trans = torch.eye(4, dtype=torch.float64)
        trans[:3, 3] = torch.tensor(_OFFSETS[i], dtype=torch.float64)
        T = T @ trans @ _rot(_AXES[i], qt[i : i + 1])[0]
        points.append(T[:3, 3].clone())
    ee_point = torch.tensor([*_EE_OFFSET, 1.0], dtype=torch.float64)
    points.append((T @ ee_point)[:3])
    return torch.stack(points, dim=0).numpy()


@dataclass(frozen=True)
class ArmCostWeights:
    """Cost weights matched to the v2 MPPI arm experiment."""

    ee: float = 1.0
    control: float = 0.002
    terminal_ee: float = 10.0


@dataclass(frozen=True)
class ArmDatasetConfig:
    """Offline rollout data settings for Koopman training."""

    num_snippets: int = 6000
    snippet_horizon: int = 15
    q_low: float = -2.0
    q_high: float = 2.0
    coherent_frac: float = 0.6
    jitter_frac: float = 0.5


@dataclass(frozen=True)
class ArmTaskConfig:
    """Arm task settings."""

    dt: float = 0.05
    horizon: int = 15
    closed_loop_steps: int = 120
    action_limit: float = 2.0
    joint_limit: float = 2.6
    reach_threshold: float = 0.05
    strict_threshold: float = 0.025


def default_targets() -> List[Array]:
    """Reach targets from the original multi-seed arm experiment."""

    return [
        np.array(t, dtype=np.float64)
        for t in [
            [0.4, 0.3, 0.55],
            [-0.3, 0.4, 0.6],
            [0.3, -0.35, 0.7],
            [-0.4, -0.2, 0.5],
            [0.5, 0.0, 0.65],
            [0.0, 0.5, 0.5],
            [-0.45, 0.25, 0.75],
        ]
    ]


class ArmTask(MBDClosedLoopMixin):
    """7-DOF arm reaching environment with kinematic true dynamics."""

    task_name = "arm"
    state_dim = NUM_JOINTS
    action_dim = NUM_JOINTS
    base_dim = NUM_JOINTS + 3

    def __init__(
        self,
        config: ArmTaskConfig | None = None,
        cost_weights: ArmCostWeights | None = None,
        dataset_config: ArmDatasetConfig | None = None,
    ) -> None:
        self.config = config or ArmTaskConfig()
        self.cost_weights = cost_weights or ArmCostWeights()
        self.dataset_config = dataset_config or ArmDatasetConfig()
        low = -self.config.action_limit * np.ones(self.action_dim, dtype=np.float64)
        high = self.config.action_limit * np.ones(self.action_dim, dtype=np.float64)
        self.action_bounds: Tuple[Array, Array] = (low, high)
        self.targets = default_targets()

    # ------------------------------------------------------------------ dynamics
    def true_step(self, q: Array, u: Array) -> Array:
        """Kinematic velocity integration with joint limits."""

        cfg = self.config
        u_clipped = np.clip(
            np.asarray(u, dtype=np.float64), -cfg.action_limit, cfg.action_limit
        )
        return np.clip(
            np.asarray(q, dtype=np.float64) + u_clipped * cfg.dt,
            -cfg.joint_limit,
            cfg.joint_limit,
        )

    def true_rollout(self, q0: Array, U: Array) -> Array:
        """Roll one NumPy control sequence, returning joint trajectories."""

        states = [np.asarray(q0, dtype=np.float64).copy()]
        q = states[0]
        for u in np.asarray(U, dtype=np.float64):
            q = self.true_step(q, u)
            states.append(q.copy())
        return np.stack(states, axis=0)

    # ---------------------------------------------------------------- base state
    def base_features_torch(self, q: torch.Tensor) -> torch.Tensor:
        """b = [q, ee(q)] for torch joint states of shape (..., 7)."""

        return torch.cat([q, forward_kinematics_torch(q)], dim=-1)

    def state_to_base_torch(
        self, q: Array, device: torch.device | str | None
    ) -> torch.Tensor:
        return self.base_features_torch(
            torch.as_tensor(q, dtype=torch.float32, device=device)
        )

    # --------------------------------------------------------------------- cost
    def trajectory_cost_base_torch(
        self,
        base_states: torch.Tensor,
        controls: torch.Tensor,
        goal: Array | torch.Tensor | None = None,
        *,
        u_prev: Array | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Original arm MPPI cost on batched base trajectories.

        `base_states` has shape (K, T + 1, 10) with rows [q, ee]; `u_prev` is
        accepted for interface compatibility but unused (the reference arm
        cost has no delta-u term).
        """

        if base_states.ndim != 3 or controls.ndim != 3:
            raise ValueError("base_states must be (K,T+1,10), controls (K,T,m)")
        if (
            base_states.shape[0] != controls.shape[0]
            or base_states.shape[1] != controls.shape[1] + 1
        ):
            raise ValueError("base_states and controls have incompatible shapes")

        target = torch.as_tensor(
            goal, dtype=base_states.dtype, device=base_states.device
        )
        ee = base_states[:, 1:, NUM_JOINTS : NUM_JOINTS + 3]
        w = self.cost_weights
        running = w.ee * torch.sum((ee - target) ** 2, dim=-1) + w.control * torch.sum(
            controls**2, dim=-1
        )
        terminal = w.terminal_ee * torch.sum((ee[:, -1] - target) ** 2, dim=-1)
        return torch.sum(running, dim=-1) + terminal

    # ------------------------------------------------------------------ metrics
    def final_error(self, q: Array, goal: Array) -> float:
        """Euclidean end-effector distance to the target."""

        ee = forward_kinematics_np(np.asarray(q, dtype=np.float64))
        return float(np.linalg.norm(ee - np.asarray(goal, dtype=np.float64)))

    def success(self, q: Array, goal: Array) -> bool:
        return self.final_error(q, goal) < self.config.reach_threshold

    def strict_success(self, q: Array, goal: Array) -> bool:
        return self.final_error(q, goal) < self.config.strict_threshold

    # -------------------------------------------------------------------- cases
    def case(self, case_id: int) -> Tuple[Array, Array, str]:
        """Return (start_q, target, case_name); the arm starts at q = 0."""

        if case_id < 0 or case_id >= len(self.targets):
            raise ValueError(f"case_id must be in [0, {len(self.targets) - 1}]")
        start = np.zeros(NUM_JOINTS, dtype=np.float64)
        return start, self.targets[case_id].copy(), f"target_{case_id}"

    # ------------------------------------------------------------------ dataset
    def sample_dataset(self, seed: int) -> Dict[str, Array]:
        """Coherent random joint-velocity snippets for Koopman training."""

        cfg = self.dataset_config
        task_cfg = self.config
        rng = np.random.default_rng(seed)
        n, horizon = cfg.num_snippets, cfg.snippet_horizon
        qv = task_cfg.action_limit

        states = np.zeros((n, horizon + 1, NUM_JOINTS), dtype=np.float64)
        controls = np.zeros((n, horizon, NUM_JOINTS), dtype=np.float64)
        states[:, 0] = rng.uniform(cfg.q_low, cfg.q_high, (n, NUM_JOINTS))
        qd_coherent = rng.uniform(
            -qv * cfg.coherent_frac, qv * cfg.coherent_frac, (n, NUM_JOINTS)
        )
        for k in range(horizon):
            jitter = rng.uniform(
                -qv * cfg.jitter_frac, qv * cfg.jitter_frac, (n, NUM_JOINTS)
            )
            qd = np.clip(qd_coherent + jitter, -qv, qv)
            controls[:, k] = qd
            states[:, k + 1] = np.clip(
                states[:, k] + qd * task_cfg.dt,
                -task_cfg.joint_limit,
                task_cfg.joint_limit,
            )

        ee = forward_kinematics_np(states)
        base_states = np.concatenate([states, ee], axis=-1)
        return {"states": states, "base_states": base_states, "controls": controls}

    # ------------------------------------------------------------ split baseline
    def closed_loop_koopman_split_mbd(
        self,
        optimizer,
        model,
        *,
        case_id: int = 0,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> Dict:
        """MPPI-DK-conditions baseline: learned q dynamics + analytic FK cost.

        The linear Koopman model predicts only the joint trajectory (where the
        input enters state-independently); the configuration-dependent
        coupling (forward kinematics) is applied analytically to the decoded
        joints when evaluating the cost, mirroring the structure used by the
        MPPI-DK experiments.
        """

        model = model.to(device) if device is not None else model
        model.eval()

        def make_evaluate(q: Array, goal: Array, u_prev: Array):
            def evaluate(candidates: Array) -> Array:
                with torch.no_grad():
                    U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                    q0 = torch.as_tensor(q, dtype=torch.float32, device=device)
                    z0 = model.lift(q0).expand(U_t.shape[0], -1)
                    q_hat = model.decode(model.rollout(z0, U_t))
                    bs = torch.cat(
                        [q_hat, forward_kinematics_torch(q_hat)], dim=-1
                    )
                    costs_t = self.trajectory_cost_base_torch(
                        bs, U_t, goal, u_prev=u_prev
                    )
                    return costs_t.cpu().numpy()

            return evaluate

        return self._closed_loop_mbd(
            optimizer,
            make_evaluate,
            method="dk_mbd_split",
            case_id=case_id,
            seed=seed,
        )

    # ------------------------------------------------------------------- oracle
    def make_true_evaluate(
        self,
        q: Array,
        goal: Array,
        u_prev: Array,
        device: torch.device | str | None,
    ):
        """Oracle backend: kinematic rollout + analytic FK per step."""

        cfg = self.config

        def evaluate(candidates: Array) -> Array:
            with torch.no_grad():
                U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                q_t = torch.as_tensor(q, dtype=torch.float32, device=device)
                q_t = q_t.unsqueeze(0).expand(U_t.shape[0], -1)
                bs = [self.base_features_torch(q_t)]
                for k in range(U_t.shape[1]):
                    q_t = torch.clamp(
                        q_t + U_t[:, k] * cfg.dt, -cfg.joint_limit, cfg.joint_limit
                    )
                    bs.append(self.base_features_torch(q_t))
                bs_t = torch.stack(bs, dim=1)
                costs_t = self.trajectory_cost_base_torch(
                    bs_t, U_t, goal, u_prev=u_prev
                )
                return costs_t.cpu().numpy()

        return evaluate
