"""Franka FR3 end-effector reaching task in MuJoCo for the BK-MBD experiments.

Realistic replacement of the synthetic 7-DOF chain (`envs/arm.py`):

- model: mujoco_menagerie `franka_fr3` with joint-velocity servo actuators
  (`envs/assets/franka_fr3/fr3_velocity.xml`), gravity off (equivalent to the
  gravity-compensated joint-velocity interface of the real robot),
- state s = [q, qd] (14): the velocity servo introduces a first-order lag
  between the commanded and realized joint velocity, so the true dynamics
  are no longer a pure integrator,
- Koopman state b = [q, ee] (10) with ee measured from the MuJoCo site
  (sensor-level information, identical for every learned method),
- the true-dynamics oracle rolls candidates through MuJoCo itself
  (multithreaded `mujoco.rollout`), which is also the wall-clock reference
  for the rollout-cost claim,
- the structure-informed split baseline uses a torch forward kinematics
  extracted *from the MuJoCo model tree* (guaranteed consistent with the
  simulator), applied to the decoded joints in the cost.

Task cost, thresholds, and protocol mirror `envs/arm.py`:
sum_k ||ee_k - target||^2 + 0.002 ||u_k||^2 + 10 ||ee_T - target||^2,
reach success < 0.05 m, early stop < 0.025 m.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import torch
from mujoco import rollout as mj_rollout

from bk_mbd.types import Array
from envs.base import MBDClosedLoopMixin

NUM_JOINTS = 7
# Task model: FR3 with the Franka hand rigidly attached (fingers welded
# half-open, no extra DoF); the control point is the "tcp" site at the
# midpoint between the two fingertips.
_XML_PATH = (
    Path(__file__).resolve().parent / "assets" / "franka_fr3" / "fr3_hand_velocity.xml"
)
SCENE_XML_PATH = (
    Path(__file__).resolve().parent / "assets" / "franka_fr3" / "scene_velocity.xml"
)


@dataclass(frozen=True)
class FrankaCostWeights:
    """Cost weights matched to the arm task for comparability."""

    ee: float = 1.0
    control: float = 0.002
    terminal_ee: float = 10.0


@dataclass(frozen=True)
class FrankaDatasetConfig:
    """Offline rollout data settings for Koopman training."""

    num_snippets: int = 6000
    snippet_horizon: int = 15
    joint_margin: float = 0.15
    coherent_frac: float = 0.6
    jitter_frac: float = 0.5
    chunk_size: int = 1000


@dataclass(frozen=True)
class FrankaTaskConfig:
    """FR3 task settings."""

    control_dt: float = 0.05
    horizon: int = 15
    closed_loop_steps: int = 120
    action_limit: float = 1.5
    reach_threshold: float = 0.05
    strict_threshold: float = 0.025
    num_rollout_threads: int = 16


def default_targets() -> List[Array]:
    """Reach targets around the FR3 home end-effector (0.55, 0.00, 0.62)."""

    return [
        np.array(t, dtype=np.float64)
        for t in [
            [0.45, 0.35, 0.35],
            [0.45, -0.35, 0.35],
            [0.65, 0.00, 0.30],
            [0.35, 0.40, 0.60],
            [0.35, -0.40, 0.60],
            [0.30, 0.00, 0.80],
            [0.60, 0.20, 0.55],
        ]
    ]


class FrankaTask(MBDClosedLoopMixin):
    """FR3 reaching environment with MuJoCo velocity-servo true dynamics."""

    task_name = "franka"
    state_dim = 2 * NUM_JOINTS  # s = [q, qd]
    action_dim = NUM_JOINTS
    base_dim = NUM_JOINTS + 3  # b = [q, ee]

    def __init__(
        self,
        config: FrankaTaskConfig | None = None,
        cost_weights: FrankaCostWeights | None = None,
        dataset_config: FrankaDatasetConfig | None = None,
    ) -> None:
        self.config = config or FrankaTaskConfig()
        self.cost_weights = cost_weights or FrankaCostWeights()
        self.dataset_config = dataset_config or FrankaDatasetConfig()

        self.model = mujoco.MjModel.from_xml_path(str(_XML_PATH))
        self._data = mujoco.MjData(self.model)
        self._rollout_datas = [
            mujoco.MjData(self.model)
            for _ in range(self.config.num_rollout_threads)
        ]
        self._nsub = int(round(self.config.control_dt / self.model.opt.timestep))
        self._nstate = mujoco.mj_stateSize(
            self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS
        )
        self._ee_site = self.model.site("tcp").id

        low = -self.config.action_limit * np.ones(self.action_dim, dtype=np.float64)
        high = self.config.action_limit * np.ones(self.action_dim, dtype=np.float64)
        self.action_bounds: Tuple[Array, Array] = (low, high)
        self.joint_low = self.model.jnt_range[:, 0].copy()
        self.joint_high = self.model.jnt_range[:, 1].copy()

        key = self.model.key("home").id
        self.home_qpos = self.model.key_qpos[key].copy()
        self.targets = default_targets()

        # Fixed transforms for the torch FK (extracted from the model tree).
        self._fk_params = self._extract_fk_params()

    # ------------------------------------------------------------------ helpers
    def _set_state(self, s: Array) -> None:
        s = np.asarray(s, dtype=np.float64)
        self._data.qpos[:] = s[:NUM_JOINTS]
        self._data.qvel[:] = s[NUM_JOINTS:]

    def _full_state(self, s: Array) -> Array:
        """FULLPHYSICS state vector [time, qpos, qvel] for mujoco.rollout."""

        s = np.asarray(s, dtype=np.float64)
        full = np.zeros(self._nstate, dtype=np.float64)
        full[1 : 1 + NUM_JOINTS] = s[:NUM_JOINTS]
        full[1 + NUM_JOINTS : 1 + 2 * NUM_JOINTS] = s[NUM_JOINTS:]
        return full

    def ee_of_q(self, q: Array) -> Array:
        """Measured end-effector position for joint angles q (kinematics)."""

        self._data.qpos[:] = np.asarray(q, dtype=np.float64)
        self._data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self._data)
        return self._data.site_xpos[self._ee_site].copy()

    # ------------------------------------------------------------------ dynamics
    def true_step(self, s: Array, u: Array) -> Array:
        """Apply a joint-velocity command for one control period (mj_step)."""

        self._set_state(s)
        self._data.ctrl[:] = np.clip(
            np.asarray(u, dtype=np.float64),
            self.action_bounds[0],
            self.action_bounds[1],
        )
        for _ in range(self._nsub):
            mujoco.mj_step(self.model, self._data)
        return np.concatenate([self._data.qpos, self._data.qvel]).astype(np.float64)

    def true_rollout(self, s0: Array, U: Array) -> Array:
        states = [np.asarray(s0, dtype=np.float64).copy()]
        s = states[0]
        for u in np.asarray(U, dtype=np.float64):
            s = self.true_step(s, u)
            states.append(s.copy())
        return np.stack(states, axis=0)

    def _batch_rollout(self, s0: Array, controls: Array):
        """Roll K control sequences through MuJoCo (threaded).

        Args:
            s0: Current state (14,).
            controls: Control-period commands with shape (K, T, 7).

        Returns:
            qs:  Joint trajectories at control boundaries, (K, T, 7).
            ees: End-effector positions at control boundaries, (K, T, 3).
        """

        controls = np.asarray(controls, dtype=np.float64)
        num, horizon = controls.shape[0], controls.shape[1]
        ctrl_sub = np.repeat(controls, self._nsub, axis=1)  # (K, T*nsub, 7)
        init = np.tile(self._full_state(s0), (num, 1))
        state, sensordata = mj_rollout.rollout(
            self.model, self._rollout_datas, init, ctrl_sub
        )
        boundary = slice(self._nsub - 1, None, self._nsub)
        qs = np.asarray(state)[:, boundary, 1 : 1 + NUM_JOINTS]
        ees = np.asarray(sensordata)[:, boundary, :3]
        assert qs.shape[1] == horizon
        return qs, ees

    # ---------------------------------------------------------------- base state
    def state_to_base_torch(
        self, s: Array, device: torch.device | str | None
    ) -> torch.Tensor:
        q = np.asarray(s, dtype=np.float64)[:NUM_JOINTS]
        b = np.concatenate([q, self.ee_of_q(q)])
        return torch.as_tensor(b, dtype=torch.float32, device=device)

    # --------------------------------------------------------------------- cost
    def trajectory_cost_base_torch(
        self,
        base_states: torch.Tensor,
        controls: torch.Tensor,
        goal: Array | torch.Tensor | None = None,
        *,
        u_prev: Array | torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Reaching cost on batched base trajectories (same form as the arm)."""

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
    def final_error(self, s: Array, goal: Array) -> float:
        q = np.asarray(s, dtype=np.float64)[:NUM_JOINTS]
        return float(np.linalg.norm(self.ee_of_q(q) - np.asarray(goal, dtype=np.float64)))

    def success(self, s: Array, goal: Array) -> bool:
        return self.final_error(s, goal) < self.config.reach_threshold

    def strict_success(self, s: Array, goal: Array) -> bool:
        return self.final_error(s, goal) < self.config.strict_threshold

    # -------------------------------------------------------------------- cases
    def case(self, case_id: int) -> Tuple[Array, Array, str]:
        """Return (start_state, target, case_name); the arm starts at home."""

        if case_id < 0 or case_id >= len(self.targets):
            raise ValueError(f"case_id must be in [0, {len(self.targets) - 1}]")
        start = np.concatenate(
            [self.home_qpos, np.zeros(NUM_JOINTS)]
        ).astype(np.float64)
        return start, self.targets[case_id].copy(), f"target_{case_id}"

    # ------------------------------------------------------------------ dataset
    def sample_dataset(self, seed: int) -> Dict[str, Array]:
        """Coherent random joint-velocity snippets rolled through MuJoCo.

        Returns states (q trajectories), base_states ([q, ee]), and controls,
        with the same array layout as the other tasks.
        """

        cfg = self.dataset_config
        rng = np.random.default_rng(seed)
        n, horizon = cfg.num_snippets, cfg.snippet_horizon
        limit = self.config.action_limit

        q_lo = self.joint_low + cfg.joint_margin
        q_hi = self.joint_high - cfg.joint_margin
        q0 = rng.uniform(q_lo, q_hi, (n, NUM_JOINTS))
        qd_coherent = rng.uniform(
            -limit * cfg.coherent_frac, limit * cfg.coherent_frac, (n, NUM_JOINTS)
        )
        jitter = rng.uniform(
            -limit * cfg.jitter_frac,
            limit * cfg.jitter_frac,
            (n, horizon, NUM_JOINTS),
        )
        controls = np.clip(qd_coherent[:, None, :] + jitter, -limit, limit)

        states = np.zeros((n, horizon + 1, NUM_JOINTS), dtype=np.float64)
        ees = np.zeros((n, horizon + 1, 3), dtype=np.float64)
        states[:, 0] = q0
        for i in range(n):
            ees[i, 0] = self.ee_of_q(q0[i])

        for start in range(0, n, cfg.chunk_size):
            end = min(start + cfg.chunk_size, n)
            s0_chunk = np.concatenate(
                [q0[start:end], np.zeros((end - start, NUM_JOINTS))], axis=-1
            )
            ctrl_sub = np.repeat(controls[start:end], self._nsub, axis=1)
            init = np.stack([self._full_state(s) for s in s0_chunk], axis=0)
            state, sensordata = mj_rollout.rollout(
                self.model, self._rollout_datas, init, ctrl_sub
            )
            boundary = slice(self._nsub - 1, None, self._nsub)
            states[start:end, 1:] = np.asarray(state)[:, boundary, 1 : 1 + NUM_JOINTS]
            ees[start:end, 1:] = np.asarray(sensordata)[:, boundary, :3]

        base_states = np.concatenate([states, ees], axis=-1)
        return {"states": states, "base_states": base_states, "controls": controls}

    # ------------------------------------------------------------------- oracle
    def make_true_evaluate(
        self,
        s: Array,
        goal: Array,
        u_prev: Array,
        device: torch.device | str | None,
    ):
        """Oracle backend: threaded MuJoCo rollouts + measured EE cost."""

        def evaluate(candidates: Array) -> Array:
            qs, ees = self._batch_rollout(s, candidates)
            b0 = np.concatenate(
                [
                    np.asarray(s, dtype=np.float64)[:NUM_JOINTS],
                    self.ee_of_q(np.asarray(s)[:NUM_JOINTS]),
                ]
            )
            bs = np.concatenate([qs, ees], axis=-1)  # (K, T, 10)
            bs = np.concatenate(
                [np.tile(b0, (bs.shape[0], 1, 1)), bs], axis=1
            )  # (K, T+1, 10)
            with torch.no_grad():
                costs = self.trajectory_cost_base_torch(
                    torch.as_tensor(bs, dtype=torch.float32),
                    torch.as_tensor(np.asarray(candidates), dtype=torch.float32),
                    goal,
                    u_prev=u_prev,
                )
            return costs.numpy()

        return evaluate

    # ---------------------------------------------------------- torch FK (split)
    def _extract_fk_params(self):
        """Fixed joint-to-joint transforms extracted from the MuJoCo tree.

        Walks the kinematic chain from the base to the attachment site and
        records, for each joint, the constant transform from the previous
        joint frame, plus the joint axis. Guarantees the torch FK matches the
        simulator by construction.
        """

        m = self.model
        site_body = m.site_bodyid[self._ee_site]
        chain = []
        body = site_body
        while body != 0:
            chain.append(body)
            body = m.body_parentid[body]
        chain = chain[::-1]  # base -> ee body

        segs = []  # list of (fixed 4x4 up to the joint, joint axis)
        fixed = np.eye(4)
        for body in chain:
            T_body = np.eye(4)
            T_body[:3, 3] = m.body_pos[body]
            quat = m.body_quat[body]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, quat)
            T_body[:3, :3] = R.reshape(3, 3)
            fixed = fixed @ T_body
            if m.body_jntnum[body] > 0:
                jnt = m.body_jntadr[body]
                T_j = np.eye(4)
                T_j[:3, 3] = m.jnt_pos[jnt]
                fixed = fixed @ T_j
                segs.append((fixed.copy(), m.jnt_axis[jnt].copy()))
                fixed = np.eye(4)
                # translate back from joint anchor to body frame origin
                T_back = np.eye(4)
                T_back[:3, 3] = -m.jnt_pos[jnt]
                fixed = fixed @ T_back
        T_site = np.eye(4)
        T_site[:3, 3] = m.site_pos[self._ee_site]
        Rs = np.zeros(9)
        mujoco.mju_quat2Mat(Rs, self.model.site_quat[self._ee_site])
        T_site[:3, :3] = Rs.reshape(3, 3)
        tail = fixed @ T_site
        assert len(segs) == NUM_JOINTS
        return segs, tail

    def forward_kinematics_torch(self, q: torch.Tensor) -> torch.Tensor:
        """Batched EE position for joint angles q (..., 7), model-consistent."""

        segs, tail = self._fk_params
        batch_shape = q.shape[:-1]
        qf = q.reshape(-1, NUM_JOINTS)
        dtype, device = qf.dtype, qf.device
        T = torch.eye(4, dtype=dtype, device=device).expand(qf.shape[0], 4, 4)
        for i, (fixed, axis) in enumerate(segs):
            fixed_t = torch.as_tensor(fixed, dtype=dtype, device=device)
            T = T @ fixed_t
            T = T @ _axis_angle_mat(axis, qf[:, i], dtype, device)
        tail_t = torch.as_tensor(tail, dtype=dtype, device=device)
        T = T @ tail_t
        ee = T[:, :3, 3]
        return ee.reshape(*batch_shape, 3)

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
        """MPPI-DK-conditions baseline: learned q dynamics + analytic FK cost."""

        model = model.to(device) if device is not None else model
        model.eval()

        def make_evaluate(s: Array, goal: Array, u_prev: Array):
            def evaluate(candidates: Array) -> Array:
                with torch.no_grad():
                    U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                    q0 = torch.as_tensor(
                        np.asarray(s, dtype=np.float64)[:NUM_JOINTS],
                        dtype=torch.float32,
                        device=device,
                    )
                    z0 = model.lift(q0).expand(U_t.shape[0], -1)
                    q_hat = model.decode(model.rollout(z0, U_t))
                    bs = torch.cat(
                        [q_hat, self.forward_kinematics_torch(q_hat)], dim=-1
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


def _axis_angle_mat(
    axis: Array, angle: torch.Tensor, dtype, device
) -> torch.Tensor:
    """Homogeneous rotation about a fixed axis for a batch of angles."""

    a = torch.as_tensor(axis, dtype=dtype, device=device)
    a = a / torch.linalg.norm(a)
    K = torch.tensor(
        [
            [0.0, -a[2], a[1]],
            [a[2], 0.0, -a[0]],
            [-a[1], a[0], 0.0],
        ],
        dtype=dtype,
        device=device,
    )
    c = torch.cos(angle)[:, None, None]
    s = torch.sin(angle)[:, None, None]
    eye = torch.eye(3, dtype=dtype, device=device)
    R = eye + s * K + (1.0 - c) * (K @ K)
    T = torch.eye(4, dtype=dtype, device=device).expand(angle.shape[0], 4, 4).clone()
    T[:, :3, :3] = R
    return T
