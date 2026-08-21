"""FR3 plant wrapper.

A thin layer over :mod:`envs.franka` that exposes exactly what the experiments
need: the observation the paper uses, one true step, a batched true rollout for
the oracle, and the excitation used to identify the rollout model.

The observation is ``b = [q, p_tcp]``: the joint angles and the tool-center-point
position, so the leading ``num_joints`` entries are configuration angles and the
trailing three are the tracked output.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from envs.franka import NUM_JOINTS, FrankaTask, FrankaTaskConfig, FrankaDatasetConfig

from .config import Config

Array = np.ndarray


@dataclass(frozen=True)
class Rollouts:
    """One batch of excitation snippets."""

    obs: Array       # (n, H+1, d)
    controls: Array  # (n, H, m)


class FrankaPlant:
    """The true system, its observation, and its excitation."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.task = FrankaTask(
            config=FrankaTaskConfig(
                control_dt=cfg.plant.control_dt,
                horizon=cfg.planner.horizon,
                closed_loop_steps=cfg.task.steps,
                action_limit=cfg.plant.action_limit,
                reach_threshold=cfg.task.reach,
                strict_threshold=cfg.task.strict,
                num_rollout_threads=cfg.plant.rollout_threads,
            ),
            dataset_config=FrankaDatasetConfig(
                num_snippets=cfg.data.num_snippets,
                snippet_horizon=cfg.data.snippet_horizon,
                joint_margin=cfg.data.joint_margin,
                coherent_frac=cfg.data.coherent_frac,
                jitter_frac=cfg.data.jitter_frac,
            ),
        )
        self.num_joints = NUM_JOINTS
        self.act_dim = NUM_JOINTS
        self.obs_dim = NUM_JOINTS + 3
        self.limit = cfg.plant.action_limit
        self.kinematic = cfg.plant.kinematic
        self.dt = cfg.plant.control_dt
        self.q_low = self.task.joint_low.copy()
        self.q_high = self.task.joint_high.copy()

    # ------------------------------------------------------------ observation
    def observe(self, state: Array) -> Array:
        """b = [q, p_tcp] from a full plant state."""
        q = np.asarray(state, dtype=np.float64)[: self.num_joints]
        return np.concatenate([q, self.task.ee_of_q(q)])

    def tcp(self, obs: Array) -> Array:
        return np.asarray(obs)[..., self.num_joints :]

    # --------------------------------------------------------------- dynamics
    def reset(self) -> Array:
        """Home state, as a full plant state."""
        return np.concatenate([self.task.home_qpos, np.zeros(self.num_joints)])

    def step(self, state: Array, u: Array) -> Array:
        if self.kinematic:
            q = np.asarray(state, dtype=np.float64)[: self.num_joints]
            u = np.clip(np.asarray(u, dtype=np.float64), -self.limit, self.limit)
            q = np.clip(q + u * self.dt, self.q_low, self.q_high)
            return np.concatenate([q, np.zeros(self.num_joints)])
        return self.task.true_step(state, u)

    def rollout_true(self, state: Array, controls: Array) -> Array:
        """Oracle rollout from the full state: (K, T, m) -> (K, T, d).

        At the physics level, rolls through MuJoCo from the full state including
        joint velocity. At the kinematic level, integrates the analytic velocity
        model and recovers the tool center point through the forward kinematics.
        Either way the observations at steps 1..T are returned to match the model
        rollout.
        """
        if self.kinematic:
            return self._rollout_true_kinematic(state, controls)
        qs, ees = self.task.batch_rollout(state, controls)
        return np.concatenate([qs, ees], axis=-1)

    def _integrate_kinematic(self, q0: Array, controls: Array) -> Array:
        """Per-step clipped velocity integration for a batch: (K,7),(K,T,7)->(K,T,7)."""
        controls = np.clip(np.asarray(controls, dtype=np.float64),
                           -self.limit, self.limit)
        q = np.asarray(q0, dtype=np.float64).copy()
        out = np.empty((controls.shape[0], controls.shape[1], self.num_joints))
        for t in range(controls.shape[1]):
            q = np.clip(q + controls[:, t] * self.dt, self.q_low, self.q_high)
            out[:, t] = q
        return out

    def _rollout_true_kinematic(self, state: Array, controls: Array) -> Array:
        import torch
        q0 = np.tile(np.asarray(state, dtype=np.float64)[: self.num_joints],
                     (len(controls), 1))
        qs = self._integrate_kinematic(q0, controls)          # (K, T, 7)
        with torch.no_grad():
            ees = self.task.forward_kinematics_torch(
                torch.as_tensor(qs, dtype=torch.float32)).numpy()
        return np.concatenate([qs, ees], axis=-1)

    def clip(self, u: Array) -> Array:
        return np.clip(u, -self.limit, self.limit)

    # --------------------------------------------------------------- excitation
    def excite(self, seed: int, white: bool | None = None) -> Rollouts:
        """Random joint-velocity snippets rolled through the plant.

        With ``white`` the per-snippet velocity bias is dropped and every step is
        drawn on its own, which is the control condition for the excitation
        claim: an input that changes sign every step barely turns the
        configuration, so the state-dependent part of the input gain is hardly
        visited.
        """
        cfg = self.cfg.data
        white = cfg.white if white is None else white
        rng = np.random.default_rng(seed)
        n, horizon = cfg.num_snippets, cfg.snippet_horizon

        q_lo = self.task.joint_low + cfg.joint_margin
        q_hi = self.task.joint_high - cfg.joint_margin
        q0 = rng.uniform(q_lo, q_hi, (n, self.num_joints))

        if white:
            controls = rng.uniform(-self.limit, self.limit,
                                   (n, horizon, self.act_dim))
        else:
            bias = rng.uniform(-self.limit * cfg.coherent_frac,
                               self.limit * cfg.coherent_frac,
                               (n, self.num_joints))
            jitter = rng.uniform(-self.limit * cfg.jitter_frac,
                                 self.limit * cfg.jitter_frac,
                                 (n, horizon, self.num_joints))
            controls = np.clip(bias[:, None, :] + jitter, -self.limit, self.limit)

        if self.kinematic:
            import torch
            qs = self._integrate_kinematic(q0, controls)          # (n, H, 7)
            qs = np.concatenate([q0[:, None, :], qs], axis=1)     # prepend q0 -> (n, H+1, 7)
            with torch.no_grad():
                ees = self.task.forward_kinematics_torch(
                    torch.as_tensor(qs, dtype=torch.float32)).numpy()
            obs = np.concatenate([qs, ees], axis=-1)
        else:
            obs = self.task.rollout_snippets(q0, controls)  # (n, H+1, d), batched
        return Rollouts(obs=obs, controls=controls)

    # -------------------------------------------------------------------- task
    def targets(self, num: int, seed: int) -> Array:
        """Reach targets drawn once and shared by every condition.

        The curated targets come first; any extra are sampled inside the box
        they span and kept only if the workspace actually reaches them, so no
        condition is scored against a goal the arm cannot attain.
        """
        base = np.stack(self.task.targets)
        if num <= len(base):
            return base[:num]

        rng = np.random.default_rng(seed)
        lo, hi = base.min(axis=0), base.max(axis=0)
        extra: list[Array] = []
        attempts = 0
        while len(extra) < num - len(base) and attempts < 2000:
            attempts += 1
            cand = rng.uniform(lo, hi, 3)
            if self._reachable(cand):
                extra.append(cand)
        if len(extra) < num - len(base):
            raise SystemExit(
                f"only found {len(base) + len(extra)} reachable targets of {num}; "
                "widen the box or lower task.num_targets")
        return np.concatenate([base, np.stack(extra)], axis=0)

    def _reachable(self, target: Array, tol: float = 0.03) -> bool:
        """True if damped-least-squares IK places the TCP within ``tol``."""
        q = self.task.home_qpos[: self.num_joints].copy()
        lo = self.task.joint_low
        hi = self.task.joint_high
        for _ in range(120):
            p = self.task.ee_of_q(q)
            e = np.asarray(target) - p
            if np.linalg.norm(e) < tol:
                return True
            j = self._position_jacobian(q)
            dq = j.T @ np.linalg.solve(j @ j.T + 1e-4 * np.eye(3), e)
            q = np.clip(q + dq, lo, hi)
        return np.linalg.norm(self.task.ee_of_q(q) - target) < tol

    def position_jacobian(self, q: Array) -> Array:
        """Public view of the TCP position Jacobian, (3, num_joints)."""
        return self._position_jacobian(np.asarray(q, dtype=np.float64))

    def _position_jacobian(self, q: Array, eps: float = 1e-5) -> Array:
        """Finite-difference TCP position Jacobian, (3, num_joints)."""
        p0 = self.task.ee_of_q(q)
        j = np.zeros((3, self.num_joints))
        for i in range(self.num_joints):
            dq = q.copy()
            dq[i] += eps
            j[:, i] = (self.task.ee_of_q(dq) - p0) / eps
        return j
