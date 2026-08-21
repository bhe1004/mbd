"""The shared world both planners see.

A :class:`Scene` owns everything that is *common* to BK-MBD and SQP-MPC: the
FR3 task and its dynamics, the whole-arm collision body, the keep-out
obstacle field, the start pose, the goal, and the MuJoCo visualization model.
It also exposes the two candidate-cost terms a sampling planner adds on top of
the task cost -- the whole-arm obstacle penalty and the joint-limit penalty --
as batched methods, so BK-MBD calls them and any future sampling planner can
too. The geometry and margins here are the single source both planners must
share; only the *way each consumes them* (soft sampling penalty vs. linearized
keep-out) differs.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import mujoco  # noqa: E402

from experiments.arm_collision import ArmFK, ObstacleField  # noqa: E402
from envs.franka import (  # noqa: E402
    NUM_JOINTS,
    SCENE_XML_PATH,
    FrankaTask,
    FrankaTaskConfig,
)

from secy.config import Config  # noqa: E402


def home_at_tcp(task: FrankaTask, tcp_xyz, iters: int = 300,
                fk=None, obstacle=None, avoid_gain: float = 0.4,
                clear_target: float = 0.03) -> np.ndarray:
    """Joints reaching TCP position ``tcp_xyz`` (orientation free), damped-least-
    squares IK from the home configuration.

    When ``fk`` and ``obstacle`` are given, a NULLSPACE collision-avoidance term
    pushes the whole arm out of the obstacles while leaving the TCP fixed: at
    each step the reach update ``J^+ e`` is augmented with
    ``gain * (I - J^+J) grad(clearance)``, ascending the arm's clearance in the
    redundant directions that do not move the tool. Crucially this uses the
    SAME sphere-cloud (``fk``) vs box/floor (``obstacle``) clearance the planner
    and the referee use, so the start pose is collision-free in the experiment's
    own model -- not a mesh model that might disagree. One-time, so the FD
    Jacobians cost nothing."""

    q = np.asarray(task.home_qpos, dtype=np.float64).copy()
    target = np.asarray(tcp_xyz, dtype=np.float64)
    lo = np.asarray(task.joint_low, dtype=np.float64) + 0.02
    hi = np.asarray(task.joint_high, dtype=np.float64) - 0.02
    avoid = fk is not None and obstacle is not None
    eps, ceps = 1e-4, 1e-3
    for _ in range(iters):
        p = np.asarray(task.ee_of_q(q), dtype=np.float64)
        err = target - p
        J = np.zeros((3, NUM_JOINTS))
        for i in range(NUM_JOINTS):
            dq = q.copy()
            dq[i] += eps
            J[:, i] = (np.asarray(task.ee_of_q(dq)) - p) / eps
        Jp = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), np.eye(3))
        step = Jp @ err
        clear = None
        if avoid:
            clear = obstacle.clearance(fk.spheres_np(q), fk.radii_np)
            if clear < clear_target:
                g = np.zeros(NUM_JOINTS)
                for i in range(NUM_JOINTS):
                    dq = q.copy()
                    dq[i] += ceps
                    g[i] = (obstacle.clearance(fk.spheres_np(dq), fk.radii_np)
                            - clear) / ceps
                N = np.eye(NUM_JOINTS) - Jp @ J
                step = step + avoid_gain * (N @ g)
        n = float(np.linalg.norm(step))
        if n > 0.2:                                   # cap the step for stability
            step *= 0.2 / n
        q = np.clip(q + step, lo, hi)
        if np.linalg.norm(err) < 1e-4 and (not avoid or clear is None
                                           or clear >= clear_target):
            break
    return q


class Scene:
    """Task + whole-arm body + obstacles + start/goal + sim, assembled once."""

    def __init__(self, cfg: Config, device: torch.device | str = "cpu") -> None:
        self.cfg = cfg
        self.device = torch.device(device)

        # ---- the task (dynamics), with the horizon / speed overrides --------
        task_kw = {}
        if cfg.env.max_joint_velocity is not None:
            task_kw["action_limit"] = float(cfg.env.max_joint_velocity)
        if cfg.env.horizon is not None:
            task_kw["horizon"] = int(cfg.env.horizon)
        self.task = FrankaTask(FrankaTaskConfig(**task_kw) if task_kw else None)
        self.period = self.task.config.control_dt
        self.horizon = self.task.config.horizon
        self.strict = self.task.config.strict_threshold

        # goal for the first segment (a null target defers to task.targets);
        # resolved before the obstacles so an "auto" sphere can use it.
        self.goal = self.goal_for(cfg.runtime.target_ids[0])

        # ---- the keep-out obstacles + the whole-arm body (BEFORE the start
        # pose, so the start IK can avoid them in this exact model) ----------
        self.fk = None
        self.obstacle = None
        if cfg.env.obstacles:
            entries = [self._resolve_obstacle(e) for e in cfg.env.obstacles]
            self.fk = ArmFK(
                self.task, device=self.device,
                first_link=cfg.arm.first_link,
                link_samples=cfg.arm.link_samples,
                hand_fingers=cfg.arm.gripper_fingers,
            )
            self.obstacle = ObstacleField(
                entries, margin=cfg.collision.margin,
                weight=cfg.collision.weight, device=self.device,
                hard=cfg.collision.hard, floor_z=cfg.collision.floor_z,
            )

        # ---- start pose: obstacle-aware IK to start_tcp (whole arm clear) ---
        if cfg.env.start_tcp is not None:
            self.start_q = home_at_tcp(self.task, cfg.env.start_tcp,
                                       fk=self.fk, obstacle=self.obstacle)
        else:
            self.start_q = np.asarray(self.task.home_qpos, dtype=np.float64).copy()

        # ---- joint-limit envelope (margin-inset) for the penalty -----------
        self._jl_lo = torch.as_tensor(self.task.joint_low + cfg.joint_limit.margin,
                                      dtype=torch.float32, device=self.device)
        self._jl_hi = torch.as_tensor(self.task.joint_high - cfg.joint_limit.margin,
                                      dtype=torch.float32, device=self.device)

        # ---- the visualization / execution sim -----------------------------
        self.scene_model = mujoco.MjModel.from_xml_path(str(SCENE_XML_PATH))
        if self.scene_model.nq != NUM_JOINTS or self.scene_model.nu != NUM_JOINTS:
            raise SystemExit("scene model does not match the 7-DoF task model")
        self.nsub = int(round(self.period / self.scene_model.opt.timestep))
        self.sub_dt = self.period / self.nsub
        self.sim = mujoco.MjData(self.scene_model)
        mujoco.mj_resetDataKeyframe(
            self.scene_model, self.sim, self.scene_model.key("home").id)
        self.sim.qpos[:NUM_JOINTS] = self.start_q
        self.sim.qvel[:] = 0.0
        mujoco.mj_forward(self.scene_model, self.sim)
        self.tcp = self.scene_model.site("tcp").id

    # ------------------------------------------------------------------ goals
    def goal_for(self, target_id: int) -> np.ndarray:
        """Reach point for a segment: the ``target`` override if set, else the
        task's built-in target by index."""

        if self.cfg.env.target is not None:
            return np.asarray(self.cfg.env.target, dtype=np.float64).copy()
        if not 0 <= target_id < len(self.task.targets):
            raise SystemExit(
                f"target id {target_id} out of range 0..{len(self.task.targets) - 1}")
        return np.asarray(self.task.targets[target_id], dtype=np.float64).copy()

    # ------------------------------------------------------------- obstacles
    def _resolve_obstacle(self, entry):
        """Turn a config obstacle into ObstacleField form, resolving an "auto"
        sphere center to 40% of the way from the home tool to the first goal."""

        kind, center, size = entry
        if isinstance(center, str) and center == "auto":
            home_ee = self.task.ee_of_q(self.task.home_qpos)
            center = home_ee + 0.4 * (np.asarray(self.goal) - home_ee)
        return (kind, np.asarray(center, dtype=np.float64), size)

    def start_clearance(self) -> float:
        """Smallest surface gap between the start-pose arm and any obstacle
        (negative = already penetrating)."""

        if self.obstacle is None:
            return float("inf")
        return self.obstacle.clearance(self.fk.spheres_np(self.start_q),
                                       self.fk.radii_np)

    # ------------------------------------------- shared candidate-cost terms
    def joint_limit_penalty(self, q_path: torch.Tensor) -> torch.Tensor:
        """q_path: (N, T, 7) decoded joints -> (N,) squared encroachment past
        the margin-inset joint limits, summed over joints and steps."""

        w = self.cfg.joint_limit.weight
        if not w:
            return torch.zeros(q_path.shape[0], dtype=q_path.dtype,
                               device=q_path.device)
        over = (q_path - self._jl_hi).clamp(min=0.0)
        under = (self._jl_lo - q_path).clamp(min=0.0)
        return w * (over.pow(2) + under.pow(2)).sum(dim=(-2, -1))

    def arm_penalty(self, q_path: torch.Tensor, q_now: np.ndarray) -> torch.Tensor:
        """q_path: (N, T, 7) decoded joints -> (N,) summed whole-arm obstacle
        penalty, charged per step and at interpolated sub-steps so a fast
        candidate cannot cross an obstacle unseen between two control instants.
        No convexification: the sampler's native way of eating geometry."""

        pts = self.fk.spheres(q_path)                             # (N, T, P, 3)
        pen = self.obstacle.penalty(pts, self.fk.radii).sum(dim=1)  # (N,)
        q0 = torch.as_tensor(np.asarray(q_now, dtype=np.float32)[:NUM_JOINTS],
                             device=pts.device)
        p_prev = torch.cat(
            [self.fk.spheres(q0[None])[None].expand(pts.shape[0], 1, -1, -1),
             pts[:, :-1]], dim=1)
        substeps = self.cfg.collision.substeps
        for i in range(substeps):
            f = (i + 1.0) / (substeps + 1.0)
            pen = pen + self.obstacle.penalty(
                p_prev + f * (pts - p_prev), self.fk.radii).sum(dim=1)
        return pen
