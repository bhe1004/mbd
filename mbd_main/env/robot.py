"""The FR3 robot description: kinematics and the whole-arm collision body.

This is the part of the environment that is the *same* on the simulator and on
the real arm -- link transforms, joint limits, forward kinematics, and the
sphere cloud used for collision checking. It is read out of the MuJoCo model
tree rather than hard-coded, so the torch FK is consistent with the simulator by
construction.

Two consumers, one geometry: the planner charges a soft penalty on the sphere
cloud, and the runner's referee tests the very same spheres margin-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List

import mujoco
import numpy as np
import torch

NUM_JOINTS = 7
ASSET_DIR = Path(__file__).resolve().parent / "assets" / "franka_fr3"
#: FR3 with the hand welded on; the control point is the "tcp" site between
#: the fingertips. Joint-velocity servo actuators, gravity off (the
#: gravity-compensated velocity interface of the real robot).
ROBOT_XML = ASSET_DIR / "fr3_hand_velocity.xml"
#: The same model plus floor/lighting/skybox, for the viewer.
SCENE_XML = ASSET_DIR / "scene_velocity.xml"


@dataclass(frozen=True)
class ArmBodyConfig:
    """How densely the arm is covered by collision spheres.

    Fewer spheres = faster planning (the penalty is evaluated per sphere).

    Attributes:
        first_link: which frame the cover starts at (0 = base ... 4 = elbow
            ... 8 = TCP). Higher drops the proximal links that can never move.
        link_samples: extra spheres interpolated per covered segment.
        arm_radius / tcp_radius / hand_radius: sphere sizes [m].
        gripper_fingers: True adds the two finger prongs (4 hand spheres
            instead of the 2 crossbar ones).
        hand_back / hand_side: gripper crossbar geometry [m].
    """

    first_link: int = 3
    link_samples: int = 3
    arm_radius: float = 0.05
    tcp_radius: float = 0.04
    gripper_fingers: bool = False
    hand_back: float = 0.05
    hand_side: float = 0.05
    hand_radius: float = 0.035


def axis_angle_matrix(axis, angle: torch.Tensor, dtype, device) -> torch.Tensor:
    """Homogeneous rotation about a fixed axis, batched over ``angle``."""

    a = torch.as_tensor(np.asarray(axis, dtype=np.float64), dtype=dtype, device=device)
    a = a / torch.linalg.norm(a)
    K = torch.zeros(3, 3, dtype=dtype, device=device)
    K[0, 1], K[0, 2] = -a[2], a[1]
    K[1, 0], K[1, 2] = a[2], -a[0]
    K[2, 0], K[2, 1] = -a[1], a[0]
    c = torch.cos(angle)[:, None, None]
    s = torch.sin(angle)[:, None, None]
    R = torch.eye(3, dtype=dtype, device=device) + s * K + (1.0 - c) * (K @ K)
    T = torch.eye(4, dtype=dtype, device=device).expand(angle.shape[0], 4, 4).clone()
    T[:, :3, :3] = R
    return T


class FrankaRobot:
    """Kinematics + collision body of the FR3, read from the MuJoCo model."""

    num_joints = NUM_JOINTS

    def __init__(self, body: ArmBodyConfig | None = None,
                 device: torch.device | str = "cpu",
                 xml_path: Path | str = ROBOT_XML) -> None:
        self.body_config = body or ArmBodyConfig()
        self.device = torch.device(device)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._data = mujoco.MjData(self.model)
        self.tcp_site = self.model.site("tcp").id

        self.joint_low = self.model.jnt_range[:, 0].copy()
        self.joint_high = self.model.jnt_range[:, 1].copy()
        self.home_qpos = self.model.key_qpos[self.model.key("home").id].copy()

        self._segments, self._tail = self._extract_fk_params()
        self._fixed = [torch.as_tensor(f, dtype=torch.float32, device=self.device)
                       for f, _ in self._segments]
        self._axes = [a for _, a in self._segments]
        self._tail_t = torch.as_tensor(self._tail, dtype=torch.float32, device=self.device)
        self.sphere_radii, self.sphere_radii_np = self._build_radii()
        self.num_spheres = int(self.sphere_radii_np.size)

    # ------------------------------------------------------------- kinematics
    def ee_of_q(self, q) -> np.ndarray:
        """Tool position for joint angles ``q`` (MuJoCo kinematics, float64)."""

        self._data.qpos[:] = np.asarray(q, dtype=np.float64)
        self._data.qvel[:] = 0.0
        mujoco.mj_kinematics(self.model, self._data)
        return self._data.site_xpos[self.tcp_site].copy()

    def _extract_fk_params(self):
        """Constant joint-to-joint transforms walked out of the model tree."""

        m = self.model
        chain: List[int] = []
        body = m.site_bodyid[self.tcp_site]
        while body != 0:
            chain.append(body)
            body = m.body_parentid[body]
        chain = chain[::-1]

        segments = []
        fixed = np.eye(4)
        for body in chain:
            T = np.eye(4)
            T[:3, 3] = m.body_pos[body]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, m.body_quat[body])
            T[:3, :3] = R.reshape(3, 3)
            fixed = fixed @ T
            if m.body_jntnum[body] > 0:
                jnt = m.body_jntadr[body]
                T_j = np.eye(4)
                T_j[:3, 3] = m.jnt_pos[jnt]
                fixed = fixed @ T_j
                segments.append((fixed.copy(), m.jnt_axis[jnt].copy()))
                fixed = np.eye(4)
                fixed[:3, 3] = -m.jnt_pos[jnt]     # back to the body frame origin
        T_site = np.eye(4)
        T_site[:3, 3] = m.site_pos[self.tcp_site]
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, m.site_quat[self.tcp_site])
        T_site[:3, :3] = R.reshape(3, 3)
        if len(segments) != NUM_JOINTS:
            raise RuntimeError(f"expected {NUM_JOINTS} joints in the chain, got {len(segments)}")
        return segments, fixed @ T_site

    def _chain(self, qf: torch.Tensor):
        """Frame origins along the chain plus the tool transform."""

        n = qf.shape[0]
        T = torch.eye(4, dtype=torch.float32, device=self.device).expand(n, 4, 4)
        origins = [torch.zeros(n, 3, dtype=torch.float32, device=self.device)]
        for i in range(NUM_JOINTS):
            T = T @ self._fixed[i]
            T = T @ axis_angle_matrix(self._axes[i], qf[:, i], torch.float32, self.device)
            origins.append(T[:, :3, 3])
        T = T @ self._tail_t
        origins.append(T[:, :3, 3])
        return T, origins

    def ee_torch(self, q: torch.Tensor) -> torch.Tensor:
        """Batched tool position, differentiable: q (..., 7) -> (..., 3)."""

        shape = q.shape[:-1]
        T, _ = self._chain(q.reshape(-1, NUM_JOINTS))
        return T[:, :3, 3].reshape(*shape, 3)

    # ------------------------------------------------------- collision spheres
    def _build_radii(self):
        cfg = self.body_config
        # origin order: base, link1..link7, TCP
        radii = ([cfg.arm_radius] * (1 + NUM_JOINTS) + [cfg.tcp_radius])[cfg.first_link:]
        all_r = list(radii)
        all_r += [cfg.arm_radius] * (len(radii) - 1) * cfg.link_samples
        n_hand = 0 if cfg.hand_side <= 0 else (4 if cfg.gripper_fingers else 2)
        all_r += [cfg.hand_radius] * n_hand
        t = torch.as_tensor(all_r, dtype=torch.float32, device=self.device)
        return t, t.cpu().numpy().astype(np.float64)

    def spheres(self, q: torch.Tensor) -> torch.Tensor:
        """Collision-sphere centres: q (..., 7) -> (..., num_spheres, 3)."""

        cfg = self.body_config
        shape = q.shape[:-1]
        T, origins = self._chain(q.reshape(-1, NUM_JOINTS))
        o = torch.stack(origins[cfg.first_link:], dim=1)          # (N, m, 3)
        parts = [o]
        for s in range(1, cfg.link_samples + 1):
            f = s / (cfg.link_samples + 1)
            parts.append(o[:, :-1] * (1 - f) + o[:, 1:] * f)
        if cfg.hand_side > 0:
            tcp = T[:, :3, 3]
            approach = T[:, :3, 2]        # points out between the fingers
            finger = T[:, :3, 1]          # the fingers separate along this axis
            bar = tcp - cfg.hand_back * approach
            hand = [bar + cfg.hand_side * finger, bar - cfg.hand_side * finger]
            if cfg.gripper_fingers:
                hand += [tcp + cfg.hand_side * finger, tcp - cfg.hand_side * finger]
            parts.append(torch.stack(hand, dim=1))
        p = torch.cat(parts, dim=1)
        return p.reshape(*shape, p.shape[1], 3)

    def spheres_np(self, q) -> np.ndarray:
        with torch.no_grad():
            qt = torch.as_tensor(np.asarray(q, dtype=np.float32)[None, :NUM_JOINTS],
                                 device=self.device)
            return self.spheres(qt)[0].cpu().numpy().astype(np.float64)

    # ---------------------------------------------------------------- helpers
    def ik_to_tcp(self, tcp_xyz, *, obstacles=None, iters: int = 300,
                  avoid_gain: float = 0.4, clear_target: float = 0.03) -> np.ndarray:
        """Joints reaching a tool position (orientation free), by damped least squares.

        With ``obstacles`` given, a nullspace term additionally pushes the whole
        arm out of them while holding the tool fixed -- using the same sphere
        cloud the planner and referee use, so the start pose is collision-free
        in the experiment's own model.
        """

        q = np.asarray(self.home_qpos, dtype=np.float64).copy()
        target = np.asarray(tcp_xyz, dtype=np.float64)
        lo, hi = self.joint_low + 0.02, self.joint_high - 0.02
        eps, ceps = 1e-4, 1e-3
        for _ in range(iters):
            p = self.ee_of_q(q)
            err = target - p
            J = np.zeros((3, NUM_JOINTS))
            for i in range(NUM_JOINTS):
                dq = q.copy()
                dq[i] += eps
                J[:, i] = (self.ee_of_q(dq) - p) / eps
            Jp = J.T @ np.linalg.solve(J @ J.T + 1e-4 * np.eye(3), np.eye(3))
            step = Jp @ err
            clear = None
            if obstacles is not None:
                clear = obstacles.clearance(self.spheres_np(q), self.sphere_radii_np)
                if clear < clear_target:
                    g = np.zeros(NUM_JOINTS)
                    for i in range(NUM_JOINTS):
                        dq = q.copy()
                        dq[i] += ceps
                        g[i] = (obstacles.clearance(self.spheres_np(dq),
                                                    self.sphere_radii_np) - clear) / ceps
                    step = step + avoid_gain * ((np.eye(NUM_JOINTS) - Jp @ J) @ g)
            norm = float(np.linalg.norm(step))
            if norm > 0.2:
                step *= 0.2 / norm
            q = np.clip(q + step, lo, hi)
            if np.linalg.norm(err) < 1e-4 and (obstacles is None or clear is None
                                               or clear >= clear_target):
                break
        return q

    def clearance(self, q, obstacles) -> float:
        """Smallest surface gap between the arm at ``q`` and any obstacle."""

        if obstacles is None:
            return float("inf")
        return obstacles.clearance(self.spheres_np(q), self.sphere_radii_np)
