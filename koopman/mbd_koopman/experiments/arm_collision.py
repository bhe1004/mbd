"""Whole-arm collision for the realtime experiment: sphere cloud + one ball.

Ported from the controlled box-world experiments (version 2/exper): the arm
is covered by spheres along the kinematic chain plus a two-sphere crossbar
for the gripper (the hand is a T: the wrist-to-tool chain is its stem, the
finger carriage its bar). The obstacle here is deliberately CONVEX -- a
single ball -- so this is about adding honest whole-body avoidance to the
realtime loop, not about trap geometry.

Two consumers, one geometry:

``SphereObstacle.penalty``
    soft quadratic penetration cost, batched over candidate rollouts,
    inflated by (sphere radius + margin) -- what the sampling planner
    charges per step (and between steps, so a fast candidate cannot cross
    the ball unseen between two control instants).
``SphereObstacle.violations``
    the honest referee, margin-free: is any executed arm sphere actually
    inside the ball.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch

from envs.franka import NUM_JOINTS, FrankaTask, _axis_angle_mat


class ArmFK:
    """Batched FK giving the whole-arm collision spheres (chain + gripper T)."""

    def __init__(self, task: FrankaTask, device: torch.device | str = "cpu",
                 link_samples: int = 3, arm_radius: float = 0.05,
                 tcp_radius: float = 0.04, first_link: int = 3,
                 hand_back: float = 0.05, hand_side: float = 0.05,
                 hand_radius: float = 0.035, hand_fingers: bool = True):
        segs, tail = task._fk_params
        self.device = torch.device(device)
        self.fixed = [torch.as_tensor(f, dtype=torch.float32, device=self.device)
                      for f, _ in segs]
        self.axes = [a for _, a in segs]
        self.tail = torch.as_tensor(tail, dtype=torch.float32, device=self.device)
        #: link_samples: how many extra spheres to interpolate between each
        #: pair of consecutive frame origins (denser -> better segment cover).
        self.link_samples = int(link_samples)
        #: first_link: which frame origin the body cover starts at (0=base ...
        #: 4=elbow ... 8=TCP). Lower reaches further up the arm toward the base.
        self.first_link = int(first_link)
        #: The gripper is a T/U: a crossbar `hand_back` behind the tool, and --
        #: when hand_fingers -- two finger prongs at the tool plane. Both span
        #: +-hand_side along the finger-separation axis.
        self.hand_back = float(hand_back)
        self.hand_side = float(hand_side)
        self.hand_fingers = bool(hand_fingers)
        self.n_hand = (0 if self.hand_side <= 0
                       else 4 if self.hand_fingers else 2)

        # origin order: base, link1..link7, TCP
        radii = ([arm_radius] * (1 + NUM_JOINTS) + [tcp_radius])[self.first_link:]
        m = len(radii)
        all_r = list(radii)
        for _ in range(self.link_samples):
            all_r += [arm_radius] * (m - 1)
        all_r += [hand_radius] * self.n_hand
        self.radii = torch.as_tensor(all_r, dtype=torch.float32, device=self.device)
        self.radii_np = self.radii.cpu().numpy().astype(np.float64)
        self.num_points = len(all_r)

    def _chain(self, qf: torch.Tensor) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        n = qf.shape[0]
        T = torch.eye(4, dtype=torch.float32, device=self.device).expand(n, 4, 4)
        outs = [torch.zeros(n, 3, dtype=torch.float32, device=self.device)]
        for i in range(NUM_JOINTS):
            T = T @ self.fixed[i]
            T = T @ _axis_angle_mat(self.axes[i], qf[:, i], torch.float32,
                                    self.device)
            outs.append(T[:, :3, 3])
        T = T @ self.tail
        outs.append(T[:, :3, 3])
        return T, outs

    def spheres(self, q: torch.Tensor) -> torch.Tensor:
        """q: (..., 7) -> (..., P, 3) collision sphere centres."""

        shape = q.shape[:-1]
        T, outs = self._chain(q.reshape(-1, NUM_JOINTS))
        o = torch.stack(outs[self.first_link:], dim=1)          # (N, m, 3)
        pts = [o]
        for s in range(1, self.link_samples + 1):
            f = s / (self.link_samples + 1)
            pts.append(o[:, :-1] * (1 - f) + o[:, 1:] * f)
        if self.n_hand:
            tcp = T[:, :3, 3]
            approach = T[:, :3, 2]                # out between the fingers
            finger = T[:, :3, 1]                  # fingers separate along this
            bar = tcp - self.hand_back * approach
            hand = [bar + self.hand_side * finger,   # crossbar, behind the tool
                    bar - self.hand_side * finger]
            if self.hand_fingers:
                hand += [tcp + self.hand_side * finger,   # finger prongs, at
                         tcp - self.hand_side * finger]    # the tool plane
            pts.append(torch.stack(hand, dim=1))
        p = torch.cat(pts, dim=1)
        return p.reshape(*shape, p.shape[1], 3)

    def spheres_np(self, q: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            qt = torch.as_tensor(np.asarray(q, dtype=np.float32)[None],
                                 device=self.device)
            return self.spheres(qt)[0].cpu().numpy().astype(np.float64)

    def spheres_and_jacobians(self, q: torch.Tensor, eps: float = 1e-3):
        """q: (N, 7) -> (points (N, P, 3), J (N, P, 3, 7)) with J the central
        finite-difference Jacobian of every collision-sphere center w.r.t. the
        joints.

        A sphere center is a trigonometric function of the joints and cannot
        enter a quadratic program directly, so a convexified planner linearizes
        it as ``p ~= p_nom + J (q - q_nom)`` around the nominal joints. One
        batched FK call of size 14N keeps it cheap. Covers the gripper spheres
        too (they move with the tool frame, so FD captures them). Mirrors
        version 2/exper/kinematics.py:spheres_and_jacobians."""

        n = q.shape[0]
        p0 = self.spheres(q)                                   # (N, P, 3)
        eye = eps * torch.eye(NUM_JOINTS, dtype=q.dtype, device=self.device)
        qp = (q[:, None] + eye[None]).reshape(-1, NUM_JOINTS)  # (N*7, 7)
        qm = (q[:, None] - eye[None]).reshape(-1, NUM_JOINTS)
        pp = self.spheres(torch.cat([qp, qm], dim=0))          # (2*N*7, P, 3)
        d = pp[: n * NUM_JOINTS] - pp[n * NUM_JOINTS:]
        J = d.reshape(n, NUM_JOINTS, self.num_points, 3).permute(0, 2, 3, 1)
        return p0, J / (2.0 * eps)


class ObstacleField:
    """A set of keep-out obstacles -- spheres AND axis-aligned boxes -- that
    charge the planner and are scored margin-free by the referee.

    ``obstacles`` is a list of shape-tagged entries in the base frame::

        ("sphere", (x, y, z), radius)
        ("box",    (x, y, z), (hx, hy, hz))     # center + half-extents

    The obstacle layout is DATA, edited as a list in the experiment script,
    not scattered across CLI flags. Both shapes are penalized the same way --
    a flat cost per overlapping (arm sphere, obstacle) pair (hard) or a
    graded depth^2 (soft) -- and the referee uses the same in/out test on
    both.
    """

    def __init__(self, obstacles, margin: float = 0.02, weight: float = 5000.0,
                 device: torch.device | str = "cpu", hard: bool = True,
                 floor_z=None):
        dev = torch.device(device)
        self.margin = float(margin)
        self.weight = float(weight)
        #: hard=True: a flat ``weight`` per overlapping (arm sphere, obstacle)
        #: pair, no depth. hard=False: the graded ``weight * depth^2``.
        self.hard = bool(hard)
        #: keep every arm sphere above this z (ground plane); None = no floor.
        self.floor_z = None if floor_z is None else float(floor_z)

        sph = [(c, r) for kind, c, r in obstacles if kind == "sphere"]
        box = [(c, h) for kind, c, h in obstacles if kind == "box"]
        unknown = {kind for kind, *_ in obstacles} - {"sphere", "box"}
        if unknown:
            raise ValueError(f"unknown obstacle kind(s): {sorted(unknown)} "
                             "(use 'sphere' or 'box')")

        # spheres: center + radius
        self.s_centers_np = np.asarray([c for c, _ in sph],
                                       dtype=np.float64).reshape(-1, 3)
        self.s_radii_np = np.asarray([r for _, r in sph], dtype=np.float64)
        self.s_centers = torch.as_tensor(self.s_centers_np, dtype=torch.float32,
                                         device=dev)
        self.s_radii = torch.as_tensor(self.s_radii_np, dtype=torch.float32,
                                       device=dev)
        # boxes: center + half-extents
        self.b_centers_np = np.asarray([c for c, _ in box],
                                       dtype=np.float64).reshape(-1, 3)
        self.b_halfs_np = np.asarray([h for _, h in box],
                                     dtype=np.float64).reshape(-1, 3)
        self.b_centers = torch.as_tensor(self.b_centers_np, dtype=torch.float32,
                                         device=dev)
        self.b_halfs = torch.as_tensor(self.b_halfs_np, dtype=torch.float32,
                                       device=dev)

    def __len__(self) -> int:
        return len(self.s_radii_np) + len(self.b_halfs_np)

    # for the viewer overlay
    @property
    def spheres_draw(self):
        return list(zip(self.s_centers_np, self.s_radii_np))

    @property
    def boxes_draw(self):
        return list(zip(self.b_centers_np, self.b_halfs_np))

    # ------------------------------------------------------- sampling planner
    def _pair_cost(self, depth: torch.Tensor) -> torch.Tensor:
        """depth: (..., P, K) leave-distance per (arm sphere, obstacle) pair,
        positive when overlapping. Reduce over (P, K) to one cost per
        candidate."""

        if self.hard:
            return self.weight * (depth > 0).to(depth.dtype).sum(dim=(-2, -1))
        return self.weight * depth.clamp(min=0.0).pow(2).sum(dim=(-2, -1))

    def penalty(self, points: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """points: (..., P, 3), radii: (P,) -> (...,) summed overlap cost.

        Reduces over the arm-sphere and obstacle axes ONLY, preserving any
        leading batch dims: batched callers get one penalty per candidate.

        ``hard`` (default): each (arm sphere, obstacle) pair whose surfaces
        overlap costs a flat ``weight``, regardless of depth -- so the cost
        is ``weight * (number of overlapping pairs)``, matching the referee's
        pure in/out test. Counting pairs (not a single global indicator)
        keeps a coarse "touch fewer spheres" gradient, which matters because
        MBD's ``softmax(-cost/alpha)`` is invariant to a constant added to
        every sample: if all candidates overlapped by the same count a single
        indicator would cancel and stop steering. A graded depth^2 penalty
        (``hard=False``) is strictly safer once the arm is already buried.

        Sphere overlap: ``|p - c| < r_obs + r_arm + margin``.
        Box overlap: ``p`` inside the box grown by ``r_arm + margin`` on every
        axis; the leave-distance is the smallest per-axis slack (moving out
        the nearest face), which is the box analogue of the sphere depth.
        """

        cost = None
        infl = radii[:, None] + self.margin                       # (P,1)
        if self.s_radii.numel():
            d = torch.linalg.norm(points.unsqueeze(-2) - self.s_centers, dim=-1)
            depth_s = (self.s_radii + infl) - d                   # (...,P,Ks)
            cost = self._pair_cost(depth_s)
        if self.b_halfs.numel():
            diff = torch.abs(points.unsqueeze(-2) - self.b_centers)  # (...,P,Kb,3)
            slack = (self.b_halfs + infl[..., None]) - diff
            inside = (slack > 0).all(dim=-1)                      # (...,P,Kb)
            depth_b = slack.min(dim=-1).values * inside.to(slack.dtype)
            cost_b = self._pair_cost(depth_b)
            cost = cost_b if cost is None else cost + cost_b
        if self.floor_z is not None:
            # keep-out below the ground plane: depth = how far the sphere's
            # lower surface has sunk past (floor + margin), as one extra "face".
            depth_f = (self.floor_z + infl.squeeze(-1)) - points[..., 2]  # (...,P)
            cost_f = self._pair_cost(depth_f.unsqueeze(-1))
            cost = cost_f if cost is None else cost + cost_f
        if cost is None:                                          # no obstacles
            return torch.zeros(points.shape[:-2], dtype=points.dtype,
                               device=points.device)
        return cost

    def segment_penalty(self, prev_pts: torch.Tensor, pts: torch.Tensor,
                        radii: torch.Tensor, substeps: int = 2) -> torch.Tensor:
        """Charge interpolated points between two instants, so a candidate
        cannot cross an obstacle between control steps for free."""

        total = torch.zeros(pts.shape[:-2], dtype=pts.dtype, device=pts.device)
        for i in range(substeps):
            f = (i + 1.0) / (substeps + 1.0)
            total = total + self.penalty(prev_pts + f * (pts - prev_pts), radii)
        return total

    # ------------------------------------------------------------- the referee
    def violations(self, points: np.ndarray, radii: np.ndarray) -> int:
        """Exact, margin-free: how many arm spheres are inside any obstacle."""

        r = np.asarray(radii)[:, None]
        hit = np.zeros(points.shape[0], dtype=bool)
        if self.s_radii_np.size:
            d = np.linalg.norm(points[:, None, :] - self.s_centers_np[None],
                               axis=-1)
            hit |= (d < self.s_radii_np[None] + r).any(axis=1)
        if self.b_halfs_np.size:
            diff = np.abs(points[:, None, :] - self.b_centers_np[None])
            hit |= ((diff < self.b_halfs_np[None] + r[..., None]).all(axis=-1)
                    ).any(axis=1)
        if self.floor_z is not None:
            hit |= (points[:, 2] - np.asarray(radii) < self.floor_z)
        return int(hit.sum())

    def clearance(self, points: np.ndarray, radii: np.ndarray) -> float:
        """Smallest surface-to-surface distance between any arm sphere and any
        obstacle (negative = penetrating)."""

        r = np.asarray(radii)[:, None]
        gaps = [np.inf]
        if self.s_radii_np.size:
            d = np.linalg.norm(points[:, None, :] - self.s_centers_np[None],
                               axis=-1)
            gaps.append(float((d - self.s_radii_np[None] - r).min()))
        if self.b_halfs_np.size:
            g = np.abs(points[:, None, :] - self.b_centers_np[None]) \
                - self.b_halfs_np[None]                            # (P,Kb,3)
            outside = np.linalg.norm(np.clip(g, 0.0, None), axis=-1)
            inside = g.max(axis=-1)                                # <0 when in
            sd = np.where((g < 0).all(axis=-1), inside, outside)   # (P,Kb)
            gaps.append(float((sd - r).min()))
        if self.floor_z is not None:
            gaps.append(float((points[:, 2] - np.asarray(radii) - self.floor_z).min()))
        return float(min(gaps))
