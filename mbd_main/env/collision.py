"""Keep-out obstacles: what the planner is charged for and what the referee checks.

One geometry, two consumers:

``ObstacleField.penalty``
    batched over candidate rollouts, inflated by (arm sphere radius + margin) --
    the soft cost the sampling planner pays per step, and at interpolated
    points between steps so a fast candidate cannot cross an obstacle unseen
    between two control instants.
``ObstacleField.violations``
    the honest referee, margin-free: is any *executed* arm sphere actually
    inside an obstacle. The planner's margin never flatters this number.

Obstacles are data, listed in the config file, not scattered across flags.
Each entry is ``["sphere", [x, y, z], r]`` or ``["box", [x, y, z], [hx, hy, hz]]``
in the robot base frame; a sphere centre may be the string ``"auto"``, which the
task places on the line from the home tool position to the first goal.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np
import torch


@dataclass(frozen=True)
class CollisionConfig:
    """The ``collision`` block of the config file.

    Attributes:
        margin: metres the planner (never the referee) inflates every obstacle by.
        weight: penalty scale -- a flat cost per overlapping (arm sphere,
            obstacle) pair when ``hard``, else per squared metre of penetration.
        hard: binary overlap penalty (default) vs a graded depth^2 one.
        substeps: interpolated penalty samples between two control instants.
        floor_z: keep every arm sphere above this height; None disables it.
    """

    margin: float = 0.02
    weight: float = 50000.0
    hard: bool = True
    substeps: int = 2
    floor_z: Optional[float] = 0.0


class ObstacleField:
    """A set of spheres and axis-aligned boxes, plus an optional floor plane."""

    def __init__(self, obstacles: Sequence, config: CollisionConfig | None = None,
                 device: torch.device | str = "cpu") -> None:
        cfg = config or CollisionConfig()
        self.config = cfg
        dev = torch.device(device)

        unknown = {kind for kind, *_ in obstacles} - {"sphere", "box"}
        if unknown:
            raise ValueError(f"unknown obstacle kind(s): {sorted(unknown)} (use sphere/box)")

        sph = [(c, r) for kind, c, r in obstacles if kind == "sphere"]
        box = [(c, h) for kind, c, h in obstacles if kind == "box"]

        self.sphere_centers_np = np.asarray([c for c, _ in sph], dtype=np.float64).reshape(-1, 3)
        self.sphere_radii_np = np.asarray([r for _, r in sph], dtype=np.float64).reshape(-1)
        self.box_centers_np = np.asarray([c for c, _ in box], dtype=np.float64).reshape(-1, 3)
        self.box_halfs_np = np.asarray([h for _, h in box], dtype=np.float64).reshape(-1, 3)

        as_t = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
        self.sphere_centers = as_t(self.sphere_centers_np)
        self.sphere_radii = as_t(self.sphere_radii_np)
        self.box_centers = as_t(self.box_centers_np)
        self.box_halfs = as_t(self.box_halfs_np)

    def __len__(self) -> int:
        return int(self.sphere_radii_np.size + self.box_halfs_np.shape[0])

    # for the viewer overlay: exactly the shapes that are being checked
    @property
    def spheres_draw(self):
        return list(zip(self.sphere_centers_np, self.sphere_radii_np))

    @property
    def boxes_draw(self):
        return list(zip(self.box_centers_np, self.box_halfs_np))

    # ------------------------------------------------------- sampling planner
    def _pair_cost(self, depth: torch.Tensor) -> torch.Tensor:
        """depth (..., P, K): positive where an (arm sphere, obstacle) pair overlaps."""

        if self.config.hard:
            return self.config.weight * (depth > 0).to(depth.dtype).sum(dim=(-2, -1))
        return self.config.weight * depth.clamp(min=0.0).pow(2).sum(dim=(-2, -1))

    def penalty(self, points: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """points (..., P, 3), radii (P,) -> (...,) overlap cost per candidate.

        Reduces over the arm-sphere and obstacle axes only, so leading batch
        dims survive.

        ``hard`` charges a flat ``weight`` per overlapping pair, matching the
        referee's pure in/out test. Counting *pairs* rather than raising one
        global flag keeps a coarse "touch fewer spheres" gradient, which matters
        because MBD's ``softmax(-cost/alpha)`` is invariant to a constant added
        to every sample: a single indicator would cancel out and stop steering
        once every candidate overlapped.
        """

        cost = None
        infl = radii[:, None] + self.config.margin                     # (P, 1)
        if self.sphere_radii.numel():
            d = torch.linalg.norm(points.unsqueeze(-2) - self.sphere_centers, dim=-1)
            cost = self._pair_cost((self.sphere_radii + infl) - d)
        if self.box_halfs.numel():
            diff = torch.abs(points.unsqueeze(-2) - self.box_centers)   # (..., P, K, 3)
            slack = (self.box_halfs + infl[..., None]) - diff
            inside = (slack > 0).all(dim=-1)
            depth = slack.min(dim=-1).values * inside.to(slack.dtype)
            cost_b = self._pair_cost(depth)
            cost = cost_b if cost is None else cost + cost_b
        if self.config.floor_z is not None:
            depth_f = (self.config.floor_z + infl.squeeze(-1)) - points[..., 2]
            cost_f = self._pair_cost(depth_f.unsqueeze(-1))
            cost = cost_f if cost is None else cost + cost_f
        if cost is None:
            return torch.zeros(points.shape[:-2], dtype=points.dtype, device=points.device)
        return cost

    def swept_penalty(self, points: torch.Tensor, previous: torch.Tensor,
                      radii: torch.Tensor) -> torch.Tensor:
        """Penalty at each step plus at interpolated points since the last one."""

        total = self.penalty(points, radii)
        for i in range(self.config.substeps):
            f = (i + 1.0) / (self.config.substeps + 1.0)
            total = total + self.penalty(previous + f * (points - previous), radii)
        return total

    # ------------------------------------------------------------- the referee
    def violations(self, points: np.ndarray, radii: np.ndarray) -> int:
        """Margin-free: how many arm spheres are inside any obstacle."""

        r = np.asarray(radii)[:, None]
        hit = np.zeros(points.shape[0], dtype=bool)
        if self.sphere_radii_np.size:
            d = np.linalg.norm(points[:, None, :] - self.sphere_centers_np[None], axis=-1)
            hit |= (d < self.sphere_radii_np[None] + r).any(axis=1)
        if self.box_halfs_np.size:
            diff = np.abs(points[:, None, :] - self.box_centers_np[None])
            hit |= (diff < self.box_halfs_np[None] + r[..., None]).all(axis=-1).any(axis=1)
        if self.config.floor_z is not None:
            hit |= points[:, 2] - np.asarray(radii) < self.config.floor_z
        return int(hit.sum())

    def clearance(self, points: np.ndarray, radii: np.ndarray) -> float:
        """Smallest surface-to-surface gap (negative = already penetrating)."""

        r = np.asarray(radii)[:, None]
        gaps = [np.inf]
        if self.sphere_radii_np.size:
            d = np.linalg.norm(points[:, None, :] - self.sphere_centers_np[None], axis=-1)
            gaps.append(float((d - self.sphere_radii_np[None] - r).min()))
        if self.box_halfs_np.size:
            g = np.abs(points[:, None, :] - self.box_centers_np[None]) - self.box_halfs_np[None]
            outside = np.linalg.norm(np.clip(g, 0.0, None), axis=-1)
            inside = g.max(axis=-1)                                  # < 0 when inside
            gaps.append(float((np.where((g < 0).all(axis=-1), inside, outside) - r).min()))
        if self.config.floor_z is not None:
            gaps.append(float((points[:, 2] - np.asarray(radii) - self.config.floor_z).min()))
        return float(min(gaps))
