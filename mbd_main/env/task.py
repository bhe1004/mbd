"""The reaching task: what a "feature" is, and what a candidate trajectory costs.

This is the environment's half of the contract with :mod:`mbd`:

``features(q)``
    the vector the Koopman model learns and predicts, ``b = [q (7), ee (3)]``.
    Joints are what the robot is commanded in; the tool position is what the
    task is about. Both are measurable on the real arm, so the same feature
    definition holds there.
``candidate_cost(...)``
    scores a batch of predicted feature trajectories: reach the target, spend
    little control, stay off the joint stops, keep the whole arm out of the
    obstacles.

The planner calls exactly these two and knows nothing else about the robot.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch

from .collision import ObstacleField
from .robot import NUM_JOINTS, FrankaRobot

FEATURE_DIM = NUM_JOINTS + 3


@dataclass(frozen=True)
class CostWeights:
    """Reaching cost weights."""

    ee: float = 1.0                  # per-step squared tool error
    control: float = 0.002           # per-step squared joint velocity
    terminal_ee: float = 10.0        # squared tool error at the horizon end
    joint_limit: float = 200.0       # squared encroachment past the inset limits
    joint_limit_margin: float = 0.3  # [rad] inset before the true joint stops


@dataclass(frozen=True)
class TaskConfig:
    """The ``task`` block of the config file."""

    control_dt: float = 0.05
    horizon: int = 15
    action_limit: float = 1.0        # commanded joint-velocity cap [rad/s]
    reach_threshold: float = 0.05    # counted as reached [m]
    strict_threshold: float = 0.025  # counted as *settled* [m]
    weights: CostWeights = field(default_factory=CostWeights)


def default_targets() -> List[np.ndarray]:
    """Reach targets spread around the FR3 home tool position."""

    return [np.array(t, dtype=np.float64) for t in [
        [0.45, 0.35, 0.35],
        [0.45, -0.35, 0.35],
        [0.65, 0.00, 0.30],
        [0.35, 0.40, 0.60],
        [0.35, -0.40, 0.60],
        [0.30, 0.00, 0.80],
        [0.60, 0.20, 0.55],
    ]]


class ReachTask:
    """Tool-position reaching with whole-arm obstacle avoidance."""

    feature_dim = FEATURE_DIM
    action_dim = NUM_JOINTS

    def __init__(self, robot: FrankaRobot, config: TaskConfig | None = None,
                 obstacles: Optional[ObstacleField] = None,
                 targets: Optional[Sequence[np.ndarray]] = None,
                 device: torch.device | str = "cpu") -> None:
        self.robot = robot
        self.config = config or TaskConfig()
        self.obstacles = obstacles
        self.device = torch.device(device)
        self.targets = [np.asarray(t, dtype=np.float64) for t in
                        (targets if targets is not None else default_targets())]

        limit = float(self.config.action_limit)
        self.action_low = -limit * np.ones(self.action_dim, dtype=np.float64)
        self.action_high = limit * np.ones(self.action_dim, dtype=np.float64)

        w = self.config.weights
        self._jl_low = torch.as_tensor(robot.joint_low + w.joint_limit_margin,
                                       dtype=torch.float32, device=self.device)
        self._jl_high = torch.as_tensor(robot.joint_high - w.joint_limit_margin,
                                        dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------- features
    def features(self, q: np.ndarray, ee: Optional[np.ndarray] = None) -> np.ndarray:
        """b = [q, ee]. ``ee`` may be the measured tool position; else FK is used."""

        q = np.asarray(q, dtype=np.float64)[:NUM_JOINTS]
        tool = self.robot.ee_of_q(q) if ee is None else np.asarray(ee, dtype=np.float64)
        return np.concatenate([q, tool])

    @staticmethod
    def joints_of(features: torch.Tensor) -> torch.Tensor:
        return features[..., :NUM_JOINTS]

    @staticmethod
    def tool_of(features: torch.Tensor) -> torch.Tensor:
        return features[..., NUM_JOINTS : NUM_JOINTS + 3]

    # ----------------------------------------------------------------- cost
    def candidate_cost(self, features: torch.Tensor, controls: torch.Tensor,
                       goal, *, features_now) -> torch.Tensor:
        """Score a batch of predicted feature trajectories.

        Args:
            features: (K, T + 1, feature_dim) predicted [q, ee].
            controls: (K, T, action_dim).
            goal: target tool position -- either a single point (3,) held for
                the whole horizon, or one target per step (T, 3). The per-step
                form is what lets the planner track a MOVING target without a
                standing lag: scoring against a target frozen at its present
                position costs roughly (target speed) x (loop delay) of error.
            features_now: the measured features the rollout started from -- the
                obstacle term needs them to charge the motion *into* the first
                predicted step, not just the steps themselves.
        """

        if features.ndim != 3 or controls.ndim != 3:
            raise ValueError("features must be (K,T+1,d) and controls (K,T,m)")
        if features.shape[0] != controls.shape[0] or features.shape[1] != controls.shape[1] + 1:
            raise ValueError("features and controls have incompatible shapes")

        w = self.config.weights
        target = torch.as_tensor(np.asarray(goal, dtype=np.float32),
                                 dtype=features.dtype, device=features.device)
        tool = self.tool_of(features[:, 1:])                  # (K, T, 3)
        if target.ndim == 2:
            if target.shape != tool.shape[1:]:
                raise ValueError(f"a per-step goal must have shape {tuple(tool.shape[1:])}, "
                                 f"got {tuple(target.shape)}")
            terminal_target = target[-1]
        elif target.ndim == 1:
            terminal_target = target
        else:
            raise ValueError("goal must be (3,) or (T, 3)")
        running = (w.ee * torch.sum((tool - target) ** 2, dim=-1)
                   + w.control * torch.sum(controls ** 2, dim=-1))
        cost = torch.sum(running, dim=-1)
        cost = cost + w.terminal_ee * torch.sum((tool[:, -1] - terminal_target) ** 2, dim=-1)

        joints = self.joints_of(features)
        if w.joint_limit:
            cost = cost + self.joint_limit_penalty(joints[:, 1:])
        if self.obstacles is not None:
            cost = cost + self.obstacle_penalty(joints, features_now)
        return cost

    def joint_limit_penalty(self, joints: torch.Tensor) -> torch.Tensor:
        """(K, T, 7) -> (K,) squared encroachment past the margin-inset limits.

        Keeps the sampler from proposing plans that ride a joint stop, where
        the velocity servo stops tracking and the learned model is worst.
        """

        over = (joints - self._jl_high).clamp(min=0.0)
        under = (self._jl_low - joints).clamp(min=0.0)
        return self.config.weights.joint_limit * (over.pow(2) + under.pow(2)).sum(dim=(-2, -1))

    def obstacle_penalty(self, joints: torch.Tensor, features_now) -> torch.Tensor:
        """(K, T+1, 7) -> (K,) whole-arm penalty, swept between control instants.

        No convexification and no linearization: every predicted pose is pushed
        through the true kinematics and every arm sphere is charged. This is the
        sampling planner's native way of eating geometry.
        """

        points = self.robot.spheres(joints)                       # (K, T+1, P, 3)
        q_now = torch.as_tensor(np.asarray(features_now, dtype=np.float32)[:NUM_JOINTS],
                                device=points.device)
        first = self.robot.spheres(q_now[None])[None].expand(points.shape[0], 1, -1, -1)
        previous = torch.cat([first, points[:, :-1]], dim=1)
        return self.obstacles.swept_penalty(points, previous,
                                            self.robot.sphere_radii).sum(dim=1)

    # -------------------------------------------------------------- outcomes
    def error(self, ee: np.ndarray, goal: np.ndarray) -> float:
        return float(np.linalg.norm(np.asarray(ee) - np.asarray(goal)))

    def reached(self, ee: np.ndarray, goal: np.ndarray) -> bool:
        return self.error(ee, goal) < self.config.reach_threshold

    def settled(self, ee: np.ndarray, goal: np.ndarray) -> bool:
        return self.error(ee, goal) < self.config.strict_threshold

    # ----------------------------------------------------------------- goals
    def goal_for(self, target_id: int, override: Optional[Sequence[float]] = None
                 ) -> np.ndarray:
        """The reach point for a segment: the override if set, else target ``id``."""

        if override is not None:
            return np.asarray(override, dtype=np.float64).copy()
        if not 0 <= target_id < len(self.targets):
            raise ValueError(f"target id {target_id} out of range 0..{len(self.targets) - 1}")
        return self.targets[target_id].copy()

    def auto_obstacle_center(self, goal: np.ndarray) -> np.ndarray:
        """Where an ``"auto"`` sphere goes: 40% of the way home tool -> goal.

        Biased toward home on purpose; at the midpoint the ball sits so close to
        the target that reaching it needs the wrist inside the ball, and the
        penalty just balances the goal cost ~50 mm short.
        """

        home_ee = self.robot.ee_of_q(self.robot.home_qpos)
        return home_ee + 0.4 * (np.asarray(goal, dtype=np.float64) - home_ee)

    def start_joints(self, start_tcp: Optional[Sequence[float]]) -> Tuple[np.ndarray, str]:
        """Start configuration: IK to ``start_tcp``, or the home keyframe."""

        if start_tcp is None:
            return np.asarray(self.robot.home_qpos, dtype=np.float64).copy(), "home keyframe"
        q = self.robot.ik_to_tcp(start_tcp, obstacles=self.obstacles)
        achieved = self.robot.ee_of_q(q)
        err_mm = 1000.0 * float(np.linalg.norm(achieved - np.asarray(start_tcp)))
        return q, (f"TCP {np.round(np.asarray(start_tcp), 3)} -> "
                   f"achieved {np.round(achieved, 3)} (err {err_mm:.0f} mm)")
