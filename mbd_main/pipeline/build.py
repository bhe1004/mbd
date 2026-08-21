"""Wiring: this is where the environment and the algorithm are introduced.

Everywhere else the two halves stay apart -- :mod:`mbd` never imports
:mod:`env`, and :mod:`env` never imports :mod:`mbd`. They meet here, in a
handful of constructor calls driven entirely by the config file:

* the environment supplies the *features* and the *cost*,
* the checkpoint supplies the *predictive model*,
* the plant supplies *execution*.

To run the same planner on different hardware, change ``run.plant``. To plan
for a different task, supply a different cost. Neither touches the planner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np
import torch

from config import Config
from env.collision import ObstacleField
from env.interface import Plant
from env.plant import MujocoPlant
from env.robot import FrankaRobot
from env.simulator import BatchSimulator
from env.task import ReachTask
from mbd.koopman import KoopmanPredictor
from mbd.planner import BKMBDPlanner
from mbd.training import load_checkpoint


@dataclass
class Environment:
    """Everything task-side, assembled and consistent."""

    robot: FrankaRobot
    task: ReachTask
    obstacles: Optional[ObstacleField]
    start_q: np.ndarray
    start_note: str

    def goal_for(self, target_id: int, cfg: Config) -> np.ndarray:
        return self.task.goal_for(target_id, cfg.scene.target)

    def clearance_at_start(self) -> float:
        return self.robot.clearance(self.start_q, self.obstacles)


def build_environment(cfg: Config, device: torch.device | str = "cpu") -> Environment:
    """Robot + obstacles + task + start pose, in dependency order.

    The order matters: an ``"auto"`` obstacle is placed relative to the first
    goal, and the start-pose IK avoids the obstacles using the very same sphere
    cloud the planner and the referee use -- so a start pose that reports clear
    here is clear in the experiment's own model, not merely in a mesh model that
    might disagree.
    """

    robot = FrankaRobot(cfg.robot, device=device)
    task = ReachTask(robot, cfg.task, obstacles=None,
                     targets=cfg.scene.targets, device=device)

    obstacles = None
    if cfg.scene.obstacles:
        first_goal = task.goal_for(cfg.run.target_ids[0], cfg.scene.target)
        entries = []
        for kind, center, size in cfg.scene.obstacles:
            if isinstance(center, str):          # "auto"
                center = task.auto_obstacle_center(first_goal)
            entries.append((kind, np.asarray(center, dtype=np.float64), size))
        obstacles = ObstacleField(entries, cfg.collision, device=device)
        task.obstacles = obstacles

    start_q, note = task.start_joints(cfg.scene.start_tcp)
    return Environment(robot=robot, task=task, obstacles=obstacles,
                       start_q=start_q, start_note=note)


def build_simulator(cfg: Config, env: Environment) -> BatchSimulator:
    """The batched simulator used to record training data."""

    return BatchSimulator(env.robot, control_dt=cfg.task.control_dt,
                          action_limit=cfg.task.action_limit,
                          num_threads=cfg.collect.num_threads)


def build_planner(cfg: Config, env: Environment, device: torch.device | str = "cpu",
                  verbose: bool = True) -> BKMBDPlanner:
    """Load the trained model and assemble the planner around it."""

    if not cfg.paths.checkpoint.exists():
        raise SystemExit(
            f"checkpoint not found: {cfg.paths.checkpoint}\n"
            "train one first:  python main.py collect && python main.py train")
    model, extras = load_checkpoint(cfg.paths.checkpoint, device=device)
    _check_checkpoint_matches(cfg, extras)
    if verbose:
        print(f"model: {cfg.paths.checkpoint}"
              + (f"  (val_mse={extras['val_error']:.6f})" if "val_error" in extras else ""))

    return BKMBDPlanner(
        KoopmanPredictor(model, device=device),
        env.task.candidate_cost,
        settings=cfg.mbd,
        action_low=env.task.action_low,
        action_high=env.task.action_high,
        horizon=cfg.task.horizon,
        device=device,
        adaptive_enabled=cfg.adaptive.enabled,
        adaptive_err_full=cfg.adaptive.err_full,
        adaptive_floor=cfg.adaptive.floor,
    )


def _check_checkpoint_matches(cfg: Config, extras: dict) -> None:
    """Refuse a model trained for a different plant than the one configured.

    The model learns "where the joints go after one control period at this
    velocity limit". Change that and it predicts a plant that no longer exists --
    silently, since the arrays still have the right shape. A checkpoint is
    committed to the repository so a fresh clone can run at once, which makes
    this the check that keeps the convenience honest.

    The two settings are not symmetric. ``control_dt`` defines what one step of
    the learned map means, so any change invalidates the model. ``action_limit``
    only bounds the commands: running under the trained limit stays inside the
    training distribution and is fine (and is what the hardware config does
    deliberately), while running above it asks the model to extrapolate.
    """

    trained_dt = extras.get("control_dt")
    if trained_dt is not None and abs(float(trained_dt) - cfg.task.control_dt) > 1e-9:
        raise SystemExit(
            f"{cfg.paths.checkpoint.name} was trained at control_dt={trained_dt} s "
            f"but task.control_dt is {cfg.task.control_dt} s -- one step of the "
            "learned dynamics no longer means the same thing.\n"
            "Retrain:  python main.py collect && python main.py train")

    trained_limit = extras.get("action_limit")
    if trained_limit is not None and cfg.task.action_limit > float(trained_limit) + 1e-9:
        raise SystemExit(
            f"{cfg.paths.checkpoint.name} was trained with commands up to "
            f"{trained_limit} rad/s but task.action_limit is {cfg.task.action_limit} -- "
            "the planner would ask the model to extrapolate beyond its data.\n"
            "Either lower task.action_limit or retrain at the higher limit:\n"
            "  python main.py collect && python main.py train")


def build_plant(cfg: Config, env: Environment,
                key_callback: Optional[Callable[[int], None]] = None) -> Plant:
    """The thing that actually moves: the simulator, or the real arm."""

    if cfg.run.plant == "ros2":
        from env.ros2_plant import Ros2VlaPlant
        r = cfg.ros2
        return Ros2VlaPlant(
            env.robot, control_dt=cfg.task.control_dt,
            action_limit=cfg.task.action_limit,
            joint_names=r.joint_names, chunk_topic=r.chunk_topic,
            joint_state_topic=r.joint_state_topic, goal_action=r.goal_action,
            model_name=r.model_name, state_timeout=r.state_timeout,
            start_tolerance=r.start_tolerance, ref_error_stop=r.ref_error_stop,
            use_measured_ee=r.use_measured_ee, ee_topic=r.ee_topic,
            goal_timeout_s=r.goal_timeout_s, marker_topic=r.marker_topic,
            marker_frame=r.marker_frame,
            reach_threshold=cfg.task.strict_threshold, obstacles=env.obstacles)
    if cfg.run.plant == "real":
        from env.real_plant import RealFrankaPlant
        return RealFrankaPlant(env.robot, control_dt=cfg.task.control_dt,
                               action_limit=cfg.task.action_limit)
    return MujocoPlant(env.robot, control_dt=cfg.task.control_dt,
                       action_limit=cfg.task.action_limit,
                       use_viewer=cfg.viewer.enabled, key_callback=key_callback)


def describe_setup(cfg: Config, env: Environment) -> List[str]:
    """The one-time banner: what is about to run, in the units it runs in."""

    lines = [
        f"config: {cfg.source}",
        f"robot: joint-velocity limit {cfg.task.action_limit} rad/s, "
        f"control period {cfg.task.control_dt * 1000:.0f} ms "
        f"({1 / cfg.task.control_dt:.0f} Hz)",
        f"planning horizon: {cfg.task.horizon} steps "
        f"({cfg.task.horizon * cfg.task.control_dt:.2f} s lookahead)",
        f"start pose: {env.start_note}",
    ]
    if cfg.scene.target is not None:
        lines.append(f"target override: {np.round(np.asarray(cfg.scene.target), 3)}")
    if env.obstacles is not None:
        mode = "hard (flat per overlap)" if cfg.collision.hard else "graded (depth^2)"
        lines.append(f"obstacle penalty: {mode}, weight {cfg.collision.weight}, "
                     f"margin {cfg.collision.margin} m")
        for c, r in env.obstacles.spheres_draw:
            lines.append(f"  sphere at {np.round(c, 3)} radius {r:.3f}")
        for c, h in env.obstacles.boxes_draw:
            lines.append(f"  box at {np.round(c, 3)} half-extents {np.round(h, 3)}")
        clear = env.clearance_at_start()
        lines.append(f"whole arm = {env.robot.num_spheres} spheres | "
                     f"closest one at the start pose is {clear * 1000:.0f} mm clear")
        if clear <= 0:
            lines.append("  WARNING: the arm already penetrates an obstacle at the "
                         "start pose (edit scene.obstacles / scene.start_tcp)")
    lines.append(f"adaptive noise: {'on' if cfg.adaptive.enabled else 'off'}"
                 + (f" (full schedule at err >= {cfg.adaptive.err_full} m, "
                    f"floor {cfg.adaptive.floor})" if cfg.adaptive.enabled else ""))
    return lines
