"""The environment half of the project: robot, task, obstacles, and plants.

Everything simulator- or hardware-specific lives here. :mod:`mbd` imports
nothing from this package; the wiring happens in :mod:`pipeline`.
"""

from .collision import CollisionConfig, ObstacleField
from .interface import Observation, Plant
from .plant import MujocoPlant
from .robot import NUM_JOINTS, ArmBodyConfig, FrankaRobot
from .simulator import BatchSimulator, CollectConfig
from .task import CostWeights, ReachTask, TaskConfig, default_targets

__all__ = [
    "ArmBodyConfig",
    "BatchSimulator",
    "CollectConfig",
    "CollisionConfig",
    "CostWeights",
    "FrankaRobot",
    "MujocoPlant",
    "NUM_JOINTS",
    "Observation",
    "ObstacleField",
    "Plant",
    "ReachTask",
    "TaskConfig",
    "default_targets",
]
