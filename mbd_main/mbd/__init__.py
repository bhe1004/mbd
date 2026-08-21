"""Model-Based Diffusion with a bilinear Koopman model.

Environment-agnostic by construction: nothing in this package imports MuJoCo,
the FR3, or any robot description. The planner is wired to an environment
through the two protocols in :mod:`mbd.types`.
"""

from .koopman import BilinearKoopman, KoopmanArchitecture, KoopmanPredictor
from .optimizer import AdaptiveNoise, MBDOptimizer, MBDSettings
from .planner import BKMBDPlanner
from .training import TrainSettings, load_checkpoint, save_checkpoint, train
from .types import CandidateCost, ModelRollout, PlanResult, PredictiveModel

__all__ = [
    "AdaptiveNoise",
    "BKMBDPlanner",
    "BilinearKoopman",
    "CandidateCost",
    "KoopmanArchitecture",
    "KoopmanPredictor",
    "MBDOptimizer",
    "MBDSettings",
    "ModelRollout",
    "PlanResult",
    "PredictiveModel",
    "TrainSettings",
    "load_checkpoint",
    "save_checkpoint",
    "train",
]
