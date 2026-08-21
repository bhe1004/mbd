"""BK-MBD planner: MBD sampling over a bilinear Koopman rollout.

The planner is the only place the three pieces meet, and it holds none of them
itself:

* a ``PredictiveModel`` (the learned bilinear Koopman rollout) predicts the
  feature trajectory of every candidate,
* a ``CandidateCost`` (supplied by the environment) scores those trajectories,
* :class:`~mbd.optimizer.MBDOptimizer` anneals the sampling noise and returns
  the softmax-weighted update.

Nothing here knows what a joint, a gripper, or a MuJoCo model is. Handing it a
different environment's cost and a model trained on that environment's data is
the whole port.
"""

from __future__ import annotations

import numpy as np
import torch

from .optimizer import AdaptiveNoise, MBDOptimizer, MBDSettings
from .types import Array, CandidateCost, PlanResult, PredictiveModel


class BKMBDPlanner:
    """Plan a control sequence with Model-Based Diffusion."""

    name = "bk_mbd"

    def __init__(
        self,
        model: PredictiveModel,
        cost: CandidateCost,
        *,
        settings: MBDSettings,
        action_low: Array,
        action_high: Array,
        horizon: int,
        device: torch.device | str = "cpu",
        adaptive_enabled: bool = True,
        adaptive_err_full: float = 0.4,
        adaptive_floor: float = 0.05,
    ) -> None:
        self.model = model
        self.cost = cost
        self.horizon = int(horizon)
        self.action_dim = len(np.asarray(action_low))
        self.device = torch.device(device)

        self.optimizer = MBDOptimizer(settings, action_low, action_high)
        self.adaptive = AdaptiveNoise(self.optimizer, enabled=adaptive_enabled,
                                      err_full=adaptive_err_full, floor=adaptive_floor)

    # -------------------------------------------------------------- scoring
    def _evaluate_fn(self, features_now: Array, goal: Array):
        """Build the closure the optimizer calls with a batch of candidates."""

        def evaluate(candidates: Array) -> Array:
            with torch.no_grad():
                U = torch.as_tensor(np.asarray(candidates, dtype=np.float32),
                                    dtype=torch.float32, device=self.device)
                out = self.model.rollout(features_now, U)
                costs = self.cost(out.features, U, goal, features_now=features_now)
                return costs.cpu().numpy()

        return evaluate

    def _predict(self, features_now: Array, U: Array) -> Array:
        with torch.no_grad():
            U_t = torch.as_tensor(np.asarray(U, dtype=np.float32)[None],
                                  dtype=torch.float32, device=self.device)
            return self.model.rollout(features_now, U_t).features[0].cpu().numpy()

    # --------------------------------------------------------------- public
    def plan(self, features_now: Array, goal: Array, U_warm: Array, err: float,
             rng: np.random.Generator) -> PlanResult:
        """One plan from the measured features toward ``goal``.

        Args:
            features_now: measured features the rollout starts from.
            goal: whatever the cost interprets as the target.
            U_warm: warm-start control sequence, (T, action_dim).
            err: current task error, used only by the adaptive noise schedule.
        """

        evaluate = self._evaluate_fn(features_now, goal)
        result = self.adaptive.optimize(U_warm, evaluate, rng, err)
        return PlanResult(controls=result.controls,
                          predicted_features=self._predict(features_now, result.controls),
                          best_cost=result.best_cost)

    def warmup(self, features_now: Array, goal: Array, num_plans: int) -> None:
        """Throwaway plans so torch/threadpool warm-up does not pollute latency."""

        if num_plans <= 0:
            return
        rng = np.random.default_rng(10**6 + self.optimizer.settings.seed)
        evaluate = self._evaluate_fn(features_now, goal)
        U0 = np.zeros((self.horizon, self.action_dim), dtype=np.float64)
        for _ in range(num_plans):
            self.optimizer.optimize(U0, evaluate, rng=rng)


__all__ = ["BKMBDPlanner"]
