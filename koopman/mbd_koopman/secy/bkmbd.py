"""BK-MBD planner: bilinear-Koopman Model-Based Diffusion.

The sampling planner. Each plan lifts the current state, rolls a batch of
candidate control sequences through the learned bilinear Koopman model,
decodes them back to joints + tool, and scores every candidate with the task
cost plus the whole-arm obstacle penalty, the joint-limit penalty, and
(optionally) the error-tube penalty. The MBD optimizer then anneals sigma
high->low within the plan and returns the softmax-weighted update.

This is the ONLY planner file that touches the Koopman model and the tube; the
dk/dk_split/vanilla variants of the old script are gone -- only the bilinear
rollout remains.
"""

from __future__ import annotations

import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import MBDConfig, UpdateRule  # noqa: E402
from bk_mbd.mbd import MBDOptimizer  # noqa: E402
from bk_mbd.train import load_checkpoint  # noqa: E402
from bk_mbd.tube import (  # noqa: E402
    bilinear_norm_bounds,
    compute_one_step_residuals,
    cost_sensitivity_torch,
    fit_tube_constants,
    propagate_tube_batch_torch,
)
from envs.franka import NUM_JOINTS  # noqa: E402

from secy.config import Config  # noqa: E402
from secy.environment import Scene  # noqa: E402

COLOR = (0.84, 0.15, 0.16)   # overlay color for this planner


class _AdaptiveNoise:
    """Scale BOTH ends of the sigma schedule by ``clip(err/err_full, floor, 1)``
    per plan. MBD already anneals sigma high->low WITHIN a plan; this adds a
    SECOND, across-plans shrink so that once the tool is near the goal the same
    sampler stops re-blasting its converged plan with the full sigma_start
    (which otherwise leaves the tool jittering tens of mm short). Scaled
    optimizers are cached by rounded scale so a new one is not built each plan.
    """

    def __init__(self, optimizer: MBDOptimizer, base_config: MBDConfig,
                 enabled: bool, err_full: float, floor: float) -> None:
        self.optimizer = optimizer
        self.base_config = base_config
        self.enabled = bool(enabled)
        self.err_full = float(err_full)
        self.floor = float(floor)
        self._cache: dict = {}

    def optimize(self, U, evaluate, rng, err: float):
        if not self.enabled:
            return self.optimizer.optimize(U, evaluate, rng=rng)
        scale = min(1.0, max(err / self.err_full, self.floor))
        key = round(scale, 2)
        opt = self._cache.get(key)
        if opt is None:
            cfg = replace(self.base_config,
                          sigma_start=self.base_config.sigma_start * scale,
                          sigma_end=self.base_config.sigma_end * scale)
            opt = MBDOptimizer(cfg, self.optimizer.action_low,
                               self.optimizer.action_high)
            self._cache[key] = opt
        return opt.optimize(U, evaluate, rng=rng)


class BKMBDPlanner:
    """MBD over the bilinear Koopman model, with the whole-arm penalties."""

    color = COLOR
    name = "bk_mbd"

    def __init__(self, scene: Scene, cfg: Config) -> None:
        self.scene = scene
        self.cfg = cfg
        self.task = scene.task
        self.device = scene.device
        self.horizon = scene.horizon
        mbd = cfg.mbd

        # ---- the Koopman model ---------------------------------------------
        checkpoint = mbd.checkpoint or (
            PROJECT_ROOT / "out" / "franka" / "models" / f"bk_seed{mbd.seed}.pt")
        if not checkpoint.exists():
            raise SystemExit(
                f"checkpoint not found: {checkpoint}\n"
                "train it first: python experiments/train_franka_koopman.py")
        self.model, _ = load_checkpoint(checkpoint, device=self.device)
        self.model.eval()
        print(f"loaded={checkpoint}")

        # ---- the error tube (bk only, optional) ----------------------------
        self.tube_constants = None
        self._norm_a = None
        self._norm_bs = None
        if mbd.tube_mode != "none":
            print("fitting tube constants (one-step residuals on the training set)...")
            dataset = self.task.sample_dataset(mbd.data_seed)
            zs, us, residuals = compute_one_step_residuals(
                self.model, dataset["base_states"], dataset["controls"],
                device=self.device)
            self.tube_constants = fit_tube_constants(zs, us, residuals, quantile=0.999)
            norm_a, norm_bs_np = bilinear_norm_bounds(self.model.bilinear_params())
            self._norm_a = norm_a
            self._norm_bs = torch.as_tensor(norm_bs_np, dtype=torch.float32,
                                            device=self.device)
            print(f"tube constants: c_x={self.tube_constants.c_x:.4e} "
                  f"c_u={self.tube_constants.c_u:.4e} beta_e={mbd.beta_e} "
                  f"tube_mode={mbd.tube_mode}")
        else:
            print("tube disabled (tube_mode none): plain bilinear rollout, no penalty")

        # ---- the MBD optimizer + adaptive noise ----------------------------
        self._base_config = MBDConfig(
            num_samples=mbd.num_samples,
            num_diffusion_steps=mbd.num_diffusion_steps,
            sigma_start=mbd.sigma_start,
            sigma_end=mbd.sigma_end,
            alpha=mbd.alpha,
            eta=mbd.eta,
            update_rule=UpdateRule(mbd.update_rule),
            seed=mbd.seed,
            add_langevin_noise=mbd.langevin_noise,
        )
        self._optimizer = MBDOptimizer(
            self._base_config, action_low=self.task.action_bounds[0],
            action_high=self.task.action_bounds[1])
        self._adaptive = _AdaptiveNoise(
            self._optimizer, self._base_config,
            cfg.adaptive.enabled, cfg.adaptive.err_full, cfg.adaptive.floor)
        print(f"adaptive noise: {'ON' if cfg.adaptive.enabled else 'OFF'}"
              + (f" (full schedule at err >= {cfg.adaptive.err_full} m, floor "
                 f"{cfg.adaptive.floor})" if cfg.adaptive.enabled else ""))

    # ------------------------------------------------------------- rollouts
    def _rollout(self, x, U_t):
        b0 = self.task.state_to_base_torch(x, self.device)
        z0 = self.model.lift(b0).expand(U_t.shape[0], -1)
        zs = self.model.rollout(z0, U_t)
        return zs, self.model.decode(zs)

    def _make_evaluate(self, x, goal, u_prev):
        scene = self.scene
        mbd = self.cfg.mbd
        use_tube = self.tube_constants is not None

        def evaluate(candidates):
            with torch.no_grad():
                U_t = torch.as_tensor(candidates, dtype=torch.float32,
                                      device=self.device)
                zs, bs = self._rollout(x, U_t)
                costs = self.task.trajectory_cost_base_torch(
                    bs, U_t, goal, u_prev=u_prev)
                if scene.obstacle is not None:
                    costs = costs + scene.arm_penalty(bs[..., :NUM_JOINTS], x)
                    costs = costs + scene.joint_limit_penalty(bs[:, 1:, :NUM_JOINTS])
                if use_tube:
                    tubes = propagate_tube_batch_torch(
                        zs, U_t, norm_a=self._norm_a, norm_bs=self._norm_bs,
                        constants=self.tube_constants)
                    if mbd.tube_mode == "cost-sens":
                        L = cost_sensitivity_torch(
                            bs,
                            lambda b: self.task.trajectory_cost_base_torch(
                                b, U_t, goal, u_prev=u_prev))
                        costs = costs + mbd.beta_e * (L * tubes).sum(dim=1)
                    else:
                        costs = costs + mbd.beta_e * tubes.sum(dim=1)
                return costs.cpu().numpy()

        return evaluate

    def _predict_ee(self, x, U):
        with torch.no_grad():
            U_t = torch.as_tensor(U[None], dtype=torch.float32, device=self.device)
            _, bs = self._rollout(x, U_t)
            return bs[0, :, NUM_JOINTS:NUM_JOINTS + 3].cpu().numpy()

    # ---------------------------------------------------------------- public
    def plan(self, x, goal, u_prev, U_warm, err, rng):
        """Return (U (T,7), ee_pred) for the given state / goal / warm start."""

        evaluate = self._make_evaluate(x, goal, u_prev)
        result = self._adaptive.optimize(U_warm, evaluate, rng, err)
        U = result.controls
        return U, self._predict_ee(x, U)

    def warmup(self, x0, goal, n_plans: int) -> None:
        """Throwaway plans (torch / threadpool warm-up) excluded from stats."""

        if n_plans <= 0:
            return
        warm_rng = np.random.default_rng(10**6 + self.cfg.mbd.seed)
        warm_eval = self._make_evaluate(x0, goal, np.zeros(NUM_JOINTS))
        U0 = np.zeros((self.horizon, NUM_JOINTS), dtype=np.float64)
        for _ in range(n_plans):
            self._optimizer.optimize(U0, warm_eval, rng=warm_rng)
