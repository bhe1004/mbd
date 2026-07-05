"""Shared receding-horizon MBD closed-loop logic for all tasks.

Tasks mix in `MBDClosedLoopMixin` and provide:

- `task_name`, `state_dim`, `action_dim`, `action_bounds`, `config`
  (with `horizon` and `closed_loop_steps`),
- `case(case_id) -> (start_state, goal, case_name)`,
- `true_step(x, u)` executed on the real system,
- `success / strict_success / final_error(x, goal)`,
- `extra_final_metrics(x, goal) -> dict` (optional extras, e.g. angle error),
- `make_true_evaluate(x, goal, u_prev, device)` for the oracle backend,
- `state_to_base_torch(x, device)` and `trajectory_cost_base_torch(...)`
  for the Koopman backends.

This keeps the fairness rule structural: every method and every task runs
through the same loop, warm start, clipping, and early-stop logic.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from bk_mbd.types import Array


class MBDClosedLoopMixin:
    """Common MBD receding-horizon control loop."""

    def extra_final_metrics(self, x: Array, goal: Array) -> Dict[str, float]:
        return {}

    def _closed_loop_mbd(
        self,
        optimizer,
        make_evaluate,
        *,
        method: str,
        case_id: int,
        seed: int,
        plan_stats=None,
    ) -> Dict[str, Any]:
        """Receding-horizon MBD loop shared by all rollout backends."""

        x, goal, case_name = self.case(case_id)
        U = np.zeros((self.config.horizon, self.action_dim), dtype=np.float64)
        u_prev = np.zeros(self.action_dim, dtype=np.float64)
        trajectory = [np.asarray(x, dtype=np.float64).copy()]
        applied_controls = []
        best_costs = []
        step_logs: Dict[str, list] = {}
        rng = np.random.default_rng(seed)

        for _step in range(self.config.closed_loop_steps):
            evaluate = make_evaluate(x, goal, u_prev)
            result = optimizer.optimize(U, evaluate, rng=rng)
            U = result.controls
            if plan_stats is not None:
                for key, value in plan_stats(x, U, goal, u_prev).items():
                    step_logs.setdefault(key, []).append(float(value))
            u0 = np.clip(U[0], self.action_bounds[0], self.action_bounds[1])
            x = self.true_step(x, u0)
            trajectory.append(np.asarray(x, dtype=np.float64).copy())
            applied_controls.append(u0.copy())
            best_costs.append(float(result.best_cost))

            U = np.roll(U, -1, axis=0)
            U[-1] = U[-2]
            u_prev = u0

            if self.strict_success(x, goal):
                break

        traj = np.stack(trajectory, axis=0)
        ctrls = (
            np.stack(applied_controls, axis=0)
            if applied_controls
            else np.zeros((0, self.action_dim), dtype=np.float64)
        )
        out: Dict[str, Any] = {
            "task": self.task_name,
            "method": method,
            "seed": seed,
            "case_id": case_name,
            "success": self.success(traj[-1], goal),
            "strict_success": self.strict_success(traj[-1], goal),
            "final_error": self.final_error(traj[-1], goal),
            "steps": len(traj) - 1,
            "trajectory": traj,
            "controls": ctrls,
            "best_costs": np.asarray(best_costs, dtype=np.float64),
        }
        out.update(self.extra_final_metrics(traj[-1], goal))
        for key, values in step_logs.items():
            out[key] = np.asarray(values, dtype=np.float64)
        return out

    def closed_loop_true_mbd(
        self,
        optimizer,
        *,
        case_id: int = 0,
        seed: int = 0,
        device: torch.device | str | None = None,
    ) -> Dict[str, Any]:
        """Run vanilla MBD with true dynamics rollout on one case."""

        def make_evaluate(x: Array, goal: Array, u_prev: Array):
            return self.make_true_evaluate(x, goal, u_prev, device)

        return self._closed_loop_mbd(
            optimizer,
            make_evaluate,
            method="vanilla_mbd_true",
            case_id=case_id,
            seed=seed,
        )

    def closed_loop_koopman_mbd(
        self,
        optimizer,
        model,
        *,
        method: str,
        case_id: int = 0,
        seed: int = 0,
        device: torch.device | str | None = None,
        tube_constants=None,
        beta_e: float = 0.0,
    ) -> Dict[str, Any]:
        """Run MBD with a deep Koopman rollout model (DK or BK).

        Candidates are rolled entirely in lifted space from the lifted current
        state; the executed control is applied to the true dynamics. When
        `tube_constants` is given (BK), the tube penalty beta_e * sum_t e_t is
        added to each candidate cost and tube statistics are logged per step.
        """

        from bk_mbd.tube import bilinear_norm_bounds, propagate_tube_batch_torch

        model = model.to(device) if device is not None else model
        model.eval()
        use_tube = tube_constants is not None
        if use_tube:
            norm_a, norm_bs_np = bilinear_norm_bounds(model.bilinear_params())
            norm_bs = torch.as_tensor(norm_bs_np, dtype=torch.float32, device=device)

        def rollout_candidates(x: Array, candidates: torch.Tensor):
            b0 = self.state_to_base_torch(x, device)
            z0 = model.lift(b0).expand(candidates.shape[0], -1)
            zs = model.rollout(z0, candidates)
            return zs, model.decode(zs)

        def make_evaluate(x: Array, goal: Array, u_prev: Array):
            def evaluate(candidates: Array) -> Array:
                with torch.no_grad():
                    U_t = torch.as_tensor(candidates, dtype=torch.float32, device=device)
                    zs, bs = rollout_candidates(x, U_t)
                    costs_t = self.trajectory_cost_base_torch(
                        bs, U_t, goal, u_prev=u_prev
                    )
                    if use_tube:
                        tubes = propagate_tube_batch_torch(
                            zs,
                            U_t,
                            norm_a=norm_a,
                            norm_bs=norm_bs,
                            constants=tube_constants,
                        )
                        costs_t = costs_t + beta_e * tubes.sum(dim=1)
                    return costs_t.cpu().numpy()

            return evaluate

        plan_stats = None
        if use_tube:

            def plan_stats(x: Array, U: Array, goal: Array, u_prev: Array):
                with torch.no_grad():
                    U_t = torch.as_tensor(
                        U[None], dtype=torch.float32, device=device
                    )
                    zs, _ = rollout_candidates(x, U_t)
                    tubes = propagate_tube_batch_torch(
                        zs,
                        U_t,
                        norm_a=norm_a,
                        norm_bs=norm_bs,
                        constants=tube_constants,
                    )[0]
                return {
                    "tube_cost": beta_e * float(tubes.sum()),
                    "max_tube": float(tubes.max()),
                }

        return self._closed_loop_mbd(
            optimizer,
            make_evaluate,
            method=method,
            case_id=case_id,
            seed=seed,
            plan_stats=plan_stats,
        )
