"""GPU oracle: the true dynamics batched through MuJoCo XLA (MJX).

The CPU oracle of :mod:`exper.backends` threads ``mujoco.rollout`` over a pool
of ``MjData``. This backend rolls the same model on the GPU instead, so the
comparison it supports is hardware against rollout model: the optimizer, the
cost, the goals and the planner stream are untouched and only the machine the
physics runs on changes.

The count of Sec. III-B is what the panel measures. One control step issues
``S * T * nsub`` physics substeps that must run in order, and a GPU parallelizes
the ``N`` candidates across the batch axis rather than that chain. The deadline
therefore rests on the time of one batched substep, not on the total substep
count.

Kept in its own module because MJX pulls in JAX, which the CPU experiments do
not need.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from .backends import trajectory_cost
from .config import PlannerCfg
from .plant import FrankaPlant

Array = np.ndarray


class MjxOracleBackend:
    """Batched true-dynamics rollout on the GPU.

    Args:
        plant: the same plant the CPU conditions use; its MuJoCo model is the
            one placed on the device.
        cfg: planner settings, for the cost weights.
        num_candidates: batch width to compile for. A stage that draws a
            different count triggers a recompile, so the runner keeps it fixed.
        horizon: rollout length to compile for.
        disable_contact: skip the collision pipeline. Free on this task: no
            contact forms anywhere in the operating region, and disabling it on
            the CPU leaves the tool path bit-identical (0 um over 64 probe
            rollouts). The seven constraints the solver does see are the
            friction-loss rows of the seven joints.
        solver_iterations, ls_iterations: Newton and line-search bounds. MuJoCo
            on the CPU exits as soon as it converges, which takes 1 to 3
            iterations here, while MJX cannot branch inside jit and runs the
            bound every substep. The model asks for 100 and 50, so leaving them
            charges the GPU for 30 to 100 times the solver work the CPU does.
            Lowering them to 8 moves the tool path by 4.5 um, far below the
            2.1 mm at which MJX and MuJoCo already disagree.
    """

    name = "oracle_gpu"

    def __init__(self, plant: FrankaPlant, cfg: PlannerCfg,
                 num_candidates: int, horizon: int,
                 disable_contact: bool = False,
                 solver_iterations: int | None = None,
                 ls_iterations: int | None = None,
                 tracked_only: bool = True) -> None:
        import jax
        import jax.numpy as jp
        import mujoco
        from mujoco import mjx

        self.plant = plant
        self.cfg = cfg
        self.num_candidates = num_candidates
        self.horizon = horizon
        self._jp = jp

        task = plant.task
        self.nsub = task._nsub
        site = task._ee_site
        nj = plant.num_joints
        limit = float(plant.limit)

        mx = mjx.put_model(task.model)
        if disable_contact:
            flags = int(mx.opt.disableflags) | int(
                mujoco.mjtDisableBit.mjDSBL_CONTACT)
            mx = mx.replace(opt=mx.opt.replace(disableflags=flags))
        if solver_iterations is not None:
            mx = mx.replace(opt=mx.opt.replace(iterations=solver_iterations))
        if ls_iterations is not None:
            mx = mx.replace(opt=mx.opt.replace(ls_iterations=ls_iterations))
        self.disable_contact = disable_contact
        self.solver_iterations = int(mx.opt.iterations)
        self.ls_iterations = int(mx.opt.ls_iterations)
        self.tracked_only = tracked_only
        # float32 unless the caller turned on x64 before importing anything
        self.dtype = jp.float64 if jax.config.jax_enable_x64 else jp.float32
        template = mjx.make_data(mx)

        def rollout_one(qpos0, qvel0, controls):
            """One candidate: (T, m) commands -> (T, nq) and (T, 3)."""
            d = template.replace(qpos=qpos0, qvel=qvel0)

            def control_step(d, u):
                d = d.replace(ctrl=jp.clip(u, -limit, limit))
                d, _ = jax.lax.scan(
                    lambda dd, _: (mjx.step(mx, dd), None), d, None,
                    length=self.nsub)
                # the observation is read at the control boundary, as on the CPU;
                # the cost reads the tool alone, so the joint block need not be
                # carried out of the scan unless a caller asks for it
                out = (d.site_xpos[site],) if tracked_only else (
                    d.qpos[:nj], d.site_xpos[site])
                return d, out

            _, out = jax.lax.scan(control_step, d, controls)
            return out

        self._rollout = jax.jit(jax.vmap(rollout_one, in_axes=(None, None, 0)))
        self._jax = jax

    # ------------------------------------------------------------------ pieces
    def rollout(self, state: Array, controls: Array):
        """Decoded joint and tool paths for a batch, (K, T, 7) and (K, T, 3)."""
        jp = self._jp
        s = np.asarray(state, dtype=np.float64)
        nj = self.plant.num_joints
        out = self._rollout(jp.asarray(s[:nj], dtype=self.dtype),
                            jp.asarray(s[nj:], dtype=self.dtype),
                            jp.asarray(controls, dtype=self.dtype))
        if self.tracked_only:
            return None, np.asarray(out[0])
        return np.asarray(out[0]), np.asarray(out[1])

    def warmup(self) -> float:
        """Compile the rollout and return the seconds it took.

        Timing a control step is meaningless until the kernel is built, so the
        runner calls this once before it measures anything.
        """
        import time

        u = np.zeros((self.num_candidates, self.horizon, self.plant.act_dim))
        t0 = time.perf_counter()
        _, ees = self.rollout(self.plant.reset(), u)
        self._jax.block_until_ready(ees)
        return time.perf_counter() - t0

    def cost_fn(self, state: Array, goal: Array) -> Callable[[Array], Array]:
        def evaluate(candidates: Array) -> Array:
            _, ees = self.rollout(state, candidates)
            return trajectory_cost(ees, candidates, goal, self.cfg)
        return evaluate


def agreement(plant: FrankaPlant, backend: MjxOracleBackend, controls: Array,
              state: Array | None = None) -> dict:
    """How far the GPU rollout drifts from the CPU one on the same commands.

    MJX runs the model in single precision, so the two paths are not bit-equal.
    A baseline is only a baseline if they are the same plant, and this reports
    the gap in the quantity the cost reads.
    """
    state = plant.reset() if state is None else state
    cpu = plant.rollout_true(state, controls)[..., plant.num_joints:]
    _, gpu = backend.rollout(state, controls)
    err = np.linalg.norm(cpu - gpu, axis=-1)
    return {
        "tcp_gap_median_m": float(np.median(err)),
        "tcp_gap_max_m": float(err.max()),
        "tcp_gap_terminal_median_m": float(np.median(err[:, -1])),
        "path_length_m": float(np.median(
            np.linalg.norm(np.diff(cpu, axis=1), axis=-1).sum(axis=1))),
    }
