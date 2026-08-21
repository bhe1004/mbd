"""Batched MuJoCo rollouts -- the data source for Koopman training.

Training data is whatever recording of the plant you can get. In simulation it
is produced here: many short snippets of *coherent* random joint-velocity
commands (a random constant drift per snippet plus per-step jitter), rolled
through the same velocity-servo model the planner will later be deployed
against, with the tool position read from the model's own sensor.

Coherence matters. Independent per-step noise averages to nothing over a
horizon, so the arm barely moves and the model never sees the sustained motions
a planner actually commands; the drift term is what puts real displacement in
the dataset.

The output is plain arrays -- ``features (N, H+1, 10)``, ``controls (N, H, 7)``
-- so a dataset logged on the real arm trains the same model with the same
code. Nothing downstream of :mod:`pipeline.collect` knows a simulator existed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import mujoco
import numpy as np
from mujoco import rollout as mj_rollout

from .robot import NUM_JOINTS, ROBOT_XML, FrankaRobot


@dataclass(frozen=True)
class CollectConfig:
    """The ``collect`` block of the config file.

    Attributes:
        num_snippets: how many independent short trajectories to record.
        snippet_horizon: control steps per snippet (>= the training horizon).
        joint_margin: [rad] kept away from the joint stops when drawing starts.
        coherent_frac: per-snippet constant drift, as a fraction of the limit.
        jitter_frac: per-step noise, as a fraction of the limit.
        chunk_size: snippets per batched MuJoCo call (memory / speed knob).
        seed: dataset RNG seed.
        num_threads: MjData copies for the threaded rollout.
    """

    num_snippets: int = 6000
    snippet_horizon: int = 15
    joint_margin: float = 0.15
    coherent_frac: float = 0.6
    jitter_frac: float = 0.5
    chunk_size: int = 1000
    seed: int = 1
    num_threads: int = 16


class BatchSimulator:
    """Rolls many control sequences through the FR3 velocity-servo model at once."""

    def __init__(self, robot: FrankaRobot, control_dt: float, action_limit: float,
                 num_threads: int = 16, xml_path: Path | str = ROBOT_XML) -> None:
        self.robot = robot
        self.control_dt = float(control_dt)
        self.action_limit = float(action_limit)
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self._datas = [mujoco.MjData(self.model) for _ in range(max(1, num_threads))]
        self.substeps = int(round(self.control_dt / self.model.opt.timestep))
        self._nstate = mujoco.mj_stateSize(self.model, mujoco.mjtState.mjSTATE_FULLPHYSICS)

    def _full_state(self, q: np.ndarray, qd: np.ndarray) -> np.ndarray:
        """FULLPHYSICS vector [time, qpos, qvel] expected by ``mujoco.rollout``."""

        full = np.zeros(self._nstate, dtype=np.float64)
        full[1 : 1 + NUM_JOINTS] = np.asarray(q, dtype=np.float64)
        full[1 + NUM_JOINTS : 1 + 2 * NUM_JOINTS] = np.asarray(qd, dtype=np.float64)
        return full

    def rollout(self, q0: np.ndarray, qd0: np.ndarray, controls: np.ndarray
                ) -> Tuple[np.ndarray, np.ndarray]:
        """Roll K sequences and sample at control boundaries.

        Args:
            q0: (K, 7) start joints, qd0: (K, 7) start velocities.
            controls: (K, T, 7) joint-velocity commands, one per control period.

        Returns:
            (joints (K, T, 7), tool (K, T, 3)) at the end of each control period.
        """

        controls = np.asarray(controls, dtype=np.float64)
        ctrl_sub = np.repeat(controls, self.substeps, axis=1)
        init = np.stack([self._full_state(q, qd) for q, qd in zip(q0, qd0)], axis=0)
        state, sensordata = mj_rollout.rollout(self.model, self._datas, init, ctrl_sub)
        boundary = slice(self.substeps - 1, None, self.substeps)
        joints = np.asarray(state)[:, boundary, 1 : 1 + NUM_JOINTS]
        tool = np.asarray(sensordata)[:, boundary, :3]
        return joints, tool

    # ------------------------------------------------------------- collection
    def sample_dataset(self, config: CollectConfig, progress: bool = True
                       ) -> Dict[str, np.ndarray]:
        """Record coherent random-velocity snippets across the whole workspace."""

        rng = np.random.default_rng(config.seed)
        n, horizon = int(config.num_snippets), int(config.snippet_horizon)
        limit = self.action_limit

        q_low = self.robot.joint_low + config.joint_margin
        q_high = self.robot.joint_high - config.joint_margin
        q0 = rng.uniform(q_low, q_high, (n, NUM_JOINTS))
        drift = rng.uniform(-limit * config.coherent_frac,
                            limit * config.coherent_frac, (n, 1, NUM_JOINTS))
        jitter = rng.uniform(-limit * config.jitter_frac,
                             limit * config.jitter_frac, (n, horizon, NUM_JOINTS))
        controls = np.clip(drift + jitter, -limit, limit)

        joints = np.zeros((n, horizon + 1, NUM_JOINTS), dtype=np.float64)
        tool = np.zeros((n, horizon + 1, 3), dtype=np.float64)
        joints[:, 0] = q0
        for i in range(n):
            tool[i, 0] = self.robot.ee_of_q(q0[i])

        zeros = np.zeros((config.chunk_size, NUM_JOINTS))
        for start in range(0, n, config.chunk_size):
            end = min(start + config.chunk_size, n)
            j, t = self.rollout(q0[start:end], zeros[: end - start], controls[start:end])
            joints[start:end, 1:] = j
            tool[start:end, 1:] = t
            if progress:
                print(f"  collected {end}/{n} snippets", end="\r", flush=True)
        if progress:
            print()

        return {"features": np.concatenate([joints, tool], axis=-1),
                "controls": controls}
