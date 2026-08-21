"""Replay a trajectory saved by :mod:`secy.runtime`."""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from envs.franka import NUM_JOINTS

from secy import viewer as vz
from secy.config import Config
from secy.environment import Scene


def _load(path: Path, scene: Scene):
    if not path.exists():
        raise SystemExit(f"replay file not found: {path}")
    with np.load(path, allow_pickle=False) as data:
        if "qpos" in data:
            qpos = np.asarray(data["qpos"], dtype=np.float64)
            qvel = np.asarray(data["qvel"], dtype=np.float64)
        elif "states" in data:
            states = np.asarray(data["states"], dtype=np.float64)
            qpos = states[:, :NUM_JOINTS]
            qvel = states[:, NUM_JOINTS:2 * NUM_JOINTS]
        else:
            raise SystemExit(f"replay file has no qpos/states array: {path}")

        ee = (np.asarray(data["ee"], dtype=np.float64)
              if "ee" in data else None)
        goals = (np.asarray(data["goals"], dtype=np.float64)
                 if "goals" in data else None)
        period = (float(np.asarray(data["period"]).item())
                  if "period" in data else scene.period)

    if qpos.ndim != 2 or qpos.shape[1] != NUM_JOINTS:
        raise SystemExit(f"invalid replay qpos shape: {qpos.shape}")
    if qvel.shape != qpos.shape:
        raise SystemExit(f"invalid replay qvel shape: {qvel.shape}")
    if len(qpos) == 0:
        raise SystemExit("replay trajectory is empty")
    if ee is not None and ee.shape != (len(qpos), 3):
        raise SystemExit(f"invalid replay ee shape: {ee.shape}")
    if goals is None:
        goals = np.repeat(scene.goal_for(0)[None, :], len(qpos), axis=0)
    if goals.shape != (len(qpos), 3):
        raise SystemExit(f"invalid replay goals shape: {goals.shape}")
    return qpos, qvel, ee, goals, period


def replay_trajectory(scene: Scene, cfg: Config, path: Path) -> None:
    """Display a saved joint-state trajectory using the configured scene."""

    qpos, qvel, ee, goals, period = _load(path, scene)
    model = scene.scene_model
    sim = scene.sim
    color = (0.35, 0.55, 0.85)
    trail = []

    print(f"replay: {path} ({len(qpos) - 1} control steps)")
    if not cfg.runtime.viewer:
        print("replay: runtime.viewer=false; opening the viewer for playback")
    print("replay: close the viewer (ESC) to exit")
    with mujoco.viewer.launch_passive(model, sim) as viewer:
        for i in range(len(qpos)):
            if not viewer.is_running():
                break
            sim.qpos[:NUM_JOINTS] = qpos[i]
            sim.qvel[:NUM_JOINTS] = qvel[i]
            mujoco.mj_forward(model, sim)
            point = (ee[i] if ee is not None
                     else sim.site_xpos[scene.tcp].copy())
            trail.append(np.asarray(point, dtype=np.float64).copy())
            vz.draw_overlay(
                viewer,
                goals[i],
                scene.strict,
                trail,
                None,
                color,
                obstacle=scene.obstacle,
                show_trail=cfg.path_line.trail,
            )
            viewer.sync()
            if i + 1 < len(qpos):
                time.sleep(max(period, 0.0))
