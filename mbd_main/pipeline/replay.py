"""Replay a saved trajectory in the viewer.

A replay is a recording, not a re-run: the joint angles are played back exactly
as they were executed, so what you see is what happened -- including any
collision the referee flagged. Nothing is re-planned and no physics is stepped,
which is why a replay is reproducible while a live run never quite is.
"""

from __future__ import annotations

import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np

from config import Config
from env.robot import SCENE_XML
from pipeline import viewer as vz
from pipeline.build import build_environment


def run(cfg: Config, path: Path | str, speed: float = 1.0, loop: bool = False) -> None:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"replay file not found: {path}")
    data = np.load(path, allow_pickle=False)
    qpos = data["qpos"]
    goals = data["goals"]
    ee = data["ee"]
    period = float(data["period"]) / max(speed, 1e-6)

    env = build_environment(cfg)
    print(f"replay: {path}")
    print(f"  {qpos.shape[0]} frames at {float(data['period']) * 1000:.0f} ms "
          f"({qpos.shape[0] * float(data['period']):.1f} s), "
          f"method={str(data['method'])}, mode={str(data['mode'])}, speed={speed}x")
    if "violations" in data and data["violations"].size:
        bad = int((data["violations"] > 0).sum())
        print(f"  referee: {bad}/{data['violations'].size} boundaries in collision")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    sim = mujoco.MjData(model)
    with mujoco.viewer.launch_passive(model, sim) as viewer:
        scn = viewer.user_scn
        while viewer.is_running():
            for i in range(qpos.shape[0]):
                if not viewer.is_running():
                    break
                frame = time.perf_counter()
                sim.qpos[:qpos.shape[1]] = qpos[i]
                sim.qvel[:] = 0.0
                mujoco.mj_forward(model, sim)
                vz.draw_overlay(scn, goal=goals[i], threshold=cfg.task.strict_threshold,
                                trail=ee[: i + 1], prediction=None,
                                obstacles=env.obstacles,
                                show_trail=cfg.viewer.trail, show_prediction=False)
                viewer.sync()
                slack = period - (time.perf_counter() - frame)
                if slack > 0:
                    time.sleep(slack)
            if not loop:
                break
    print("replay finished")
