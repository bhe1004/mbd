"""Score the trained model against a trajectory the robot actually executed.

`train` reports open-loop error on data drawn the same way the training set was.
That says nothing about whether the model still holds on the machine it is
deployed to. This stage closes that gap: it replays a recorded run through the
model and reports the same metric, so the two numbers are directly comparable.

A replay stores exactly what is needed -- ``qpos[k+1]`` is the measured
configuration after ``controls[k]`` was applied for one control period, which is
the training format. Any recording in that shape works, from the simulator or
from hardware.

Read it as: if the deployment RMSE is close to the training RMSE, the model
transferred and retraining buys little. If it is several times worse, the plant
the model learned is not the plant it is driving, and the fix is data from the
real one.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from config import Config
from mbd.training import build_windows, load_checkpoint
from pipeline.build import build_environment


def run(cfg: Config, path: Path | str) -> None:
    path = Path(path)
    if not path.exists():
        raise SystemExit(f"recording not found: {path}")
    data = np.load(path, allow_pickle=False)
    for key in ("qpos", "controls"):
        if key not in data:
            raise SystemExit(f"{path} has no '{key}' array")

    qpos, controls = data["qpos"], data["controls"]
    n = min(qpos.shape[0] - 1, controls.shape[0])
    if n < cfg.train.rollout_horizon + 1:
        raise SystemExit(f"{path} is too short: {n} steps, need at least "
                         f"{cfg.train.rollout_horizon + 1}")
    qpos, controls = qpos[: n + 1], controls[:n]

    env = build_environment(cfg)
    features = np.stack([env.task.features(q) for q in qpos])[None]   # (1, n+1, 10)
    model, extras = load_checkpoint(cfg.paths.checkpoint)

    horizon = cfg.train.rollout_horizon
    x, u = build_windows(features, controls[None], horizon)
    xt = torch.as_tensor(x, dtype=torch.float32)
    ut = torch.as_tensor(u, dtype=torch.float32)
    with torch.no_grad():
        zs = model.rollout(model.lift(xt[:, 0]), ut)
        pred = model.decode(zs[:, 1:])
    err = (pred - xt[:, 1:]).numpy()

    joints = np.sqrt((err[..., :7] ** 2).mean(axis=(0, 2)))    # per step, rad
    tool = np.linalg.norm(err[..., 7:10], axis=-1).mean(axis=0)  # per step, m
    overall = float(np.sqrt((err ** 2).mean()))

    print(f"recording: {path.name}  ({n} steps, {n * cfg.task.control_dt:.1f} s, "
          f"{x.shape[0]} windows of {horizon})")
    print(f"model: {cfg.paths.checkpoint.name}")
    print()
    print(f"open-loop RMSE over {horizon} steps (all features): {overall:.5f}")
    print(f"  joint RMSE   [rad]  step 1 {joints[0]:.5f}   mid {joints[horizon // 2]:.5f}"
          f"   step {horizon} {joints[-1]:.5f}")
    print(f"  tool error   [mm]   step 1 {tool[0] * 1000:6.1f}   "
          f"mid {tool[horizon // 2] * 1000:6.1f}   step {horizon} {tool[-1] * 1000:6.1f}")

    trained = extras.get("open_loop_rmse_mean")
    trained_final = extras.get("open_loop_rmse_final")
    if trained is not None:
        print()
        print(f"training-set open-loop RMSE: mean {trained:.5f}, "
              f"final-step {trained_final:.5f}")
        ratio = overall / max(float(trained), 1e-12)
        print(f"deployment / training ratio: {ratio:.2f}x")
        if ratio < 1.5:
            print("  -> the model transferred; retraining on this plant would buy little")
        elif ratio < 4.0:
            print("  -> noticeably worse than training. Worth retraining on data from "
                  "this plant, though the closed loop may still be fine")
        else:
            print("  -> the model is predicting a different plant. Retrain on data "
                  "recorded here")
    print()
    print("note: a deployment recording covers only the states the planner visited, "
          "so this is a check on the operating region, not a replacement for a "
          "workspace-wide validation set")
