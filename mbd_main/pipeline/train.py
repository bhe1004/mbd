"""Stage 2 -- train the bilinear Koopman model on the recorded dataset.

Reads the ``.npz`` from stage 1, trains with the multi-step lifted loss, and
writes a checkpoint that carries its own architecture, so stage 3 rebuilds the
model without being told anything about it.

The reported number that matters is the open-loop RMSE at the *end* of a
horizon-length window: that is the error the planner actually plans against,
since every candidate is rolled the full horizon in the lifted space with no
feedback.
"""

from __future__ import annotations

import json
import time

import numpy as np
import torch

from config import Config
from mbd.training import save_checkpoint, train as train_model


def run(cfg: Config) -> None:
    if not cfg.paths.dataset.exists():
        raise SystemExit(f"dataset not found: {cfg.paths.dataset}\n"
                         "record one first:  python main.py collect")
    if cfg.train.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"train.device is {cfg.train.device!r} but CUDA is unavailable")

    data = np.load(cfg.paths.dataset)
    features, controls = data["features"], data["controls"]
    if features.shape[-1] != cfg.koopman.feature_dim:
        raise SystemExit(f"dataset feature dim {features.shape[-1]} != "
                         f"koopman.feature_dim {cfg.koopman.feature_dim}")
    if controls.shape[-1] != cfg.koopman.action_dim:
        raise SystemExit(f"dataset action dim {controls.shape[-1]} != "
                         f"koopman.action_dim {cfg.koopman.action_dim}")
    if float(data["control_dt"]) != cfg.task.control_dt:
        raise SystemExit(
            f"dataset was recorded at control_dt={float(data['control_dt'])} s but "
            f"task.control_dt is {cfg.task.control_dt} s -- the model would learn the "
            "wrong step size; re-collect or fix the config")

    windows = features.shape[0] * (controls.shape[1] - cfg.train.rollout_horizon + 1)
    print(f"dataset: {cfg.paths.dataset}  features {features.shape}, "
          f"controls {controls.shape}")
    print(f"training windows: {windows} of {cfg.train.rollout_horizon} steps "
          f"({cfg.train.val_fraction:.0%} held out)")
    print(f"model: lift {cfg.koopman.feature_dim} + {cfg.koopman.lift_extra} = "
          f"{cfg.koopman.lift_dim}, encoder {cfg.koopman.encoder_input} "
          f"{cfg.koopman.hidden_depth}x{cfg.koopman.hidden_width}, bilinear terms "
          f"{cfg.koopman.action_dim}x{cfg.koopman.lift_dim}^2 (zero-initialized)")

    start = time.perf_counter()
    result = train_model(cfg.koopman, features, controls, cfg.train, desc="bk-koopman")
    elapsed = time.perf_counter() - start

    ol = result.open_loop
    extras = {
        "train_loss": result.train_loss,
        "val_error": result.val_error,
        "open_loop_rmse_mean": ol.get("rmse_mean", float("nan")),
        "open_loop_rmse_final": ol.get("rmse_final", float("nan")),
        "rollout_horizon": cfg.train.rollout_horizon,
        "dataset": str(cfg.paths.dataset),
        "control_dt": cfg.task.control_dt,
        "action_limit": cfg.task.action_limit,
        "train_time_s": elapsed,
        "config": str(cfg.source),
    }
    save_checkpoint(cfg.paths.checkpoint, result.model, extras=extras)

    summary = cfg.paths.checkpoint.with_suffix(".json")
    summary.write_text(json.dumps(
        {**extras, "open_loop_rmse_per_step": ol.get("rmse_per_step", np.zeros(0)).tolist()},
        indent=2))

    print(f"train_loss={result.train_loss:.6f}  val_mse={result.val_error:.6f}")
    if ol:
        print(f"open-loop RMSE over {cfg.train.rollout_horizon} steps: "
              f"mean {ol['rmse_mean']:.5f}  final {ol['rmse_final']:.5f} "
              "(metres/radians, mixed features)")
    print(f"saved: {cfg.paths.checkpoint}  ({elapsed:.1f} s)")
    print(f"       {summary}")
