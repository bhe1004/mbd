"""Stage 1 -- record the dataset the Koopman model is trained on.

Writes a single ``.npz`` holding

    features  (N, H + 1, 10)   the [q, ee] trajectory of each snippet
    controls  (N, H, 7)        the joint-velocity command applied at each step

plus the metadata needed to tell later whether a checkpoint and a dataset
belong together (control period, velocity limit, how the commands were drawn).

That file is the whole interface to stage 2. A dataset logged on the real arm
trains the identical model as long as it uses the same two array names, the
same units, and the same control period -- which is the point of keeping
collection behind a file rather than behind a live simulator object.
"""

from __future__ import annotations

import time

import numpy as np

from config import Config
from pipeline.build import build_environment, build_simulator


def run(cfg: Config) -> None:
    env = build_environment(cfg)
    sim = build_simulator(cfg, env)
    c = cfg.collect

    print(f"collecting {c.num_snippets} snippets x {c.snippet_horizon} steps "
          f"({c.snippet_horizon * cfg.task.control_dt:.2f} s each) "
          f"at {cfg.task.control_dt * 1000:.0f} ms, |u| <= {cfg.task.action_limit} rad/s")
    print(f"  MuJoCo substeps per control period: {sim.substeps}, "
          f"threads: {c.num_threads}, seed: {c.seed}")

    start = time.perf_counter()
    data = sim.sample_dataset(c)
    elapsed = time.perf_counter() - start

    features, controls = data["features"], data["controls"]
    displacement = np.linalg.norm(features[:, -1, 7:] - features[:, 0, 7:], axis=-1)

    cfg.paths.dataset.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cfg.paths.dataset,
        features=features,
        controls=controls,
        control_dt=cfg.task.control_dt,
        action_limit=cfg.task.action_limit,
        snippet_horizon=c.snippet_horizon,
        coherent_frac=c.coherent_frac,
        jitter_frac=c.jitter_frac,
        joint_margin=c.joint_margin,
        seed=c.seed,
        feature_layout="q(7) + ee(3)",
        source=str(cfg.source),
    )
    size_mb = cfg.paths.dataset.stat().st_size / 1e6
    print(f"features {features.shape}, controls {controls.shape}")
    print(f"tool displacement per snippet: mean {displacement.mean() * 100:.1f} cm, "
          f"p95 {np.percentile(displacement, 95) * 100:.1f} cm "
          f"(near zero would mean the commands cancel out -- raise "
          f"collect.coherent_frac)")
    print(f"saved: {cfg.paths.dataset}  ({size_mb:.1f} MB, {elapsed:.1f} s)")
