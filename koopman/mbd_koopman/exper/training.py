"""Multi-step identification of the rollout model.

The planner rolls the model over the whole horizon, so the fit is made to hold
over ``H`` steps rather than one. The loss carries two terms: the prediction
error of the decoded output, and a latent consistency term that holds the
propagated lifted state to the image of the encoder. A stop-gradient on the
latent target keeps the encoder from lowering the loss by moving the target.

Checkpoints are cached per (variant, seed, tag) so a sweep pays for training
once and every later run loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn

from .config import Config
from .koopman import build_model
from .plant import FrankaPlant, Rollouts

CKPT_DIR = Path(__file__).resolve().parent / "checkpoints"


@dataclass
class TrainReport:
    variant: str
    seed: int
    epochs: int
    final_loss: float
    val_horizon_error: float


# --------------------------------------------------------------------- helpers
def _windows(obs: np.ndarray, controls: np.ndarray, horizon: int
             ) -> Tuple[torch.Tensor, torch.Tensor]:
    """Snippets to (b0, b_1..H) windows of the requested length."""
    if horizon > controls.shape[1]:
        raise SystemExit(f"train horizon {horizon} exceeds snippet length "
                         f"{controls.shape[1]}")
    b = torch.as_tensor(obs[:, : horizon + 1], dtype=torch.float32)
    u = torch.as_tensor(controls[:, :horizon], dtype=torch.float32)
    return b, u


def checkpoint_path(cfg: Config, variant: str, seed: int, tag: str = "") -> Path:
    plant = "kin" if cfg.plant.kinematic else "phys"
    stem = f"{plant}_{variant}_r{cfg.model.lift_dim}_seed{seed}"
    if tag:
        stem += f"_{tag}"
    elif cfg.train.one_step:
        stem += "_onestep"
    elif cfg.data.white:
        stem += "_white"
    return CKPT_DIR / f"{stem}.pt"


# -------------------------------------------------------------------- training
def train(cfg: Config, plant: FrankaPlant, data: Rollouts, variant: str,
          seed: int, verbose: bool = True) -> Tuple[nn.Module, TrainReport]:
    """Fit one rollout model to one excitation set."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = build_model(variant, plant.obs_dim, plant.act_dim, plant.num_joints,
                        cfg.model.lift_dim, cfg.model.hidden, cfg.model.layers)
    horizon = 1 if cfg.train.one_step else cfg.train.horizon
    b, u = _windows(data.obs, data.controls, horizon)

    n_val = max(1, int(0.1 * len(b)))
    b_val, u_val = b[:n_val], u[:n_val]
    b_tr, u_tr = b[n_val:], u[n_val:]

    opt = torch.optim.Adam(model.parameters(), cfg.train.lr,
                           weight_decay=cfg.train.weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.train.epochs)
    gamma = cfg.train.latent_weight
    n = len(b_tr)
    loss_value = float("nan")

    for epoch in range(cfg.train.epochs):
        perm = torch.randperm(n)
        for i in range(0, n, cfg.train.batch_size):
            idx = perm[i : i + cfg.train.batch_size]
            bb, uu = b_tr[idx], u_tr[idx]
            loss = torch.zeros((), dtype=torch.float32)
            for h, (dec, lat) in enumerate(model.unroll_train(bb[:, 0], uu), start=1):
                loss = loss + ((dec - bb[:, h]) ** 2).mean()
                if gamma:
                    with torch.no_grad():
                        target = model.lift(bb[:, h])
                    loss = loss + gamma * ((lat - target) ** 2).mean()
            loss = loss / uu.shape[1]
            opt.zero_grad()
            loss.backward()
            if cfg.train.grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            opt.step()
            loss_value = float(loss.detach())
        sched.step()
        if verbose and (epoch + 1) % max(1, cfg.train.epochs // 6) == 0:
            print(f"    epoch {epoch + 1}/{cfg.train.epochs}  loss {loss_value:.5f}",
                  flush=True)

    model.eval()
    b_full, u_full = _windows(data.obs, data.controls, cfg.train.horizon)
    val = horizon_error(model, b_full[:n_val], u_full[:n_val], plant.num_joints)
    return model, TrainReport(variant, seed, cfg.train.epochs, loss_value, val)


@torch.no_grad()
def horizon_error(model: nn.Module, b: torch.Tensor, u: torch.Tensor,
                  num_joints: int) -> float:
    """Held-out tool-center-point error at the end of the rolled window."""
    pred = model.rollout(b[:, 0], u)
    err = pred[:, -1, num_joints:] - b[:, -1, num_joints:]
    return float(err.norm(dim=-1).mean())


def get_model(cfg: Config, plant: FrankaPlant, variant: str, seed: int,
              tag: str = "", verbose: bool = True) -> nn.Module:
    """Load a cached checkpoint or train and cache one."""
    path = checkpoint_path(cfg, variant, seed, tag)
    model = build_model(variant, plant.obs_dim, plant.act_dim, plant.num_joints,
                        cfg.model.lift_dim, cfg.model.hidden, cfg.model.layers)
    if path.exists():
        model.load_state_dict(torch.load(path, map_location="cpu"))
        model.eval()
        if verbose:
            print(f"  loaded {path.name}", flush=True)
        return model

    if verbose:
        print(f"  training {variant} seed {seed} ...", flush=True)
    data = plant.excite(seed=100 + seed, white=cfg.data.white)
    model, report = train(cfg, plant, data, variant, seed, verbose=verbose)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)
    if verbose:
        print(f"  saved {path.name}  held-out horizon error "
              f"{report.val_horizon_error:.4f} m", flush=True)
    return model
