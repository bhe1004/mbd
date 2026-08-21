"""Multi-step training for the bilinear Koopman model.

Loss over a window of ``H`` control steps, rolled entirely in the lifted space
from a single initial lift::

    L = (1/H) sum_h [ ||C zhat_h - x_h||^2
                      + gamma ||zhat_h - Psi(x_h).detach()||^2 ]

The first term is the decoded prediction error; the second keeps the rolled
latent on the encoder's own manifold (detached, so the encoder is not trained
to chase the rollout).

Training consumes plain arrays -- ``features (N, T+1, d)`` and
``controls (N, T, m)`` -- so a dataset recorded on a real robot trains the same
model with the same code as the simulated one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - progress bar is optional
    tqdm = None

from .koopman import BilinearKoopman, KoopmanArchitecture
from .types import Array


@dataclass(frozen=True)
class TrainSettings:
    """The ``train`` block of the config file."""

    rollout_horizon: int = 15
    gamma_latent: float = 0.1
    batch_size: int = 512
    num_epochs: int = 300
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 2.0
    val_fraction: float = 0.2
    cosine_lr: bool = True
    eval_every: int = 10
    seed: int = 0
    device: str = "cpu"


@dataclass
class TrainResult:
    model: BilinearKoopman
    train_loss: float
    val_error: float
    open_loop: Dict[str, Array] = field(default_factory=dict)
    history: Dict[str, Array] = field(default_factory=dict)


def build_windows(features: Array, controls: Array, horizon: int) -> Tuple[Array, Array]:
    """Slide fixed-length windows over trajectories.

    Args:
        features: (num_traj, T + 1, feature_dim).
        controls: (num_traj, T, action_dim).

    Returns:
        (x_windows (N, H + 1, d), u_windows (N, H, m)).
    """

    x = np.asarray(features)
    u = np.asarray(controls)
    if x.ndim != 3 or u.ndim != 3:
        raise ValueError("features and controls must be batched trajectories")
    if x.shape[0] != u.shape[0]:
        raise ValueError("features and controls must have the same trajectory count")
    if x.shape[1] != u.shape[1] + 1:
        raise ValueError("features must have one more time step than controls")
    if not 0 < horizon <= u.shape[1]:
        raise ValueError(f"horizon must be in (0, {u.shape[1]}]")

    starts = np.arange(u.shape[1] - horizon + 1)
    xs = np.stack([x[:, s : s + horizon + 1] for s in starts], axis=1)
    us = np.stack([u[:, s : s + horizon] for s in starts], axis=1)
    return (xs.reshape(-1, horizon + 1, x.shape[-1]),
            us.reshape(-1, horizon, u.shape[-1]))


def split_train_val(num_windows: int, val_fraction: float, seed: int) -> Tuple[Array, Array]:
    """Deterministic index split (same seed -> same split)."""

    perm = np.random.default_rng(seed).permutation(num_windows)
    num_val = int(round(val_fraction * num_windows))
    return perm[num_val:], perm[:num_val]


def multistep_loss(model: BilinearKoopman, x: torch.Tensor, u: torch.Tensor,
                   gamma_latent: float) -> torch.Tensor:
    horizon = u.shape[1]
    z = model.lift(x[:, 0])
    loss = x.new_zeros(())
    for k in range(horizon):
        z = model.step(z, u[:, k])
        loss = loss + ((model.decode(z) - x[:, k + 1]) ** 2).mean()
        if gamma_latent > 0.0:
            loss = loss + gamma_latent * ((z - model.lift(x[:, k + 1]).detach()) ** 2).mean()
    return loss / horizon


@torch.no_grad()
def validation_error(model: BilinearKoopman, x: torch.Tensor, u: torch.Tensor,
                     batch_size: int = 2048) -> float:
    """Decoded multi-step MSE (no latent term) over a validation set."""

    total, count = 0.0, 0
    for i in range(0, x.shape[0], batch_size):
        xb, ub = x[i : i + batch_size], u[i : i + batch_size]
        zs = model.rollout(model.lift(xb[:, 0]), ub)
        total += float(((model.decode(zs[:, 1:]) - xb[:, 1:]) ** 2).mean()) * xb.shape[0]
        count += xb.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def open_loop_error(model: BilinearKoopman, x: torch.Tensor, u: torch.Tensor,
                    batch_size: int = 2048) -> Dict[str, Array]:
    """Per-step open-loop decoded RMSE over full windows."""

    horizon = u.shape[1]
    sq_sum = np.zeros(horizon, dtype=np.float64)
    count = 0
    for i in range(0, x.shape[0], batch_size):
        xb, ub = x[i : i + batch_size], u[i : i + batch_size]
        zs = model.rollout(model.lift(xb[:, 0]), ub)
        sq = ((model.decode(zs[:, 1:]) - xb[:, 1:]) ** 2).mean(dim=-1)
        sq_sum += sq.sum(dim=0).cpu().numpy()
        count += xb.shape[0]
    rmse = np.sqrt(sq_sum / max(count, 1))
    return {"rmse_per_step": rmse,
            "rmse_mean": float(rmse.mean()),
            "rmse_final": float(rmse[-1])}


def train(arch: KoopmanArchitecture, features: Array, controls: Array,
          settings: TrainSettings, *, progress: bool = True,
          desc: str = "bk") -> TrainResult:
    """Train one bilinear Koopman model on a recorded dataset."""

    x_windows, u_windows = build_windows(features, controls, settings.rollout_horizon)
    train_idx, val_idx = split_train_val(
        x_windows.shape[0], settings.val_fraction, settings.seed)

    device = torch.device(settings.device)
    x_all = torch.as_tensor(x_windows, dtype=torch.float32, device=device)
    u_all = torch.as_tensor(u_windows, dtype=torch.float32, device=device)
    x_train, u_train = x_all[train_idx], u_all[train_idx]
    x_val, u_val = x_all[val_idx], u_all[val_idx]

    torch.manual_seed(settings.seed)
    model = BilinearKoopman(arch).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=settings.learning_rate,
                                 weight_decay=settings.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, settings.num_epochs)
                 if settings.cosine_lr else None)
    shuffle = torch.Generator(device="cpu").manual_seed(settings.seed)

    train_loss_history, val_history, val_epochs = [], [], []
    last_loss = float("nan")
    epochs = range(settings.num_epochs)
    bar = tqdm(epochs, desc=desc, unit="epoch", ncols=110) if (progress and tqdm) else None

    for epoch in (bar or epochs):
        model.train()
        perm = torch.randperm(x_train.shape[0], generator=shuffle)
        epoch_loss, num_batches = 0.0, 0
        for i in range(0, x_train.shape[0], settings.batch_size):
            idx = perm[i : i + settings.batch_size]
            loss = multistep_loss(model, x_train[idx], u_train[idx], settings.gamma_latent)
            optimizer.zero_grad()
            loss.backward()
            if settings.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip)
            optimizer.step()
            epoch_loss += float(loss)
            num_batches += 1
        if scheduler is not None:
            scheduler.step()
        last_loss = epoch_loss / max(num_batches, 1)
        train_loss_history.append(last_loss)

        is_last = epoch == settings.num_epochs - 1
        if x_val.shape[0] and (epoch % settings.eval_every == 0 or is_last):
            model.eval()
            val_history.append(validation_error(model, x_val, u_val))
            val_epochs.append(epoch)
        if bar is not None:
            post = {"loss": f"{last_loss:.5f}"}
            if val_history:
                post["val_mse"] = f"{val_history[-1]:.5f}"
            bar.set_postfix(post)
    if bar is not None:
        bar.close()

    model.eval()
    ol = open_loop_error(model, x_val, u_val) if x_val.shape[0] else {}
    return TrainResult(
        model=model,
        train_loss=last_loss,
        val_error=val_history[-1] if val_history else float("nan"),
        open_loop=ol,
        history={"train_loss": np.asarray(train_loss_history),
                 "val_error": np.asarray(val_history),
                 "val_epochs": np.asarray(val_epochs)},
    )


# ------------------------------------------------------------------ checkpoints
def save_checkpoint(path: Path | str, model: BilinearKoopman,
                    *, extras: Dict[str, object] | None = None) -> None:
    """Store weights together with everything needed to rebuild the model."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": "mbd_main.bilinear_koopman.v1",
                "architecture": model.arch.as_dict(),
                "state_dict": model.state_dict(),
                "extras": extras or {}}, path)


def load_checkpoint(path: Path | str, *, device: torch.device | str = "cpu"
                    ) -> Tuple[BilinearKoopman, Dict[str, object]]:
    """Rebuild a model from a checkpoint.

    Also accepts checkpoints written by the older ``bk_mbd.train`` code (they
    store ``model_config`` / ``model_kind`` instead of ``architecture``), so an
    already-trained bilinear model can be dropped in without retraining.
    """

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    if "architecture" in payload:
        arch = KoopmanArchitecture(**payload["architecture"])
    else:  # legacy layout
        mc = payload["model_config"]
        if not payload.get("bilinear", False):
            raise ValueError(f"{path} holds a linear model; BK-MBD needs a bilinear one")
        arch = KoopmanArchitecture(
            feature_dim=int(mc["state_dim"]),
            action_dim=int(mc["action_dim"]),
            lift_extra=int(mc["lift_dim"]) - int(mc["state_dim"]),
            hidden_width=int(mc["hidden_width"]),
            hidden_depth=int(mc["hidden_depth"]),
            encoder_input=("sincos_prefix" if payload.get("model_kind") == "arm_sincos"
                           else "identity"),
            angle_dim=7,
        )
    model = BilinearKoopman(arch)
    model.load_state_dict(payload["state_dict"])
    model.to(device).eval()
    return model, payload.get("extras", {})
