"""Training helpers for deep Koopman models.

Implements the multi-step training loss from the Bilinear Koopman MPPI paper
(`koopman/mppi_koopman/verify/unicycle_multiseed_v2.py`):

```text
L = (1/H) sum_h [ || C zhat_h - x_h ||^2
                  + gamma || zhat_h - Psi_theta(x_h).detach() ||^2 ]
```

The same loop trains both linear DK and bilinear BK by switching the model.
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

from .config import KoopmanModelConfig, KoopmanTrainConfig
from .models import MODEL_KINDS, DeepKoopmanModel
from .types import Array


def build_multistep_windows(xs: Array, us: Array, horizon: int) -> tuple[Array, Array]:
    """Build aligned x/u windows for multi-step rollout training.

    Args:
        xs: State trajectories with shape (num_traj, T + 1, state_dim).
        us: Control trajectories with shape (num_traj, T, action_dim).
        horizon: Number of control steps per training window.

    Returns:
        x_windows: Shape (num_windows, horizon + 1, state_dim).
        u_windows: Shape (num_windows, horizon, action_dim).
    """

    xs_arr = np.asarray(xs)
    us_arr = np.asarray(us)
    if xs_arr.ndim != 3 or us_arr.ndim != 3:
        raise ValueError("xs and us must be batched trajectories")
    if xs_arr.shape[0] != us_arr.shape[0]:
        raise ValueError("xs and us must have the same number of trajectories")
    if xs_arr.shape[1] != us_arr.shape[1] + 1:
        raise ValueError("xs must have one more time step than us")
    if horizon <= 0 or horizon > us_arr.shape[1]:
        raise ValueError("invalid horizon")

    x_windows = []
    u_windows = []
    for traj_idx in range(xs_arr.shape[0]):
        for start in range(us_arr.shape[1] - horizon + 1):
            x_windows.append(xs_arr[traj_idx, start : start + horizon + 1])
            u_windows.append(us_arr[traj_idx, start : start + horizon])
    return np.stack(x_windows, axis=0), np.stack(u_windows, axis=0)


@dataclass
class TrainResult:
    """Output of one deep Koopman training run."""

    model: DeepKoopmanModel
    train_loss: float
    val_multistep_error: float
    history: Dict[str, Array] = field(default_factory=dict)


def split_train_val(
    num_windows: int, val_fraction: float, seed: int
) -> Tuple[Array, Array]:
    """Deterministic train/validation index split."""

    rng = np.random.default_rng(seed)
    perm = rng.permutation(num_windows)
    num_val = int(round(val_fraction * num_windows))
    return perm[num_val:], perm[:num_val]


def multistep_loss(
    model: DeepKoopmanModel,
    x: torch.Tensor,
    u: torch.Tensor,
    gamma_latent: float,
) -> torch.Tensor:
    """Average per-step decoded + latent-consistency loss over one window batch.

    Args:
        x: States with shape (batch, H + 1, state_dim).
        u: Controls with shape (batch, H, action_dim).
    """

    horizon = u.shape[1]
    z = model.lift(x[:, 0])
    loss = x.new_zeros(())
    for k in range(horizon):
        z = model.step(z, u[:, k])
        loss = loss + ((model.decode(z) - x[:, k + 1]) ** 2).mean()
        if gamma_latent > 0.0:
            latent_target = model.lift(x[:, k + 1]).detach()
            loss = loss + gamma_latent * ((z - latent_target) ** 2).mean()
    return loss / horizon


@torch.no_grad()
def multistep_val_error(
    model: DeepKoopmanModel,
    x: torch.Tensor,
    u: torch.Tensor,
    batch_size: int = 2048,
) -> float:
    """Decoded multi-step MSE (no latent term) on a validation set."""

    total = 0.0
    count = 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        ub = u[i : i + batch_size]
        zs = model.rollout(model.lift(xb[:, 0]), ub)
        err = ((model.decode(zs[:, 1:]) - xb[:, 1:]) ** 2).mean()
        total += float(err) * xb.shape[0]
        count += xb.shape[0]
    return total / max(count, 1)


@torch.no_grad()
def open_loop_error(
    model: DeepKoopmanModel,
    x: torch.Tensor,
    u: torch.Tensor,
    batch_size: int = 2048,
) -> Dict[str, Array]:
    """Per-step open-loop decoded RMSE over full windows."""

    horizon = u.shape[1]
    sq_sum = np.zeros(horizon, dtype=np.float64)
    count = 0
    for i in range(0, x.shape[0], batch_size):
        xb = x[i : i + batch_size]
        ub = u[i : i + batch_size]
        zs = model.rollout(model.lift(xb[:, 0]), ub)
        sq = ((model.decode(zs[:, 1:]) - xb[:, 1:]) ** 2).mean(dim=-1)
        sq_sum += sq.sum(dim=0).cpu().numpy()
        count += xb.shape[0]
    rmse_per_step = np.sqrt(sq_sum / max(count, 1))
    return {
        "rmse_per_step": rmse_per_step,
        "rmse_mean": float(rmse_per_step.mean()),
        "rmse_final": float(rmse_per_step[-1]),
    }


def train_deep_koopman(
    model: DeepKoopmanModel,
    states: Array,
    controls: Array,
    config: KoopmanTrainConfig,
    *,
    device: torch.device | str = "cpu",
    eval_every: int = 10,
    verbose: bool = False,
    progress: bool = True,
    progress_desc: str = "train",
) -> TrainResult:
    """Train a linear DK or bilinear BK model with the shared multi-step loss.

    Args:
        model: DeepKoopmanModel (linear or bilinear).
        states: Koopman states, shape (num_traj, T + 1, state_dim).
        controls: Controls, shape (num_traj, T, action_dim).
        config: Shared training settings.
    """

    x_windows, u_windows = build_multistep_windows(
        states, controls, config.rollout_horizon
    )
    train_idx, val_idx = split_train_val(
        x_windows.shape[0], config.val_fraction, config.seed
    )
    device = torch.device(device)
    x_all = torch.as_tensor(x_windows, dtype=torch.float32, device=device)
    u_all = torch.as_tensor(u_windows, dtype=torch.float32, device=device)
    x_train, u_train = x_all[train_idx], u_all[train_idx]
    x_val, u_val = x_all[val_idx], u_all[val_idx]

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, config.num_epochs)
        if config.cosine_lr
        else None
    )
    shuffle_gen = torch.Generator(device="cpu").manual_seed(config.seed)

    train_loss_history = []
    val_error_history = []
    val_error_epochs = []
    last_train_loss = float("nan")

    epoch_iter = range(config.num_epochs)
    bar = None
    if progress and tqdm is not None:
        bar = tqdm(epoch_iter, desc=progress_desc, unit="epoch", ncols=110)
        epoch_iter = bar

    for epoch in epoch_iter:
        model.train()
        perm = torch.randperm(x_train.shape[0], generator=shuffle_gen)
        epoch_loss = 0.0
        num_batches = 0
        for i in range(0, x_train.shape[0], config.batch_size):
            idx = perm[i : i + config.batch_size]
            loss = multistep_loss(
                model, x_train[idx], u_train[idx], config.gamma_latent
            )
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()
            epoch_loss += float(loss)
            num_batches += 1
        if scheduler is not None:
            scheduler.step()
        last_train_loss = epoch_loss / max(num_batches, 1)
        train_loss_history.append(last_train_loss)

        is_last = epoch == config.num_epochs - 1
        if x_val.shape[0] > 0 and (epoch % eval_every == 0 or is_last):
            model.eval()
            val_err = multistep_val_error(model, x_val, u_val)
            val_error_history.append(val_err)
            val_error_epochs.append(epoch)
            if verbose and bar is None:
                print(
                    f"epoch {epoch + 1}/{config.num_epochs} "
                    f"train_loss={last_train_loss:.6f} val_mse={val_err:.6f}"
                )

        if bar is not None:
            postfix = {"train_loss": f"{last_train_loss:.5f}"}
            if val_error_history:
                postfix["val_mse"] = f"{val_error_history[-1]:.5f}"
            bar.set_postfix(postfix)

    if bar is not None:
        bar.close()

    final_val = val_error_history[-1] if val_error_history else float("nan")
    return TrainResult(
        model=model,
        train_loss=last_train_loss,
        val_multistep_error=final_val,
        history={
            "train_loss": np.asarray(train_loss_history),
            "val_multistep_error": np.asarray(val_error_history),
            "val_epochs": np.asarray(val_error_epochs),
            "train_indices": train_idx,
            "val_indices": val_idx,
        },
    )


def save_checkpoint(
    path: Path | str,
    model: DeepKoopmanModel,
    *,
    extras: Dict[str, object] | None = None,
) -> None:
    """Save model weights together with the config needed to rebuild it."""

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cfg = model.config
    torch.save(
        {
            "state_dict": model.state_dict(),
            "bilinear": model.bilinear,
            "model_kind": model.kind,
            "model_config": {
                "state_dim": cfg.state_dim,
                "action_dim": cfg.action_dim,
                "lift_dim": cfg.lift_dim,
                "hidden_width": cfg.hidden_width,
                "hidden_depth": cfg.hidden_depth,
            },
            "extras": extras or {},
        },
        path,
    )


def load_checkpoint(
    path: Path | str, *, device: torch.device | str = "cpu"
) -> Tuple[DeepKoopmanModel, Dict[str, object]]:
    """Rebuild a DeepKoopmanModel from a checkpoint file."""

    payload = torch.load(Path(path), map_location=device, weights_only=False)
    config = KoopmanModelConfig(**payload["model_config"])
    model_cls = MODEL_KINDS[payload.get("model_kind", "default")]
    model = model_cls(config, bilinear=payload["bilinear"])
    model.load_state_dict(payload["state_dict"])
    model.to(device)
    model.eval()
    return model, payload.get("extras", {})

