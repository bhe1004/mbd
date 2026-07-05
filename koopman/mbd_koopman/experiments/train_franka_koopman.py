"""Train linear DK and bilinear BK Koopman models on the Franka FR3 MuJoCo dataset.

Mirrors `experiments/train_unicycle_koopman.py` with the arm settings of
`koopman/mppi_koopman/verify/arm_multiseed_v2.py`:

- Koopman state b = [q, ee] (10-dim), encoder input [sin q, cos q],
- lift 10 + 10 = 20, hidden width 96,
- 300 epochs, Adam(1e-3, weight_decay=1e-4), cosine LR decay,
- gradient clipping at norm 2.0,
- multi-step horizon 15 with latent weight 0.1,
- zero-initialized bilinear matrices for BK.

Checkpoints and a training summary CSV are written to `out/arm/models/`.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import KoopmanModelConfig, KoopmanTrainConfig  # noqa: E402
from bk_mbd.models import ArmDeepKoopmanModel  # noqa: E402
from bk_mbd.train import (  # noqa: E402
    build_multistep_windows,
    open_loop_error,
    save_checkpoint,
    split_train_val,
    train_deep_koopman,
)
from envs.franka import FrankaTask  # noqa: E402
from train_unicycle_koopman import write_training_summary  # noqa: E402

MODEL_NAMES = ("dk", "bk", "dk_split")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--seed", type=int, default=0, help="training seed")
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--lift-extra", type=int, default=10)
    parser.add_argument("--hidden-width", type=int, default=96)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--rollout-horizon", type=int, default=15)
    parser.add_argument("--gamma-latent", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip", type=float, default=2.0)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--output-dir", type=Path, default=PROJECT_ROOT / "out" / "franka" / "models"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)

    task = FrankaTask()
    dataset = task.sample_dataset(args.data_seed)
    controls = dataset["controls"]

    train_config = KoopmanTrainConfig(
        rollout_horizon=args.rollout_horizon,
        gamma_latent=args.gamma_latent,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        val_fraction=args.val_fraction,
        seed=args.seed,
        grad_clip=args.grad_clip,
        cosine_lr=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "training_summary.csv"

    for name in args.models:
        # dk_split learns q-only dynamics (input enters state-independently);
        # dk/bk learn the full base state b = [q, ee] including the coupling.
        # Dimensions are read off the data (the FR3 physical state is 14-dim
        # [q, qd], but the split Koopman state is the 7-dim q only).
        if name == "dk_split":
            train_states = dataset["states"]
        else:
            train_states = dataset["base_states"]
        state_dim = int(train_states.shape[-1])
        model_config = KoopmanModelConfig(
            state_dim=state_dim,
            action_dim=task.action_dim,
            lift_dim=state_dim + args.lift_extra,
            hidden_width=args.hidden_width,
            hidden_depth=args.hidden_depth,
        )

        x_windows, u_windows = build_multistep_windows(
            train_states, controls, train_config.rollout_horizon
        )
        _, val_idx = split_train_val(
            x_windows.shape[0], train_config.val_fraction, train_config.seed
        )
        x_val = torch.as_tensor(x_windows[val_idx], dtype=torch.float32, device=device)
        u_val = torch.as_tensor(u_windows[val_idx], dtype=torch.float32, device=device)

        torch.manual_seed(args.seed)
        model = ArmDeepKoopmanModel(model_config, bilinear=(name == "bk"))

        start = time.perf_counter()
        result = train_deep_koopman(
            model,
            train_states,
            controls,
            train_config,
            device=device,
            eval_every=args.eval_every,
            verbose=args.verbose,
            progress_desc=f"arm {name} seed{args.seed}",
        )
        train_time = time.perf_counter() - start

        result.model.eval()
        ol = open_loop_error(result.model, x_val, u_val)

        checkpoint = args.output_dir / f"{name}_seed{args.seed}.pt"
        save_checkpoint(
            checkpoint,
            result.model,
            extras={
                "task": "franka",
                "model": name,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "train_loss": result.train_loss,
                "val_multistep_error": result.val_multistep_error,
                "open_loop_rmse_mean": ol["rmse_mean"],
                "open_loop_rmse_final": ol["rmse_final"],
                "open_loop_rmse_per_step": ol["rmse_per_step"],
                "rollout_horizon": train_config.rollout_horizon,
            },
        )
        write_training_summary(
            summary_path,
            {
                "task": "franka",
                "model": name,
                "seed": args.seed,
                "data_seed": args.data_seed,
                "lift_dim": model_config.lift_dim,
                "rollout_horizon": train_config.rollout_horizon,
                "epochs": train_config.num_epochs,
                "train_loss": result.train_loss,
                "val_multistep_error": result.val_multistep_error,
                "open_loop_rmse_mean": ol["rmse_mean"],
                "open_loop_rmse_final": ol["rmse_final"],
                "train_time_s": train_time,
                "checkpoint": str(checkpoint),
            },
        )
        print(
            " ".join(
                [
                    f"model={name}",
                    f"seed={args.seed}",
                    f"train_loss={result.train_loss:.6f}",
                    f"val_mse={result.val_multistep_error:.6f}",
                    f"open_loop_rmse_mean={ol['rmse_mean']:.6f}",
                    f"open_loop_rmse_final={ol['rmse_final']:.6f}",
                    f"train_time_s={train_time:.1f}",
                ]
            )
        )
        print(f"saved={checkpoint}")

    print(f"summary={summary_path}")


if __name__ == "__main__":
    main()
