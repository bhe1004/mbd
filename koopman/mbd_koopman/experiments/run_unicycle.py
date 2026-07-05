"""Run the unicycle BK-MBD comparison.

Supports all three methods of the roadmap:

```text
vanilla_mbd_true = MBD + true dynamics rollout
dk_mbd           = MBD + linear deep Koopman rollout
bk_mbd           = MBD + bilinear deep Koopman rollout + tube cost
```

`dk_mbd` / `bk_mbd` load checkpoints produced by
`experiments/train_unicycle_koopman.py`. For `bk_mbd`, tube constants are
fitted on one-step residuals of the training dataset (linprog, full
coverage) before planning.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import MBDConfig, MethodName, UpdateRule  # noqa: E402
from bk_mbd.mbd import MBDOptimizer  # noqa: E402
from bk_mbd.train import load_checkpoint  # noqa: E402
from bk_mbd.tube import (  # noqa: E402
    compute_one_step_residuals,
    fit_tube_constants,
)
from envs.unicycle import UnicycleTask, UnicycleTaskConfig  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method",
        choices=[item.value for item in MethodName],
        default=MethodName.VANILLA_MBD_TRUE.value,
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Koopman checkpoint; defaults to out/unicycle/models/{dk,bk}_seed{seed}.pt",
    )
    parser.add_argument(
        "--data-seed",
        type=int,
        default=1,
        help="dataset seed used for BK tube fitting (must match training data)",
    )
    parser.add_argument("--beta-e", type=float, default=0.01, help="tube cost weight")
    parser.add_argument("--case-id", type=int, default=0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-samples", type=int, default=768)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--sigma-start", type=float, default=0.9)
    parser.add_argument("--sigma-end", type=float, default=0.2)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="score step scale; eta_s = eta * sigma_s^2 (1.0 = weighted mean)",
    )
    parser.add_argument(
        "--absolute-eta",
        action="store_true",
        help="legacy: use eta as a fixed absolute step size",
    )
    parser.add_argument(
        "--update-rule",
        choices=[item.value for item in UpdateRule],
        default=UpdateRule.SCORE_LANGEVIN.value,
    )
    parser.add_argument("--langevin-noise", action="store_true")
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--closed-loop-steps", type=int, default=180)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "out" / "unicycle",
    )
    return parser.parse_args()


def write_summary(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "task",
        "method",
        "seed",
        "case_id",
        "success",
        "strict_success",
        "final_error",
        "angle_error",
        "steps",
        "planning_time_ms",
        "total_cost",
        "train_loss",
        "val_multistep_error",
        "koopman_open_loop_error",
        "tube_cost",
        "max_tube",
    ]
    exists = path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, np.nan) for key in fieldnames})


def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)

    task = UnicycleTask(
        UnicycleTaskConfig(
            horizon=args.horizon,
            closed_loop_steps=args.closed_loop_steps,
        )
    )
    mbd_config = MBDConfig(
        num_samples=args.num_samples,
        num_diffusion_steps=args.num_diffusion_steps,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        alpha=args.alpha,
        eta=args.eta,
        update_rule=UpdateRule(args.update_rule),
        seed=args.seed,
        add_langevin_noise=args.langevin_noise,
        eta_relative=not args.absolute_eta,
    )
    optimizer = MBDOptimizer(
        mbd_config,
        action_low=task.action_bounds[0],
        action_high=task.action_bounds[1],
    )

    koopman_fields: Dict[str, Any] = {}
    model = None
    tube_constants = None
    split_G = None
    if args.method == MethodName.DK_MBD_SPLIT.value:
        # MPPI-DK-conditions baseline: linear body-frame model fitted by
        # least squares on the same dataset; no neural checkpoint involved.
        dataset = task.sample_dataset(args.data_seed)
        split_G = task.fit_body_frame_linear(dataset)
        print(f"split model G=\n{np.array2string(split_G, precision=4)}")
    if args.method in (MethodName.DK_MBD.value, MethodName.BK_MBD.value):
        short = "dk" if args.method == MethodName.DK_MBD.value else "bk"
        checkpoint = args.checkpoint or (
            PROJECT_ROOT / "out" / "unicycle" / "models" / f"{short}_seed{args.seed}.pt"
        )
        if not checkpoint.exists():
            raise SystemExit(
                f"checkpoint not found: {checkpoint}\n"
                "train it first: python experiments/train_unicycle_koopman.py"
            )
        model, extras = load_checkpoint(checkpoint, device=device)
        koopman_fields = {
            "train_loss": extras.get("train_loss", np.nan),
            "val_multistep_error": extras.get("val_multistep_error", np.nan),
            "koopman_open_loop_error": extras.get("open_loop_rmse_mean", np.nan),
        }
        print(f"loaded={checkpoint}")

        if args.method == MethodName.BK_MBD.value:
            dataset = task.sample_dataset(args.data_seed)
            zs, us, residuals = compute_one_step_residuals(
                model, dataset["base_states"], dataset["controls"], device=device
            )
            tube_constants = fit_tube_constants(zs, us, residuals, quantile=0.999)
            print(
                f"tube constants: c_x={tube_constants.c_x:.4e} "
                f"c_u={tube_constants.c_u:.4e} beta_e={args.beta_e}"
            )

    start_time = time.perf_counter()
    if args.method == MethodName.VANILLA_MBD_TRUE.value:
        result = task.closed_loop_true_mbd(
            optimizer,
            case_id=args.case_id,
            seed=args.seed,
            device=device,
        )
    elif args.method == MethodName.DK_MBD_SPLIT.value:
        result = task.closed_loop_koopman_split_mbd(
            optimizer,
            split_G,
            case_id=args.case_id,
            seed=args.seed,
            device=device,
        )
    else:
        result = task.closed_loop_koopman_mbd(
            optimizer,
            model,
            method=args.method,
            case_id=args.case_id,
            seed=args.seed,
            device=device,
            tube_constants=tube_constants,
            beta_e=args.beta_e,
        )
    elapsed = time.perf_counter() - start_time
    steps = max(int(result["steps"]), 1)
    planning_time_ms = 1000.0 * elapsed / steps

    total_cost = float(np.sum(result["best_costs"]))
    row = {
        **result,
        **koopman_fields,
        "planning_time_ms": planning_time_ms,
        "total_cost": total_cost,
    }
    if "tube_cost" in result:
        row["tube_cost"] = float(np.mean(result["tube_cost"]))
        row["max_tube"] = float(np.max(result["max_tube"]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.method}_case{args.case_id}_seed{args.seed}"
    npz_payload = {
        "trajectory": result["trajectory"],
        "controls": result["controls"],
        "best_costs": result["best_costs"],
    }
    for key in ("tube_cost", "max_tube"):
        if key in result:
            npz_payload[key] = result[key]
    np.savez(args.output_dir / f"{stem}.npz", **npz_payload)
    write_summary(args.output_dir / "summary.csv", row)

    print(
        " ".join(
            [
                f"method={args.method}",
                f"case={args.case_id}",
                f"seed={args.seed}",
                f"success={bool(result['success'])}",
                f"strict_success={bool(result['strict_success'])}",
                f"final_error={float(result['final_error']):.4f}",
                f"angle_error={float(result['angle_error']):.4f}",
                f"steps={int(result['steps'])}",
                f"planning_time_ms={planning_time_ms:.2f}",
            ]
        )
    )
    print(f"saved={args.output_dir / (stem + '.npz')}")
    print(f"summary={args.output_dir / 'summary.csv'}")


if __name__ == "__main__":
    main()

