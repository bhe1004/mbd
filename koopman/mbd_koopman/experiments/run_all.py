"""Multi-seed comparison runner for the unicycle and arm tasks (Milestone 3).

Follows the protocol of the reference multi-seed scripts
(`unicycle_multiseed_v2.py`, `arm_multiseed_v2.py`):

- dataset generated once with a fixed data seed,
- per training seed: train linear DK and bilinear BK on the same data,
- per (seed, case) trial: planning seed = seed * 10 + case,
- the true-dynamics oracle runs on every trial as well,
- reports success rate and final-error statistics over
  len(seeds) x len(cases) trials per method.

Existing checkpoints are reused unless --retrain is given. Per-trial rows are
appended to `out/{task}/summary.csv`; the aggregate table is written to
`out/{task}/multiseed_summary.csv`.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from bk_mbd.config import (  # noqa: E402
    KoopmanModelConfig,
    KoopmanTrainConfig,
    MBDConfig,
    MethodName,
    UpdateRule,
)
from bk_mbd.mbd import MBDOptimizer  # noqa: E402
from bk_mbd.models import ArmDeepKoopmanModel, DeepKoopmanModel  # noqa: E402
from bk_mbd.train import (  # noqa: E402
    load_checkpoint,
    open_loop_error,
    save_checkpoint,
    split_train_val,
    build_multistep_windows,
    train_deep_koopman,
)
from bk_mbd.tube import (  # noqa: E402
    compute_one_step_residuals,
    fit_tube_constants,
)
from envs.arm import ArmTask, ArmTaskConfig  # noqa: E402
from envs.franka import FrankaTask, FrankaTaskConfig  # noqa: E402
from envs.unicycle import UnicycleTask, UnicycleTaskConfig  # noqa: E402
from run_unicycle import write_summary  # noqa: E402

KOOPMAN_SHORT = {MethodName.DK_MBD.value: "dk", MethodName.BK_MBD.value: "bk"}
# dk_mbd_split: on the arm/franka it uses a q-only checkpoint ("dk_split");
# on the unicycle it is a least-squares body-frame model fitted at run time
# (no checkpoint).
SPLIT_CHECKPOINT_TASKS = {"arm": "dk_split", "franka": "dk_split"}

# Per-task defaults; every entry can be overridden from the CLI.
TASK_PRESETS: Dict[str, Dict[str, Any]] = {
    "unicycle": dict(
        task_cls=UnicycleTask,
        task_config_cls=UnicycleTaskConfig,
        model_cls=DeepKoopmanModel,
        cases=[0, 1, 2, 3],
        case_stem="case",
        # Koopman training (reference unicycle_multiseed_v2).
        lift_extra=4,
        hidden_width=64,
        epochs=200,
        weight_decay=0.0,
        grad_clip=0.0,
        cosine_lr=False,
        # MBD planner.
        num_samples=768,
        sigma_start=0.9,
        sigma_end=0.2,
        horizon=40,
        closed_loop_steps=180,
        beta_e=0.01,
    ),
    "arm": dict(
        task_cls=ArmTask,
        task_config_cls=ArmTaskConfig,
        model_cls=ArmDeepKoopmanModel,
        cases=[0, 1, 2, 3, 4, 5, 6],
        case_stem="target",
        # Koopman training (reference arm_multiseed_v2).
        lift_extra=10,
        hidden_width=96,
        epochs=300,
        weight_decay=1e-4,
        grad_clip=2.0,
        cosine_lr=True,
        # MBD planner.
        num_samples=800,
        sigma_start=1.2,
        sigma_end=0.3,
        horizon=15,
        closed_loop_steps=120,
        beta_e=0.0002,
    ),
    "franka": dict(
        task_cls=FrankaTask,
        task_config_cls=FrankaTaskConfig,
        model_cls=ArmDeepKoopmanModel,  # same b = [q, ee] structure
        cases=[0, 1, 2, 3, 4, 5, 6],
        case_stem="target",
        # Koopman training (same recipe as the arm task).
        lift_extra=10,
        hidden_width=96,
        epochs=300,
        weight_decay=1e-4,
        grad_clip=2.0,
        cosine_lr=True,
        # MBD planner.
        num_samples=800,
        sigma_start=1.2,
        sigma_end=0.3,
        horizon=15,
        closed_loop_steps=120,
        beta_e=0.0002,
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=list(TASK_PRESETS), default="unicycle")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument(
        "--cases", type=int, nargs="+", default=None, help="default: all task cases"
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        choices=[item.value for item in MethodName],
        default=[item.value for item in MethodName],
    )
    parser.add_argument("--data-seed", type=int, default=1)
    parser.add_argument("--retrain", action="store_true")
    # Koopman training overrides (None -> task preset).
    parser.add_argument("--lift-extra", type=int, default=None)
    parser.add_argument("--hidden-width", type=int, default=None)
    parser.add_argument("--hidden-depth", type=int, default=2)
    parser.add_argument("--rollout-horizon", type=int, default=15)
    parser.add_argument("--gamma-latent", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    # MBD planner overrides (None -> task preset).
    parser.add_argument("--num-samples", type=int, default=None)
    parser.add_argument("--num-diffusion-steps", type=int, default=5)
    parser.add_argument("--sigma-start", type=float, default=None)
    parser.add_argument("--sigma-end", type=float, default=None)
    parser.add_argument("--alpha", type=float, default=0.4)
    parser.add_argument(
        "--eta",
        type=float,
        default=1.0,
        help="score step scale; eta_s = eta * sigma_s^2 (1.0 = weighted mean)",
    )
    parser.add_argument("--absolute-eta", action="store_true")
    parser.add_argument(
        "--update-rule",
        choices=[item.value for item in UpdateRule],
        default=UpdateRule.SCORE_LANGEVIN.value,
    )
    parser.add_argument("--langevin-noise", action="store_true")
    parser.add_argument("--horizon", type=int, default=None)
    parser.add_argument("--closed-loop-steps", type=int, default=None)
    parser.add_argument("--beta-e", type=float, default=None)
    parser.add_argument(
        "--rollout-threads",
        type=int,
        default=None,
        help="MuJoCo rollout worker threads (franka only); lower this when "
        "running several cases in parallel processes",
    )
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    preset = TASK_PRESETS[args.task]
    for key in (
        "cases",
        "lift_extra",
        "hidden_width",
        "epochs",
        "num_samples",
        "sigma_start",
        "sigma_end",
        "horizon",
        "closed_loop_steps",
        "beta_e",
    ):
        if getattr(args, key) is None:
            setattr(args, key, preset[key])
    if args.output_dir is None:
        args.output_dir = PROJECT_ROOT / "out" / args.task
    return args


def ensure_model(
    name: str,
    seed: int,
    task,
    dataset: Dict[str, np.ndarray],
    args: argparse.Namespace,
    device: torch.device,
):
    """Load a checkpoint for (name, seed), training it first if needed."""

    preset = TASK_PRESETS[args.task]
    checkpoint = args.output_dir / "models" / f"{name}_seed{seed}.pt"
    if checkpoint.exists() and not args.retrain:
        return load_checkpoint(checkpoint, device=device)

    # dk_split learns q-only dynamics; dk/bk learn the full base state.
    # The Koopman state dimension is read off the data (e.g. the Franka task
    # has a 14-dim physical state [q, qd] but a 7-dim q-only split state).
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
    train_config = KoopmanTrainConfig(
        rollout_horizon=args.rollout_horizon,
        gamma_latent=args.gamma_latent,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
        weight_decay=preset["weight_decay"],
        val_fraction=args.val_fraction,
        seed=seed,
        grad_clip=preset["grad_clip"],
        cosine_lr=preset["cosine_lr"],
    )
    torch.manual_seed(seed)
    model = preset["model_cls"](model_config, bilinear=(name == "bk"))
    result = train_deep_koopman(
        model,
        train_states,
        dataset["controls"],
        train_config,
        device=device,
        progress_desc=f"{args.task} {name} seed{seed}",
    )

    x_windows, u_windows = build_multistep_windows(
        train_states, dataset["controls"], train_config.rollout_horizon
    )
    _, val_idx = split_train_val(
        x_windows.shape[0], train_config.val_fraction, train_config.seed
    )
    x_val = torch.as_tensor(x_windows[val_idx], dtype=torch.float32, device=device)
    u_val = torch.as_tensor(u_windows[val_idx], dtype=torch.float32, device=device)
    result.model.eval()
    ol = open_loop_error(result.model, x_val, u_val)

    extras = {
        "task": args.task,
        "model": name,
        "seed": seed,
        "data_seed": args.data_seed,
        "train_loss": result.train_loss,
        "val_multistep_error": result.val_multistep_error,
        "open_loop_rmse_mean": ol["rmse_mean"],
        "open_loop_rmse_final": ol["rmse_final"],
        "rollout_horizon": train_config.rollout_horizon,
    }
    save_checkpoint(checkpoint, result.model, extras=extras)
    return result.model, extras


def aggregate(results: List[Dict[str, Any]], methods: List[str]) -> List[Dict[str, Any]]:
    rows = []
    for method in methods:
        rs = [r for r in results if r["method"] == method]
        if not rs:
            continue
        final_errors = np.array([r["final_error"] for r in rs], dtype=np.float64)
        successes = np.array([bool(r["success"]) for r in rs])
        koopman_errs = [r.get("koopman_open_loop_error", np.nan) for r in rs]
        tube_costs = [r.get("tube_cost", np.nan) for r in rs]
        max_tubes = [r.get("max_tube", np.nan) for r in rs]
        rows.append(
            {
                "method": method,
                "num_trials": len(rs),
                "success_rate": float(successes.mean()),
                "final_error_mean": float(final_errors.mean()),
                "final_error_std": float(final_errors.std()),
                "final_error_mean_success": float(final_errors[successes].mean())
                if successes.any()
                else np.nan,
                "planning_time_ms_mean": float(
                    np.mean([r["planning_time_ms"] for r in rs])
                ),
                "koopman_open_loop_error_mean": float(np.nanmean(koopman_errs))
                if not all(np.isnan(v) for v in koopman_errs)
                else np.nan,
                "tube_cost_mean": float(np.nanmean(tube_costs))
                if not all(np.isnan(v) for v in tube_costs)
                else np.nan,
                "max_tube": float(np.nanmax(max_tubes))
                if not all(np.isnan(v) for v in max_tubes)
                else np.nan,
            }
        )
    return rows


def main() -> None:
    args = parse_args()
    if args.device != "cpu" and not torch.cuda.is_available():
        raise SystemExit(f"requested device {args.device}, but CUDA is unavailable")
    device = torch.device(args.device)

    preset = TASK_PRESETS[args.task]
    task_config_kwargs = dict(
        horizon=args.horizon, closed_loop_steps=args.closed_loop_steps
    )
    if args.rollout_threads is not None:
        if args.task != "franka":
            raise SystemExit("--rollout-threads only applies to --task franka")
        task_config_kwargs["num_rollout_threads"] = args.rollout_threads
    task = preset["task_cls"](preset["task_config_cls"](**task_config_kwargs))
    dataset = task.sample_dataset(args.data_seed)
    mbd_config = MBDConfig(
        num_samples=args.num_samples,
        num_diffusion_steps=args.num_diffusion_steps,
        sigma_start=args.sigma_start,
        sigma_end=args.sigma_end,
        alpha=args.alpha,
        eta=args.eta,
        update_rule=UpdateRule(args.update_rule),
        add_langevin_noise=args.langevin_noise,
        eta_relative=not args.absolute_eta,
    )
    optimizer = MBDOptimizer(
        mbd_config,
        action_low=task.action_bounds[0],
        action_high=task.action_bounds[1],
    )

    koopman_methods = [m for m in args.methods if m in KOOPMAN_SHORT]
    results: List[Dict[str, Any]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem_key = preset["case_stem"]

    use_split = MethodName.DK_MBD_SPLIT.value in args.methods
    split_G = None
    if use_split and args.task == "unicycle":
        split_G = task.fit_body_frame_linear(dataset)
        print(f"unicycle split model G=\n{np.array2string(split_G, precision=4)}")

    for seed in args.seeds:
        models: Dict[str, Any] = {}
        tube_constants = None
        for method in koopman_methods:
            short = KOOPMAN_SHORT[method]
            models[short] = ensure_model(short, seed, task, dataset, args, device)
        if use_split and args.task in SPLIT_CHECKPOINT_TASKS:
            short = SPLIT_CHECKPOINT_TASKS[args.task]
            models[short] = ensure_model(short, seed, task, dataset, args, device)
        if MethodName.BK_MBD.value in args.methods:
            bk_model = models["bk"][0]
            zs, us, residuals = compute_one_step_residuals(
                bk_model, dataset["base_states"], dataset["controls"], device=device
            )
            tube_constants = fit_tube_constants(zs, us, residuals, quantile=0.999)
            print(
                f"seed {seed}: tube c_x={tube_constants.c_x:.4e} "
                f"c_u={tube_constants.c_u:.4e}"
            )

        seed_success: Dict[str, int] = {m: 0 for m in args.methods}
        for case_id in args.cases:
            plan_seed = seed * 10 + case_id
            for method in args.methods:
                start_time = time.perf_counter()
                if method == MethodName.VANILLA_MBD_TRUE.value:
                    result = task.closed_loop_true_mbd(
                        optimizer, case_id=case_id, seed=plan_seed, device=device
                    )
                    koopman_fields: Dict[str, Any] = {}
                elif method == MethodName.DK_MBD_SPLIT.value:
                    if args.task == "unicycle":
                        result = task.closed_loop_koopman_split_mbd(
                            optimizer,
                            split_G,
                            case_id=case_id,
                            seed=plan_seed,
                            device=device,
                        )
                        koopman_fields = {}
                    else:
                        model, extras = models[SPLIT_CHECKPOINT_TASKS[args.task]]
                        result = task.closed_loop_koopman_split_mbd(
                            optimizer,
                            model,
                            case_id=case_id,
                            seed=plan_seed,
                            device=device,
                        )
                        koopman_fields = {
                            "train_loss": extras.get("train_loss", np.nan),
                            "val_multistep_error": extras.get(
                                "val_multistep_error", np.nan
                            ),
                            "koopman_open_loop_error": extras.get(
                                "open_loop_rmse_mean", np.nan
                            ),
                        }
                else:
                    short = KOOPMAN_SHORT[method]
                    model, extras = models[short]
                    result = task.closed_loop_koopman_mbd(
                        optimizer,
                        model,
                        method=method,
                        case_id=case_id,
                        seed=plan_seed,
                        device=device,
                        tube_constants=(
                            tube_constants
                            if method == MethodName.BK_MBD.value
                            else None
                        ),
                        beta_e=args.beta_e,
                    )
                    koopman_fields = {
                        "train_loss": extras.get("train_loss", np.nan),
                        "val_multistep_error": extras.get(
                            "val_multistep_error", np.nan
                        ),
                        "koopman_open_loop_error": extras.get(
                            "open_loop_rmse_mean", np.nan
                        ),
                    }
                elapsed = time.perf_counter() - start_time
                steps = max(int(result["steps"]), 1)

                row = {
                    **result,
                    **koopman_fields,
                    "seed": seed,
                    "planning_time_ms": 1000.0 * elapsed / steps,
                    "total_cost": float(np.sum(result["best_costs"])),
                }
                if "tube_cost" in result:
                    row["tube_cost"] = float(np.mean(result["tube_cost"]))
                    row["max_tube"] = float(np.max(result["max_tube"]))
                results.append(row)
                seed_success[method] += int(bool(result["success"]))
                write_summary(args.output_dir / "summary.csv", row)

                np.savez(
                    args.output_dir / f"{method}_{stem_key}{case_id}_seed{seed}.npz",
                    trajectory=result["trajectory"],
                    controls=result["controls"],
                    best_costs=result["best_costs"],
                )
        stats = "  ".join(
            f"{m}={seed_success[m]}/{len(args.cases)}" for m in args.methods
        )
        print(f"seed {seed}: {stats}")

    agg_rows = aggregate(results, args.methods)
    agg_path = args.output_dir / "multiseed_summary.csv"
    with agg_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(agg_rows[0].keys()))
        writer.writeheader()
        writer.writerows(agg_rows)

    total = len(args.seeds) * len(args.cases)
    print(f"\n[{args.task}] success rate over {total} trials:")
    for row in agg_rows:
        print(
            f"  {row['method']:<18}: {int(row['success_rate'] * row['num_trials'])}"
            f"/{row['num_trials']} ({100 * row['success_rate']:.0f}%)  "
            f"final {row['final_error_mean']:.3f}+/-{row['final_error_std']:.3f} m  "
            f"plan {row['planning_time_ms_mean']:.1f} ms/step"
        )
    print(f"aggregate={agg_path}")


if __name__ == "__main__":
    main()
