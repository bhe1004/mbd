"""Section C batch experiment: BK-MBD vs QP-MPC on the wall-reaching task.

Runs both planners headless in lockstep on a shared set of wall-behind targets
and writes one JSON line per trial to ``out/exp_c/trials.jsonl``. Re-running
skips trials already recorded (resume-safe).

    python secy/experiment_c.py                 # both planners, all targets
    python secy/experiment_c.py --planner bk_mbd

Design (frozen for reproducibility), following the single-model / vary-the-
initial-conditions protocol of the prior Koopman-MPPI work:
  * ONE fixed, well-trained model (bk_seed0.pt) for BOTH planners, so the
    comparison isolates the optimizer (global sampling vs per-step
    convexification), not model-training variance across seeds.
  * N_TARGETS reach targets sampled once (fixed seed) from a LOW region behind
    the wall (z 0.15-0.24): the arm must clear the wall top (z=0.5) and descend
    deep behind it, the regime where the convexification cannot re-select the
    passage branch. QP-MPC is deterministic given (model, target); BK-MBD's
    sampling rng is seeded per target index (reproducible, independent stream).
  * lockstep, viewer off, wait-for-start off; episode capped by MAX_SIM_STEPS
    sim control steps (max_time raised so the wall clock never binds), so both
    planners get the SAME sim-time budget regardless of per-plan latency.
Statistics are computed separately by ``experiment_c_stats.py``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from secy.config import DEFAULT_CONFIG, load_config  # noqa: E402
from secy.environment import Scene  # noqa: E402
from secy.bkmbd import BKMBDPlanner  # noqa: E402
from secy.sqp_mpc import SQPMPCPlanner  # noqa: E402
from secy.runtime import RealtimeRunner  # noqa: E402

# ---- frozen experiment design -------------------------------------------------
FIXED_MODEL_SEED = 0         # both planners load bk_seed0.pt (a good model)
MODEL_CKPT = PROJECT_ROOT / "out" / "franka" / "models" / f"bk_seed{FIXED_MODEL_SEED}.pt"
N_TARGETS = 35
TARGET_SAMPLING_SEED = 20260727   # fixed -> the 35 targets are reproducible
# Wall-behind sampling box: x within the wall span, y on the far side, z spanning
# deep-to-moderate descent (0.20-0.35) below the wall top (config box, top ~0.55)
# so the tool must clear the wall and descend behind it. Obstacle geometry itself
# is read from config.json (base_cfg), so editing the wall there changes the task.
X_RANGE = (0.35, 0.55)
Y_RANGE = (0.28, 0.36)
Z_RANGE = (0.20, 0.35)
MAX_SIM_STEPS = 600          # 30 s of sim at a 50 ms control period (the sole cap)
PLANNERS = {"bk_mbd": BKMBDPlanner, "sqp_mpc": SQPMPCPlanner}

OUT_DIR = PROJECT_ROOT / "out" / "exp_c"
TRIALS_PATH = OUT_DIR / "trials.jsonl"


def sample_targets(n: int, seed: int):
    rng = np.random.default_rng(seed)
    xs = rng.uniform(*X_RANGE, n)
    ys = rng.uniform(*Y_RANGE, n)
    zs = rng.uniform(*Z_RANGE, n)
    return [(float(x), float(y), float(z)) for x, y, z in zip(xs, ys, zs)]


TARGETS = sample_targets(N_TARGETS, TARGET_SAMPLING_SEED)


def trial_config(cfg, target_idx: int, target):
    """Config for one headless lockstep trial: model fixed to bk_seed0, the
    rng seeded by the target index (BK only; QP-MPC ignores it)."""
    return replace(
        cfg,
        env=replace(cfg.env, target=tuple(float(c) for c in target),
                    wait_for_start=False),
        mbd=replace(cfg.mbd, seed=int(target_idx), checkpoint=MODEL_CKPT),
        runtime=replace(cfg.runtime, mode="lockstep", viewer=False,
                        target_ids=[0], max_time=1e9),  # sim-step cap is the only bound
    )


def done_keys(path: Path):
    """(planner, target_idx) tuples already recorded, for resume."""
    seen = set()
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            seen.add((r["planner"], r["target_idx"]))
    return seen


def run_one(base_cfg, planner_name, target_idx, target, device):
    cfg = trial_config(base_cfg, target_idx, target)
    scene = Scene(cfg, device=device)
    planner = PLANNERS[planner_name](scene, cfg)
    runner = RealtimeRunner(scene, planner, cfg)
    runner.run(interactive=False, max_boundaries=MAX_SIM_STEPS)
    res = dict(runner.last_result)
    res.update(planner=planner_name, target_idx=int(target_idx),
               target=list(map(float, target)),
               model_seed=FIXED_MODEL_SEED, rng_seed=int(target_idx))
    # derived reach flags at both tolerances (min_err is the closest approach)
    res["reached_5cm"] = bool(res["min_err"] < 0.05)
    res["reached_2p5cm"] = bool(res["min_err"] < 0.025)
    res["safe_stall"] = bool(not res["reached_strict"]
                             and not res["executed_violation"])
    return res


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--planner", choices=list(PLANNERS), default=None,
                    help="restrict to one planner (default: both)")
    ap.add_argument("-c", "--config", type=Path, default=DEFAULT_CONFIG)
    args = ap.parse_args()

    base_cfg = load_config(args.config)
    if base_cfg.runtime.torch_threads > 0:
        torch.set_num_threads(base_cfg.runtime.torch_threads)
    device = torch.device("cpu")
    if not MODEL_CKPT.exists():
        raise SystemExit(f"fixed model checkpoint not found: {MODEL_CKPT}")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    planners = [args.planner] if args.planner else list(PLANNERS)
    seen = done_keys(TRIALS_PATH)

    jobs = [(p, ti, t) for p in planners for ti, t in enumerate(TARGETS)
            if (p, int(ti)) not in seen]
    total = len(planners) * len(TARGETS)
    print(f"exp_c: model=bk_seed{FIXED_MODEL_SEED}, {N_TARGETS} targets, "
          f"{len(jobs)} trials to run ({total - len(jobs)} done, {total} total) "
          f"-> {TRIALS_PATH}", flush=True)

    t0 = time.perf_counter()
    with open(TRIALS_PATH, "a") as fh:
        for i, (p, ti, t) in enumerate(jobs, 1):
            print(f"\n=== [{i}/{len(jobs)}] {p} target={ti} "
                  f"{tuple(round(c, 3) for c in t)} ===", flush=True)
            try:
                res = run_one(base_cfg, p, ti, t, device)
                res["error"] = None
            except Exception as exc:  # keep the batch alive; record the failure
                res = dict(planner=p, target_idx=int(ti),
                           target=list(map(float, t)), error=repr(exc))
                print(f"  TRIAL FAILED: {exc!r}", flush=True)
            fh.write(json.dumps(res) + "\n")
            fh.flush()
            if res.get("error") is None:
                print(f"  reached_strict={res['reached_strict']} "
                      f"reached_5cm={res['reached_5cm']} "
                      f"min_err={res['min_err']:.4f} "
                      f"violation={res['executed_violation']} "
                      f"safe_stall={res['safe_stall']}", flush=True)
    dt = time.perf_counter() - t0
    print(f"\nexp_c done: {len(jobs)} trials in {dt / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
