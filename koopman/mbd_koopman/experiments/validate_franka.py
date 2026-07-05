"""Validation checks for the FR3 MuJoCo environment (run before training).

Checks, in order:

1. velocity-servo step response  -- commanded vs realized joint velocity,
   reports the settling behaviour within one control period (50 ms),
2. torch FK vs MuJoCo site       -- the split-baseline FK must match the
   simulator to sub-millimetre accuracy,
3. dataset sanity                -- EE workspace coverage and joint-limit
   saturation fraction on a small sample,
4. oracle rollout timing         -- wall time of one batched MuJoCo
   evaluation (the rollout-cost reference for the paper).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs.franka import NUM_JOINTS, FrankaTask  # noqa: E402


def check_servo(task: FrankaTask) -> None:
    print("=== 1. velocity-servo step response ===")
    start, _, _ = task.case(0)
    u = np.zeros(NUM_JOINTS)
    for joint in range(NUM_JOINTS):
        u_j = u.copy()
        u_j[joint] = 0.5
        s = task.true_step(start, u_j)
        qd = s[NUM_JOINTS:]
        print(
            f"  joint {joint + 1}: qd after one control period (cmd 0.5) = "
            f"{qd[joint]:+.3f}  (tracking {100 * qd[joint] / 0.5:.0f}%)"
        )


def check_fk(task: FrankaTask) -> None:
    print("=== 2. torch FK vs MuJoCo site ===")
    rng = np.random.default_rng(0)
    qs = rng.uniform(task.joint_low + 0.05, task.joint_high - 0.05, (200, NUM_JOINTS))
    torch_ee = task.forward_kinematics_torch(
        torch.as_tensor(qs, dtype=torch.float64)
    ).numpy()
    mj_ee = np.stack([task.ee_of_q(q) for q in qs])
    err = np.abs(torch_ee - mj_ee).max()
    print(f"  max |torch FK - MuJoCo site| over 200 random q: {err:.2e} m")
    if err > 1e-6:
        raise SystemExit("FK mismatch: split baseline would be invalid")


def check_dataset(task: FrankaTask) -> None:
    print("=== 3. dataset sanity (200 snippets) ===")
    from dataclasses import replace

    small = FrankaTask(
        task.config,
        task.cost_weights,
        replace(task.dataset_config, num_snippets=200),
    )
    t0 = time.perf_counter()
    ds = small.sample_dataset(0)
    dt = time.perf_counter() - t0
    ee = ds["base_states"][..., NUM_JOINTS:]
    q = ds["states"]
    at_limit = np.mean(
        (q <= task.joint_low + 1e-3) | (q >= task.joint_high - 1e-3)
    )
    print(f"  generation time: {dt:.1f}s for 200 snippets "
          f"(full 6000 ~ {30 * dt:.0f}s)")
    print(f"  ee x range [{ee[..., 0].min():.2f}, {ee[..., 0].max():.2f}]  "
          f"y [{ee[..., 1].min():.2f}, {ee[..., 1].max():.2f}]  "
          f"z [{ee[..., 2].min():.2f}, {ee[..., 2].max():.2f}]")
    print(f"  joint-limit saturation fraction: {100 * at_limit:.2f}%")


def check_oracle_timing(task: FrankaTask) -> None:
    print("=== 4. oracle rollout timing (K=800, T=15) ===")
    start, goal, _ = task.case(0)
    evaluate = task.make_true_evaluate(start, goal, np.zeros(NUM_JOINTS), None)
    rng = np.random.default_rng(0)
    candidates = rng.uniform(-1.5, 1.5, (800, task.config.horizon, NUM_JOINTS))
    evaluate(candidates)  # warm-up
    t0 = time.perf_counter()
    costs = evaluate(candidates)
    dt = time.perf_counter() - t0
    print(f"  one batched evaluation: {1000 * dt:.0f} ms "
          f"(x{5} diffusion steps -> ~{5 * dt:.1f} s per control step)")
    print(f"  cost range: [{costs.min():.2f}, {costs.max():.2f}]")


def main() -> None:
    task = FrankaTask()
    print(f"model: {task.model.nq} dof, timestep {task.model.opt.timestep}, "
          f"substeps/control {task._nsub}, "
          f"rollout threads {task.config.num_rollout_threads}")
    home_ee = task.ee_of_q(task.home_qpos)
    print(f"home ee: {np.array2string(home_ee, precision=3)}")
    check_servo(task)
    check_fk(task)
    check_dataset(task)
    check_oracle_timing(task)
    print("all checks done")


if __name__ == "__main__":
    main()
