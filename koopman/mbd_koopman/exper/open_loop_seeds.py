"""Held-out horizon error of the multi-step loss against a one-step fit, over seeds.

The paper justifies the multi-step loss (eq. 15) with a single pair of numbers,
measured on one training seed. This measures the same pair over every seed the
closed-loop studies use, so the comparison is reported the way the rest of the
paper is. The held-out set is drawn per seed, and both variants of a seed are
evaluated on that seed's set.

    python -m exper.open_loop_seeds --config exp_b
"""

from __future__ import annotations

import json
import statistics as st
from dataclasses import replace
from pathlib import Path

import torch

from .config import build_parser, load_config
from .plant import FrankaPlant
from .training import get_model, horizon_error

VARIANTS = ("multi_step", "one_step")


def main() -> None:
    cfg = load_config(**vars(build_parser(__doc__).parse_args()))
    torch.set_num_threads(cfg.run.torch_threads)
    plant = FrankaPlant(cfg)

    subs = {
        "multi_step": cfg,
        "one_step": cfg.replace(train=replace(cfg.train, one_step=True)),
    }

    rows: dict = {v: [] for v in VARIANTS}
    for seed in cfg.run.model_seeds:
        held = plant.excite(seed=9_000 + seed, white=False)
        b = torch.as_tensor(held.obs[:, : cfg.train.horizon + 1], dtype=torch.float32)
        u = torch.as_tensor(held.controls[:, : cfg.train.horizon], dtype=torch.float32)
        for name in VARIANTS:
            model = get_model(subs[name], plant, "bilinear", seed, tag=name)
            err = horizon_error(model, b, u, plant.num_joints)
            rows[name].append(err)
            print(f"  seed {seed}  {name:<11} {err:.4f} m", flush=True)

    print()
    print(f"{'variant':<12}{'mean':>9}{'s.d.':>9}{'min':>9}{'max':>9}")
    for name in VARIANTS:
        v = rows[name]
        print(f"{name:<12}{st.mean(v):>9.4f}{st.pstdev(v):>9.4f}"
              f"{min(v):>9.4f}{max(v):>9.4f}")

    paired = [o - m for m, o in zip(rows["multi_step"], rows["one_step"])]
    print(f"\nper-seed gap (one_step - multi_step): "
          f"{[round(x, 4) for x in paired]}")
    print(f"multi-step better on {sum(1 for x in paired if x > 0)}"
          f"/{len(paired)} seeds")

    out = Path(__file__).resolve().parent / cfg.run.out_dir / "open_loop_seeds.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=2))
    print(f"\n-> {out}", flush=True)


if __name__ == "__main__":
    main()
