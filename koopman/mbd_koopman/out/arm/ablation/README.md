# Optimizer ablation (arm, BK model, 5 seeds x 7 targets)

Rerun 2026-07-02 on the quantile-tube stack (fit_tube_constants,
quantile=0.999; beta_e=2e-4). Same trained BK checkpoints (out/arm/models),
same task/cost; only the MBD optimizer settings vary.
Planning seed = seed*10 + target.

| config        | structure                      | reach | strict | final err [m]   | ms/step |
|---------------|--------------------------------|-------|--------|-----------------|---------|
| mppi_equiv    | 1 iter x 800,  sigma 1.2 fixed | 32/35 | 25/35  | 0.029 +/- 0.013 | 18.4    |
| equal_compute | 1 iter x 4000, sigma 1.2 fixed | 32/35 | 28/35  | 0.028 +/- 0.008 | 23.9    |
| no_anneal     | 5 iter x 800,  sigma 1.2 fixed | 34/35 | 32/35  | 0.027 +/- 0.017 | 71.0    |
| (main table)  | 5 iter x 800,  sigma 1.2->0.3  | 35/35 | 35/35  | 0.024 +/- 0.001 | 68.8    |

Seven of eight fixed-sigma failures are marginal stalls at 0.050-0.079 m,
six on the farthest target T7 (the eighth: 0.124 m, target 3 / seed 3);
never divergence -> fixed-sigma precision-floor mechanism. Annealing to
sigma_end=0.3 removes all of them at the same rollout budget as no_anneal.

FR3 replication (out/franka/ablation/): all four configs 35/35 strict
(17/22/64/64 ms/step) -- the tolerance sits above the precision floor
there; scope, not contradiction (see paper/framing.md Sec. 8.3/9).

Reproduce, e.g.:
  cd koopman/mbd_koopman
  python experiments/run_all.py --task arm --methods bk_mbd --device cuda \
    --num-diffusion-steps 1 --sigma-start 1.2 --sigma-end 1.2 --num-samples 800 \
    --output-dir <dir-with-models-symlink>
