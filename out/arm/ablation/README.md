# Optimizer ablation (arm, BK model, 5 seeds x 7 targets)

Same trained BK checkpoints (out/arm/models), same task/cost/tube (beta_e=2e-4);
only the MBD optimizer settings vary. Planning seed = seed*10 + target.

| config        | structure                      | reach   | final err [m]   | ms/step |
|---------------|--------------------------------|---------|-----------------|---------|
| mppi_equiv    | 1 iter x 800,  sigma 1.2 fixed | 31/35   | 0.031 +/- 0.012 | 17.4    |
| equal_compute | 1 iter x 4000, sigma 1.2 fixed | 32/35   | 0.028 +/- 0.009 | 22.1    |
| no_anneal     | 5 iter x 800,  sigma 1.2 fixed | 33/35   | 0.027 +/- 0.009 | 64.9    |
| (main table)  | 5 iter x 800,  sigma 1.2->0.3  | 35/35   | 0.024 +/- 0.004 | 62.3    |

All failures are marginal stalls (0.050-0.066 m, mostly farthest target T7),
not divergence -> consistent with the fixed-sigma precision-floor mechanism;
annealing to sigma_end=0.3 removes them (all trials strict < 0.025).

Reproduce, e.g.:
  python experiments/run_all.py --task arm --methods bk_mbd --device cuda \
    --num-diffusion-steps 1 --sigma-start 1.2 --sigma-end 1.2 --num-samples 800 \
    --output-dir <dir-with-models-symlink>
