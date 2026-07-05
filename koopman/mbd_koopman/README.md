# BK-MBD

BK-MBD ports the Bilinear Koopman MPPI comparison to Model-Based Diffusion.

The comparison covers four methods:

```text
vanilla_mbd_true  MBD + true dynamics rollout (oracle)
dk_mbd            MBD + linear deep Koopman rollout (full coupling learned)
dk_mbd_split      MBD + linear model under MPPI-DK's own conditions
                  (first-order coupling handled analytically)
bk_mbd            MBD + bilinear deep Koopman rollout + error-tube cost
```

## Baseline relationships

Two different "linear Koopman baselines" exist in this line of work; keeping
them apart matters:

- `dk_mbd` matches the **MPPI-DK baseline of the mppi_koopman paper**
  (`koopman/mppi_koopman/template/main.tex`): the full lifted state, including
  the first-order coupling (heading rotation / forward kinematics), is
  learned, and no analytical model is supplied. It fails in both projects
  (0% success), because a constant input matrix cannot represent a
  configuration-dependent input gain.
- `dk_mbd_split` reproduces the conditions of the **original MPPI-DK paper**
  (Hao et al., "Accelerating Sampling-Based Control via Learned Linear
  Koopman Dynamics", arXiv:2603.05385), where in every experiment the input
  enters state-independently and the coupling is handled analytically
  (surface vehicle: analytic rotation integration; quadruped: body-frame
  inputs). Under these conditions the linear model succeeds (100%),
  consistent with that paper's reported results. **This split baseline does
  not exist in the mppi_koopman paper; it is an addition of this project.**

Multi-seed result summary (unicycle 20 trials / arm 35 trials / Franka FR3
in MuJoCo 35 trials; FR3 column is the strict 2.5 cm success rate):

```text
vanilla_mbd_true  100% / 100% / 100%   (oracle; FR3: 4.1 s per plan step)
dk_mbd              0% /   0% /  20%   (linear, coupling learned)
dk_mbd_split      100% / 100% / 100%   (linear, coupling analytic - needs structure)
bk_mbd            100% / 100% / 100%   (bilinear, coupling learned - no structure;
                                        FR3: 64 ms per plan step, 64x oracle)
```

The FR3 task uses the vendored MuJoCo Menagerie FR3 + Franka Hand model
(`envs/assets/franka_fr3/`), a gravity-compensated joint-velocity servo
interface, and the TCP between the fingertips as the controlled point.
Error-tube constants are fitted at 99.9%-quantile coverage (see
`paper/framing.md` Sec. 6 for why worst-case fitting fails on FR3 data).
Replay any saved FR3 run in the viewer with
`python experiments/view_franka.py --method bk_mbd --target-id 0`.

The implementation roadmap is fixed in:

```text
plan/plan.md
```

This folder is intentionally independent from the source projects. Code in this
folder may use their experiment settings and algorithms, but all BK-MBD
implementation and outputs should live under `koopman/mbd_koopman`.

