# BK-MBD Project Roadmap

This project implements Model-Based Diffusion with Koopman rollout models under
`/home/home/playground/mbd/koopman/mbd_koopman`.

The immediate goal is to port the comparison structure of
`koopman/mppi_koopman` from MPPI to MBD:

1. `vanilla_mbd_true`: MBD with true dynamics rollout.
2. `dk_mbd`: MBD with linear deep Koopman rollout.
3. `bk_mbd`: MBD with bilinear deep Koopman rollout and error-tube cost.

The central question is:

```text
Can bilinear Koopman rollout replace true-dynamics rollout inside MBD,
while linear deep Koopman MBD fails on first-order coupled systems?
```

## Source Projects

### Bilinear Koopman MPPI

Reference path:

```text
/home/home/playground/mbd/koopman/mppi_koopman
```

Use this project for:

- experiment settings,
- unicycle and arm tasks,
- training data generation style,
- linear and bilinear deep Koopman model definitions,
- multi-step Koopman training loss,
- zero initialization of bilinear input matrices,
- error tube propagation,
- success metrics and figures.

Important reference files:

```text
koopman/mppi_koopman/template/main.tex
koopman/mppi_koopman/verify/koopman_core.py
koopman/mppi_koopman/verify/unicycle_multiseed_v2.py
koopman/mppi_koopman/verify/arm_multiseed_v2.py
koopman/mppi_koopman/verify/tube_deep.py
```

The core Koopman rollout models are:

```text
Linear DK:
  z_next = A z + B u

Bilinear BK:
  z_next = A z + B0 u + sum_i u_i B_i z
         = M(u) z + B0 u

  M(u) = A + sum_i u_i B_i
```

The deep Koopman models must be trained with the same multi-step loss used in
the Bilinear Koopman MPPI paper:

```text
L = sum_h || C zhat_h - x_h ||^2
  + gamma sum_h || zhat_h - Psi_theta(x_h) ||^2
```

For the bilinear model, each `B_i` must be initialized to zero so training
starts from the linear Koopman model and adds bilinear terms only when useful.

### Model-Based Diffusion

Reference path:

```text
/home/home/playground/mbd/model-based-diffusion
```

Use this project for:

- diffusion-style population optimization over control sequences,
- cost-based weight computation,
- reverse diffusion / denoising update structure.

Important reference files:

```text
model-based-diffusion/mbd/planners/mbd_planner.py
model-based-diffusion/mbd/planners/path_integral.py
model-based-diffusion/mbd/utils.py
```

For this project, do not import the full Brax/JAX environment structure. Rebuild
the planner around the `mppi_koopman` tasks so the comparison remains aligned
with the Bilinear Koopman MPPI experiments.

## Scope

### In Scope

- Unicycle parking.
- 7-DOF arm end-effector reaching.
- MBD with true dynamics rollout.
- MBD with linear deep Koopman rollout.
- MBD with bilinear deep Koopman rollout and tube cost.
- Shared MBD update rule across all three methods.
- Shared training data, architecture, horizon, sample count, action bounds, and
  task cost between `dk_mbd` and `bk_mbd`.

### Out of Scope for the First Implementation

- Pendulum neuron efficiency experiment.
- Quadrotor boundary experiment.
- OOD penalty.
- Risk-sensitive variants.
- Hardware experiment.
- Full SafEDMD-certified bounds.

OOD penalty can be added later as an ablation after the three-method comparison
is working.

## Method Definitions

### 1. Vanilla MBD with True Dynamics

Name:

```text
vanilla_mbd_true
```

Rollout:

```text
x_next = F_true(x, u)
```

Cost:

```text
J = J_task(x_0:T, u_0:T-1)
```

This is the oracle rollout reference. It is not a deployable learned-model
controller, but it shows what MBD can do when rollout dynamics are exact.

### 2. Linear Deep Koopman MBD

Name:

```text
dk_mbd
```

Lift:

```text
z = Psi_theta(x)
```

Rollout:

```text
z_next = A z + B u
x_hat = C z
```

Cost:

```text
J = J_task(x_hat_0:T, u_0:T-1)
```

This is the MBD version of MPPI-DK. It should expose the same structural
limitation as in the MPPI paper: a constant input matrix cannot represent a
configuration-dependent input gain such as unicycle heading rotation or an arm
Jacobian.

### 3. Bilinear Koopman MBD

Name:

```text
bk_mbd
```

Lift:

```text
z = Psi_theta(x)
```

Rollout:

```text
z_next = A z + B0 u + sum_i u_i B_i z
x_hat = C z
```

Tube:

```text
e_next = (||M(u)|| + c_x) e + c_x ||z|| + c_u ||u||
```

Cost:

```text
J_tilde = J_task(x_hat_0:T, u_0:T-1) + beta_e sum_t e_t
```

This is the proposed method. It uses the same bilinear Koopman model class and
error-tube idea as Bilinear Koopman MPPI, but replaces the MPPI update with an
MBD score / denoising update.

### 4. Structure-Informed Linear Koopman MBD (added baseline)

Name:

```text
dk_mbd_split
```

Purpose: reproduce the conditions of the MPPI-DK paper (Hao et al.,
"Accelerating Sampling-Based Control via Learned Linear Koopman Dynamics").
In all MPPI-DK experiments the input enters state-independently and the
first-order coupling is handled analytically outside the learned model
(surface vehicle: learned body-frame dynamics + analytic rotation
integration; quadruped: body-frame acceleration inputs). `dk_mbd_split`
applies the same decomposition to our tasks:

```text
unicycle: [dp_body; dtheta] = G u fitted by least squares,
          rollout composes analytically with R(theta) (SE(2)).
arm:      linear deep Koopman learns q-only dynamics,
          cost applies analytic forward kinematics to decoded q.
```

Expected outcome: near-oracle success. Together with `dk_mbd` (0% success,
full coupling learned) this shows the linear model class only works when the
first-order coupling is removed analytically, whereas `bk_mbd` matches the
oracle without any analytic structure.

Note: the mppi_koopman paper (`koopman/mppi_koopman/template/main.tex`) does
NOT include this split baseline. Its MPPI-DK baseline is the full-lift linear
DK — the same construction as our `dk_mbd` (shared deep lifting, linear vs
bilinear lifted model, no analytical model supplied) — and it fails there the
same way (0% parking, arm reaches none). From the original MPPI-DK paper only
method-specific hyperparameters are adopted there. `dk_mbd_split` is new to
BK-MBD and goes one step beyond both the MPPI-DK paper and the mppi_koopman
comparison.

## MBD Update

All methods must use the same control-sequence optimizer.

Decision variable:

```text
U = u_0:T-1
```

At diffusion step `s`, sample candidate control sequences:

```text
U_k = U + sigma_s eps_k
```

Roll out each candidate with the method-specific rollout model, compute cost,
and form weights:

```text
w_k = exp(-J_k / alpha_s) / sum_j exp(-J_j / alpha_s)
```

Estimate the score:

```text
score = (sum_k w_k U_k - U) / sigma_s^2
```

Update:

```text
U = U + eta_s score + sqrt(2 eta_s) xi
```

Implementation note:

The original MBD code often reduces algebraically to a weighted-mean update.
This project keeps the explicit score update as the default because it matches
the BK-MBD formulation, with the step size tied to the noise level:

```text
eta_s = eta * sigma_s^2
```

so that `eta = 1` without Langevin noise reproduces the weighted-mean update
exactly. A fixed absolute step (`eta_relative = false`) was found to stall on
the arm task (the effective step `eta / sigma_s^2` becomes negligible at large
sigma), so it is kept only as a debug option, along with `weighted_mean`.

## Fairness Rules

The comparison is only meaningful if the differences are controlled.

All methods share:

- task definition,
- initial states and goals,
- planning horizon,
- number of candidate samples,
- diffusion schedule,
- action bounds,
- task cost,
- closed-loop receding-horizon execution loop,
- random seeds where applicable.

`dk_mbd` and `bk_mbd` additionally share:

- training dataset,
- lift architecture,
- lifted dimension,
- decoder convention,
- multi-step training horizon,
- optimizer,
- training budget,
- batch size,
- validation split.

They differ only in the lifted dynamics class:

```text
dk_mbd:
  z_next = A z + B u

bk_mbd:
  z_next = A z + B0 u + sum_i u_i B_i z
```

The `bk_mbd` final method additionally includes the error-tube cost. If needed,
`bk_mbd_no_tube` may be logged internally, but it is not part of the first main
comparison table.

## Proposed Directory Structure

```text
koopman/mbd_koopman/
├── README.md
├── plan/
│   └── plan.md
├── bk_mbd/
│   ├── __init__.py
│   ├── config.py
│   ├── models.py
│   ├── train.py
│   ├── rollout.py
│   ├── mbd.py
│   ├── costs.py
│   ├── tube.py
│   └── metrics.py
├── envs/
│   ├── __init__.py
│   ├── unicycle.py
│   └── arm.py
├── experiments/
│   ├── run_unicycle.py
│   ├── run_arm.py
│   ├── run_all.py
│   └── make_figs.py
└── out/
    ├── unicycle/
    └── arm/
```

## Module Responsibilities

### `bk_mbd/config.py`

Central dataclasses for:

- task config,
- Koopman training config,
- MBD planner config,
- tube config,
- output paths.

### `bk_mbd/models.py`

Define:

- shared deep lift `Psi_theta`,
- linear deep Koopman model,
- bilinear deep Koopman model,
- decoder `C`.

Preferred decoder convention for the first implementation:

```text
z = [x, learned_features]
C z = z[:state_dim]
```

This avoids learning a fragile decoder and matches the practical style of the
existing verification scripts.

### `bk_mbd/train.py`

Implement:

- dataset window construction,
- multi-step rollout loss,
- validation loss,
- model checkpoint save/load,
- training logs.

The same training function should train both linear DK and bilinear BK by
switching the model class.

### `bk_mbd/rollout.py`

Implement batched rollouts:

- true dynamics rollout,
- linear Koopman rollout,
- bilinear Koopman rollout,
- bilinear Koopman rollout with tube.

The rollout backend should use Torch tensors for the candidate/sample dimension:

```text
U:  (K, T, action_dim)
z0: (lift_dim,) or (K, lift_dim)
xs: (K, T + 1, state_dim)
zs: (K, T + 1, lift_dim)
e:  (K, T + 1)
```

The horizon scan remains sequential, but the `K` MBD candidates must be
processed in one batched Torch rollout. NumPy rollout helpers may remain for
small smoke tests and debugging, but experiment code should use the Torch batch
path.

### `bk_mbd/mbd.py`

Implement the common MBD planner:

- candidate control sampling,
- action clipping,
- rollout call,
- cost weighting,
- score estimate,
- diffusion update,
- receding-horizon control loop.

The planner should accept a rollout backend rather than hard-coding the method.

### `bk_mbd/costs.py`

Implement task costs:

- unicycle pose parking cost,
- arm end-effector reaching cost,
- terminal cost,
- control penalty.

### `bk_mbd/tube.py`

Implement:

- residual computation,
- fitting of `c_x, c_u`,
- scalar tube propagation.

For the first implementation, fitted constants are empirical proxies, matching
the current status of the MPPI paper.

> **Update (2026-07-02):** the experiment runners fit `(c_x, c_u)` at
> 99.9%-quantile coverage (`fit_tube_constants`) instead of the reference's
> worst-case linprog. On the Franka FR3 dataset the rest-to-motion
> transient outliers inflate a worst-case `c_x` by ~40x and the tube then
> dominates planning (BK-MBD stalls 0.23 m from the target); quantile
> coverage restores the regularizing role. Unicycle/arm results are
> unchanged under the switch. See paper/framing.md Sec. 6.

### `bk_mbd/metrics.py`

Implement:

- success rate,
- final error,
- total cost,
- planning time per step,
- Koopman open-loop prediction error,
- tube statistics.

### `envs/unicycle.py`

Replicate the unicycle task from `mppi_koopman/verify`.

Required functionality:

- true dynamics,
- dataset generation,
- start-goal cases,
- task cost,
- success criterion,
- plotting helpers.

### `envs/arm.py`

Replicate the 7-DOF arm task from `mppi_koopman/verify`.

Required functionality:

- true dynamics or MuJoCo stepping wrapper,
- forward kinematics,
- target set,
- task cost,
- success criterion,
- plotting helpers.

## Experiment Roadmap

### Milestone 1: Unicycle End-to-End

Goal:

```text
Run vanilla_mbd_true, dk_mbd, and bk_mbd on the unicycle task.
```

Steps:

1. Port unicycle dynamics and task settings.
2. Implement common MBD planner using true dynamics rollout.
3. Verify `vanilla_mbd_true` can park the unicycle.
4. Train linear DK with multi-step loss.
5. Train bilinear BK with multi-step loss and zero-initialized `B_i`.
6. Fit BK tube constants.
7. Run `dk_mbd` and `bk_mbd`.
8. Save trajectories, summary CSV, and a paper-style figure.

Expected qualitative result:

```text
vanilla_mbd_true: high success
dk_mbd: fails or drifts because heading-dependent input gain is lost
bk_mbd: approaches vanilla_mbd_true behavior
```

### Milestone 2: Arm End-to-End

Goal:

```text
Run the same three-method comparison on the 7-DOF arm reaching task.
```

Steps:

1. Port arm dynamics/FK/task settings.
2. Verify `vanilla_mbd_true` reaches targets.
3. Train linear DK and bilinear BK on the same data.
4. Fit BK tube constants.
5. Run all targets and seeds.
6. Save final end-effector errors and figures.

Expected qualitative result:

```text
vanilla_mbd_true: reaches targets
dk_mbd: large final EE error because constant input matrix cannot model J(q)
bk_mbd: lower final EE error, possibly seed-dependent
```

### Milestone 3: Multi-Seed Tables

Goal:

```text
Match the reporting style of the Bilinear Koopman MPPI paper.
```

Unicycle:

```text
5 training seeds x 4 start-goal pairs = 20 trials
```

Arm:

```text
5 training seeds x 7 targets = 35 trials
```

Report:

- success rate,
- final error mean and standard deviation,
- planning time per control step,
- Koopman prediction error,
- tube statistics for BK-MBD.

### Milestone 4: Figures

Generate:

- unicycle trajectory comparison,
- arm trajectory / end-effector comparison,
- arm final error bar plot,
- optional open-loop prediction error plot.

Figures should be saved under:

```text
koopman/mbd_koopman/out/
```

## Output Format

Each experiment should write a `summary.csv` with at least:

```text
task
method
seed
case_id
success
final_error
total_cost
planning_time_ms
train_loss
val_multistep_error
koopman_open_loop_error
tube_cost
max_tube
```

For methods where a field does not apply, use an empty value or `nan`.

Recommended output tree:

```text
out/
├── unicycle/
│   ├── summary.csv
│   ├── trajectories.npz
│   ├── models/
│   └── figures/
└── arm/
    ├── summary.csv
    ├── trajectories.npz
    ├── models/
    └── figures/
```

## Implementation Order

1. Create package skeleton.
2. Port unicycle environment.
3. Implement common MBD planner with true dynamics rollout.
4. Validate `vanilla_mbd_true` on one unicycle case.
5. Implement linear and bilinear deep Koopman models.
6. Implement multi-step training.
7. Train/evaluate DK and BK open-loop on unicycle.
8. Implement BK tube fitting and propagation.
9. Run `dk_mbd` and `bk_mbd` on one unicycle case.
10. Expand to unicycle multi-seed.
11. Port arm task.
12. Run arm single-seed smoke test.
13. Expand to arm multi-seed.
14. Generate summary tables and figures.

## Acceptance Criteria

The first complete version is acceptable when:

1. `vanilla_mbd_true`, `dk_mbd`, and `bk_mbd` run from the same experiment
   script for unicycle.
2. DK and BK are trained from the same dataset and training config.
3. BK uses zero-initialized bilinear matrices and multi-step loss.
4. BK-MBD uses bilinear rollout and tube-augmented cost.
5. Results are saved to `out/unicycle/summary.csv`.
6. A trajectory figure comparing the three methods is saved.
7. The same pipeline can run on at least one arm target.

The project should not move to additional ablations until these criteria are
met.
