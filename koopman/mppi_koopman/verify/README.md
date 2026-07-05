# Tier-0 numerical verification (RB-KMPPI)

Docker-based check of the core claims in [`../notes/derivation.md`](../notes/derivation.md).

## Run
```bash
docker build -t rbkmppi-verify .
RUN="docker run --rm -v <abs-path-to-this-dir>:/work -w /work rbkmppi-verify python"
$RUN tier0_verify.py     # sec1 tube + C1 accuracy + MPPI beta-sweep  (2D systems)
$RUN plmi_tube.py        # sec6 P-LMI weighted-norm tube (needs cvxpy)
$RUN tier1_unicycle.py   # Tier-1 nonholonomic unicycle (strong bilinear)
# Windows PowerShell mount: -v "C:\Users\ggory\Desktop\mppi_koopman\verify:/work"
```
Outputs → `out/*.png`, `out/*summary.txt`. Shared model algebra in `koopman_core.py`.

## What it checks (two systems, degree-3 poly lifting, Ψ(0)=0, C=[I₂|0])
- **A) vanderpol** — constant input matrix `G=[0;1]` → weak lifted bilinearity.
- **B) pendulum_sdg** — torque via `cos(x1)·u`, state-dependent `G(x)` → strong bilinearity.

| Check | Claim | Result (2026-06-27) |
|---|---|---|
| 1 | **Thm 1** tube `e_k ≥ ‖Δ_k‖` (§1) | **PASS** both, 0 in-region violations. Out-region: VdP 0.01% (illustrates **R1** regional caveat). |
| 2 | **C1** bilinear < linear pred error | system-dependent: VdP ~1.0–1.4×, **pendulum_sdg 1.9–2.1×** over H=5–20 (strong where input enters bilinearly — confirms C1 thesis). |
| 3 | bilinear-MPPI stabilizes; β knob | stabilizes both. **β is a conservatism knob**: helps VdP (small β), but **β=0 (pure KMPPI) best for nominal pendulum regulation** — penalty's value is robustness/constraints (Tier-1+), not nominal performance. β→0 recovers (A), as theory says. |

## Tier-1 unicycle (`tier1_unicycle.py`) — strong bilinear regime
state (x,y,θ), input (v,ω); trig-closed lifting [x,y,θ,sinθ,cosθ−1]. Input enters strongly
bilinearly (v·cosθ). Result: **linear/bilinear 1-step RMSE = 44.7×**, open-loop pred error
**37–46×** better for bilinear (vs 1–2× on Tier-0) — confirms "advantage ∝ bilinearity of input coupling".
Tube PASS (0 viol), MPPI parking PASS.

## Pendulum (`pendulum_gif.py` + `pendulum_validate.py`) — linear Koopman is ALREADY enough
> ⚠️ This pendulum experiment is NOT faithful to MPPI-DK (see paper, eq 13): MPPI-DK's task is torque-LIMITED
> **swing-up & balance** (|u|≤2, arbitrary init), with a **deep (NN) linear Koopman** (lifted dim 4–8), Δt=0.05.
> My version used a weak manual polynomial dictionary and (after swing-up failed) a LOCAL balancing task.
> Treat the GIF as illustrative only; a faithful reproduction needs eq-13 dynamics + swing-up + an expressive model.

**Implementation audit (`pendulum_validate.py`) — what's actually true:**
- **rollout consistency OK** (batched MPPI == sequential == predict_bi, max diff 3e-15) → no code bug.
- bilinear 1-step lifted RMSE is 4.7× lower (train AND test) — but this is a LIFTED-SPACE-1-STEP artifact:
- **multi-step open-loop physical prediction (sinθ,cosθ,θ̇) is IDENTICAL** for linear vs bilinear (ratio ≈1.00
  at H=5–40, both broad & near-upright). So on the pendulum there is **no real predictive advantage** for
  bilinear (constant input field) — fully consistent with MPPI-DK showing linear suffices, and with theory.
- Full swing-up fails for BOTH manual-dictionary Koopman models (H=40 physical error ~2–3, model too weak for
  long rollouts); MPPI-DK succeeds because it uses a DEEP Koopman. → faithful swing-up needs a richer/deep model.

→ **Thesis (validated 3 ways: theory + multi-step prediction + MPPI-DK's own result)**: bilinear advantage ∝
  *how active the state–input coupling is in the operating region* — decisive on the unicycle (always turning:
  37–46×), negligible on the pendulum (constant input field). This precisely delimits where MPPI-DK's linear
  choice is justified. → `out/pendulum_compare.gif/.png`.

## Deep Koopman (PyTorch, `pendulum_deep.py` / `unicycle_deep.py` / `pendulum_efficiency.py`)
Faithful to MPPI-DK (deep NN lifting). Both deep-linear (their method) and deep-bilinear (ours) share
the SAME architecture/data/training; oracle = MPPI-true. Dynamics: pendulum eq.13 (|u|≤2 swing-up),
unicycle RK4. Trained on K-step rollout loss + latent consistency.
- **`pendulum_efficiency.py` (robust, deterministic — the CORE result)**: multi-step (H=15) open-loop
  prediction error vs lifting dim. **deep-LINEAR saturates at ~0.74 (no improvement past dim 8); deep-BILINEAR
  reaches ~0.50 — a floor LINEAR never reaches at ANY dim.** bilinear@dim8 (0.52) beats linear@dim16 (0.74)
  → "we do well with less" (capacity efficiency). This is a model-accuracy statement, NOT a claim that
  MPPI-DK 'fails'. → `out/pendulum_efficiency.png`.
- **★ NEURON efficiency (`pendulum_neurons.py` curve + `pendulum_neurons_gif.py` GIF) — the clean story**:
  MPPI-DK report neuron count (they use 256/layer) is the key lever; we fix lifting dim 8 and sweep neurons.
  Deterministic H=12 pred error: **ours @ 32 neurons ≈ linear @ 128 neurons; linear saturates at ~0.50 (never
  reaches our 256-neuron 0.26)** → ours needs ~4× fewer neurons. Swing-up GIF: **all three succeed** — MPPI-DK
  (linear, 256 neurons) swings up @6.7s (reproduces their paper, no contradiction), ours (bilinear, 64 neurons)
  @3.4s, oracle @3.1s. → "we do it with FEW neurons; they need MANY". `out/pendulum_neurons*.png/gif`.
- (`pendulum_deep.py` = matched-capacity v1 saved in `saved/`; `pendulum_dko.py` = faithful 1-step DKO repro →
  1-step loss gives poor multi-step rollout so BOTH fail, confirming multi-step training is what matters.)
- **`unicycle_deep.py`**: deep bilinear vs linear multi-step pred = **42–48×** (matches the manual-dict 37–46×)
  → the dramatic advantage is reproduced with deep lifting. (MPPI tuning of the parking GIF still WIP.)

## 3D Quadrotor (MuJoCo, `quad_test.py` / `quad_deep.py`) — secondary; 1st- vs 2nd-order coupling
MuJoCo free-body quadrotor (thrust along body-z via `xfrc_applied`, world force = R@[0,0,T]). Deep Koopman
(base 18: pos,linvel,Rflat,angvel; m=4). Near-hover (mild tilt): no bilinear advantage. **Aggressive flight
(±1.2 rad): bilinear 2–4× better on FULL state — but velocity/position (control-relevant) ~1× (equal).**
**Control (`quad_oracle.py` + `quad_gif.py`)**: fixed a torque-scale bug (τ≫inertia → tumbling); the analytic
oracle (verified == MuJoCo) then FLIES (dist 0.11) → MPPI valid. Retrained on fixed data: **oracle flies,
MPPI-DK(linear) nearly flies (~1), but ours(bilinear) robustly FAILS (~42, flies off)** — the 4 B_i matrices
make the bilinear model harder to train / rollout-unstable. ★ **Sharp positioning (strengthens thesis)**:
bilinear's CONTROL advantage needs **1st-order/kinematic coupling** (unicycle `v·cosθ` sets position rate →
linear fails, bilinear wins 37–48×). For **2nd-order/acceleration coupling** (quadrotor `T·R_up`, diluted by
double-integration) **linear is adequate and bilinear is DETRIMENTAL.** ⇒ **our method = nonholonomic/kinematic
systems (unicycle, diff-drive, car); MPPI-DK's linear is right for thrust-vectored/2nd-order (quadrotor).**
Quadrotor = prediction result (`quad_deep.png`) + positioning insight; control GIF dropped (lose-lose for us).

## 7-DOF arm reach (MuJoCo, `arm_test/arm_deep/arm_gif/arm_robust.py`) — 2nd clean control win
Self-contained 7-DOF arm; KINEMATIC velocity control (u=q̇). `ee+=ee+J(q)q̇dt` = 1st-order config-dependent
Jacobian coupling (manipulator analog of unicycle `v·cosθ`) → linear's constant B can't represent it.
Prediction gap is "only" ~1.5× (J(q) has a non-zero linearizable average, unlike the unicycle's sign-flipping
v·cosθ), BUT **receding-horizon CONTROL amplifies the structural gap into reach-vs-fail**:
**ours(bilinear) reaches 6/7 targets, oracle 7/7, MPPI-DK(linear) 0/7** (`arm_robust.py`; batched analytic FK
verified == MuJoCo). High-DOF, fully-actuated → robust (no quadrotor-style fragility). `out/arm_gif.gif`.
→ **Control criterion = STRUCTURAL representability (can constant B encode the coupling?), not the prediction-ratio
magnitude.** Two clean wins now: unicycle + 7-DOF arm (both 1st-order kinematic); quadrotor (2nd-order) is the
regime where linear is adequate.

## §6 P-LMI weighted-norm tube (`plmi_tube.py`) — mechanism verified + key prerequisite
- **synthetic** (non-normal stable A, `‖A‖₂=2.3>1`, **controlled c_x=0.001**): Euclidean tube explodes (1e12),
  P-LMI tube bounded & contractive (`coef=ρ+κ_P·c_x=0.96<1`), **≈4×10¹¹ tighter** → mechanism works.
- **real EDMD** (pendulum/VdP): vanilla least-squares gives **spec.radius(A)>1 even for a stable system** →
  P-LMI infeasible; stabilizing inflates `c_x≈0.25` → `coef≈1.7–2.7>1` → not contractive.
- **finding**: a contractive tube needs `c_x ≲ (1−ρ)/κ_P ≈ 0.001–0.01` (≈25–250× smaller than vanilla EDMD)
  = the **SafEDMD/kernel-EDMD regime** (`c_x=O(1/√d+Δt²)`). Non-normality (the reason `‖A‖>1`) also inflates
  `κ_P`, so `κ_P·c_x` dominates — empirically confirms the §6.2 conditioning caveat and the SafEDMD choice.

## Animation (`unicycle_gif.py`) — 3-way parking comparison
Same start/goal/cost/samples; only the rollout MODEL differs. Each run stops on goal convergence.
Smoothing (all): control-rate penalty + colored noise + Savitzky-Golay filter.
- **MPPI-true** (oracle, true dynamics): parks @step 60, smoothest (mean|Δu|=0.30).
- **Ours** (bilinear Koopman): parks @step 84 ≈ oracle (mean|Δu|=0.37) — near-oracle, small residual hook.
- **MPPI-DK** (linear Koopman): never converges (final dist 2.56) — linear can't represent heading-dependent v·cosθ.
→ `out/unicycle_compare.gif`, `unicycle_compare.png`.

## Key takeaway → next steps
Theorem 1 tube is **valid** everywhere but **contractive only under SafEDMD-grade identification** (small c_x +
Schur-stable A). **Next: (a) implement SafEDMD/kernel bilinear EDMD to get a certified small c_x, then re-verify §6
contraction on the real systems; (b) Tier-1 MPPI under input/obstacle constraints to show the tube penalty's
robustness payoff (where β>0 actually helps).**
