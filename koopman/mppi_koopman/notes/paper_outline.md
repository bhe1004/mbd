# Paper Outline — Bilinear Koopman MPPI

## Title & Abstract (draft v1)

**Title (primary):** *When Bilinear Koopman Models Help Sampling-Based Control: Path-Integral MPC with Lifted Bilinear Dynamics*
**Alt:** *Bilinear Koopman Model Predictive Path Integral Control for Kinematically-Coupled Systems* · *RB-KMPPI: Bilinear Koopman MPPI and When It Helps*

**Abstract (~215 words):**
> Model Predictive Path Integral (MPPI) control is sampling-based and therefore imposes no convexity
> requirement on its rollout model—yet existing Koopman-accelerated MPPI restricts itself to *linear* lifted
> dynamics, the very restriction that Koopman model predictive control adopts only to keep its optimization
> convex. We remove this gratuitous restriction and roll out the *bilinear* Koopman model, the provably correct
> lifted class for control-affine systems, at the same matrix–vector compute regime as linear models. To make
> the controller model-error-aware, we inject the Koopman proportional error bound as an online error-tube
> penalty, and we establish a free-energy practical-stability guarantee whose ultimate bound vanishes with the
> model error. Our central finding is a precise criterion for *when* the bilinear term matters: it is decisive
> exactly when the state–input coupling is first-order and configuration-dependent—such as the heading rotation
> of a nonholonomic vehicle or the Jacobian of a manipulator—where a constant input matrix structurally cannot
> represent the dynamics. Across four systems, bilinear Koopman MPPI cleanly succeeds where linear deep-Koopman
> MPPI fails on a nonholonomic unicycle and a 7-DOF manipulator reach (6/7 vs. 0/7 targets), reaches parity on a
> quadrotor whose coupling is second-order, and swings up a pendulum with four times fewer lifting neurons. The
> results delineate exactly when lifted bilinear dynamics should be used in sampling-based predictive control.

(other working titles:)
- "Bilinear Koopman Path-Integral Control: When Lifted Bilinear Dynamics Help Sampling-Based MPC"

Anchors: [[formulation_AB]], [[derivation]], [[lit_digest]], `verify/README.md` (all experiments), `ref/`.
Target: ISA Transactions / RA-L / T-RO style. Template `template/` is a DIFFERENT paper — start fresh.

---

## One-sentence thesis
MPPI is sampling-based ⇒ needs no convexity ⇒ it can roll out the *correct* (bilinear) Koopman class for free —
and the bilinear term is **decisive exactly when the state–input coupling is first-order/kinematic and
configuration-dependent** (nonholonomic vehicles, manipulator Jacobian), where linear Koopman (MPPI-DK)
**structurally** cannot represent the dynamics; it is unnecessary for second-order/acceleration coupling.

## Contributions (claims)
- **C1** First use of a **bilinear** Koopman model as the MPPI rollout. Rationale: sampling ⇒ no convexity ⇒
  no need to restrict to linear (which Koopman-MPC does to keep the QP convex). Same GPU matvec regime as
  linear MPPI-DK (`m+1` matvecs/step).
- **C2** **Error-aware** MPPI: inject the Koopman *proportional error bound* `‖r‖≤c_x‖z‖+c_u‖u‖` as an
  online **error-tube penalty / CVaR** (risk-aware), making risk-neutral MPPI model-error-aware.
- **C3** **Theory**: error-tube recursion (Thm 1), free-energy practical-stability with ultimate bound
  `∝(c_x,c_u)` (no value-function smoothness needed), P-LMI weighted-norm tube + the SafEDMD prerequisite.
- **C4** **The geometry of when it helps** (the differentiator): a precise criterion — bilinear wins control
  iff the coupling is 1st-order/kinematic & not representable by a constant input matrix. Validated by TWO
  clean wins (unicycle, 7-DOF arm) and a negative control case (quadrotor), plus a neuron-efficiency result
  (pendulum) consistent with and non-contradictory to MPPI-DK.

---

## RA-L structure (5 sections, **8 pages** incl. refs) — page budget in [brackets]
> RA-L letter = up to 8 pages. Budget: **I Intro+RW ~1.25 · II Background ~1 · III Method+Theory ~2 ·
> IV Experiments ~2.75 · V Conclusion ~0.4 · refs ~0.6.** Still experiment-led: stars = the TWO control wins
> (unicycle, 7-DOF arm) + the *when-it-helps* criterion. With 8 pages the theory can be a bit fuller — **tube
> Proposition + stability Theorem (proof sketch) + one line each on CVaR and P-LMI/SafEDMD** — but full §1–§8
> proofs still go to a journal extension / supplementary.

## I. Introduction (related work woven in)  [~1.25 pg]
- Fast MPC for aggressive tasks; nonlinear MPC expensive; **MPPI** sampling-based, GPU-parallel, arbitrary
  dynamics (IT form), but **rollout model is the bottleneck**, risk-neutral, no stability guarantees.
- **Koopman** lifts to (near-)linear; **bilinear** is the provably correct minimal class (Iacob–Tóth);
  **proportional error bound** (Strässer) → guarantees. Two literatures don't cite each other.
- Closest prior **MPPI-DK** (linear deep Koopman, speed-only) — wrong class, ignores error.
- **Key idea**: MPPI needs no convexity ⇒ linear restriction is gratuitous ⇒ use bilinear at the same compute
  regime, inject the certified error bound, **and characterize precisely when it matters.**
- Related work folded in here (1 short paragraph each: MPPI variants; Koopman control/SafEDMD; MPPI-DK).
- Contributions C1–C4 + forward-pointer to **Fig. 1** (unicycle + 7-DOF arm wins).

## II. Problem & Background  [~1 pg]
- Control-affine `ẋ=f+Gu`, discrete `x⁺=F(x,u)`, receding-horizon cost.
- **MPPI** (IT): proposal `q=N(u,Σ)`, free energy, softmax update `u←u+Σ_k w_k ε_k`.
- **Koopman**: dictionary `Ψ`, `Ψ(0)=0`; linear `Ψ⁺≈AΨ+Bu` vs **bilinear** `Ψ⁺≈AΨ+B₀u+Σu_iB_iΨ`; deep
  multi-step identification; **proportional bound** `‖r‖≤c_x‖Ψ‖+c_u‖u‖` (SafEDMD/kernel give `c_x,c_u`).

## III. Method — Bilinear Koopman MPPI (incl. compact theory)  [~2 pg]
- **A. Bilinear rollout**: `ẑ⁺=M(u)ẑ+B₀u`, `M(u)=A+Σu_iB_i` — matvec-only ⇒ same GPU regime as linear
  (`m+1` matvecs/step). **Algorithm 1** (MPPI loop with bilinear rollout + tube penalty).
- **B. Identification (recipe)**: deep Koopman, **multi-step** rollout loss; **B₁=0 init** (bilinear ⊇ linear
  ⇒ never worse); coherent-control data (match deployment).
- **C. Error-tube injection**: tube `e_{k+1}=(‖M(v_k)‖+c_x)e_k+c_x‖ẑ_k‖+c_u‖v_k‖`; risk cost `S̃=Ŝ+βΣe_k`.
- **★ D. When it helps (the criterion — the differentiator)**: tracked-state increment
  `Δ(tracked)=(config-dependent input map)·u·dt`. If that map is **1st-order & configuration-dependent**
  (heading `R(θ)`, Jacobian `J(q)`), a **constant** `B` cannot represent it ⇒ bilinear is *structurally*
  required. If 2nd-order (enters acceleration), constant `B` is adequate. **State as a short Proposition.**
- **E. Compact theory** (1 Prop + 1 Thm, proofs sketched; full → extended version):
  - **Prop 1 (tube)**: `‖z_k−ẑ_k‖≤e_k` (1-line induction).
  - **Thm 1 (stability)**: free-energy bound `|F_real−F_nom|≤ΔS_max` ⇒ practical asymptotic stability,
    ultimate bound `∝(c_x,c_u,λ,K^{-1/2})` (no value-function smoothness). Proof sketch; cite extended.
  - One sentence: contractive tube needs SafEDMD-grade small `c_x` (P-LMI detail → extended version).

## IV. Experiments  [~2.75 pg — THE CORE]  [all Docker-reproducible, deep Koopman/PyTorch]
- **Setup**: deep Koopman (NN lifting), MPPI on GPU/CPU, baselines = **oracle** (true dynamics MPPI, upper bound)
  and **MPPI-DK** (linear deep Koopman, their method). All share architecture/data/training for fairness.
> **Trimmed per RA-L (experiment-led):** core = the two control WINS + efficiency + tube validation.
> Quadrotor demoted to a one-line *boundary* remark (keeps the "when it helps" criterion honest), figure → supplementary.

- **A. Headline — where bilinear WINS (control) [2 figs, most space]:**
  - **Unicycle** (2D nonholonomic, `v·cosθ`): ours reaches/parks ≈ oracle; **MPPI-DK fails** (drives off). 37–48× ee-prediction. *Fig 1: 3-panel parking + prediction inset.*
  - **7-DOF arm reach** (Jacobian `J(q)q̇`): **ours 6/7 targets, oracle 7/7, MPPI-DK 0/7** (robust across targets). High-DOF, fully-actuated → robust. *Fig 2: 3-panel arm reach.*
- **B. Efficiency (no contradiction with MPPI-DK):**
  - **Pendulum swing-up** (eq.13, |u|≤2): **all three swing up** — ours (bilinear, 64 neurons) ≈ oracle,
    **MPPI-DK (linear) succeeds at 256 neurons** (reproduces their paper). Neuron-efficiency curve:
    ours @32 ≈ linear @128; linear saturates. "4× fewer neurons." *Fig 3: neuron curve + swing-up snapshots.*
- **C. Theory / tube validation:** tube `e_k` upper-bounds true error (0 in-region violations; out-of-region =
  regional caveat). One line: contractive tube needs SafEDMD-grade `c_x` (P-LMI synthetic in supplementary).
- **D. Summary map (small table)** + **boundary remark**: criterion table {coupling order → control outcome}.
  *One sentence*: "For second-order/acceleration coupling (e.g. a quadrotor's thrust `T·R_up`) a constant input
  matrix already suffices and we observe no advantage — a quadrotor study is in the supplementary." (drop entirely
  if space-tight; otherwise this honest boundary sharpens the criterion.)

## V. Conclusion (discussion folded in)  [~0.4 pg]
- **Summary**: MPPI needs no convexity ⇒ roll out the correct *bilinear* Koopman class; it is **decisive
  precisely for first-order kinematic, configuration-dependent coupling** (unicycle, 7-DOF arm — linear
  structurally fails) and unnecessary for 2nd-order coupling (quadrotor — parity). Plus error-tube robustness,
  free-energy practical stability, and a neuron-efficiency edge that is consistent with (not contradicting) MPPI-DK.
- **When to use / recipe (1–2 sentences)**: nonholonomic/kinematic-coupling systems; deep Koopman + multi-step
  loss + B₁=0 init + coherent data; SafEDMD for certified small `c_x`.
- **Limitations (1 sentence)**: `(m+1)×` matvecs at high `m`; deep Koopman lacks full error theory; unstable
  underactuated systems are fragile for any Koopman-MPPI.
- **Future**: kernel/SafEDMD bilinear for certified `c_x`; hardware.

---

## Figure / asset list (have → `verify/saved/`, `verify/out/`)
1. **Unicycle 3-panel** parking (ours/oracle/MPPI-DK) — `unicycle_compare.gif/png` ✅
2. **7-DOF arm 3-panel** reach + 6/7 vs 0/7 table — `arm_gif.*`, `arm_robust.py` ✅
3. **Pendulum** neuron-efficiency curve + swing-up GIF — `pendulum_neurons*.*` ✅
4. **Quadrotor** prediction curve (boundary case) — `quad_deep.png` ✅
5. **Tube validation** (`e_k ≥ ‖Δ_k‖`) — `tier0_verify.png` ✅
6. **§6 P-LMI** synthetic contraction — `plmi_tube.png` ✅
7. **Summary map table** (coupling order vs win) — to make.

## TODO before submission
- [ ] Robustness: a couple of training seeds for unicycle/arm "linear-fails" claim (arm already 7 targets).
- [ ] Make `c_x,c_u` from SafEDMD/kernel (not residual quantile) → close §6 contraction on a real system.
- [ ] (optional) hardware or higher-fidelity sim.
- [ ] Port outline → `template/`-style IEEE tex (new file, not the constraint-manifold main.tex).
- [ ] Tighten Theory: A2 `Δ_opt` bound (Remark 5 free-energy route), weighted-norm stability constants.
