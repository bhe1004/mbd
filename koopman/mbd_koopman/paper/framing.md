# BK-MBD: Bilinear Koopman Rollouts for Model-Based Diffusion

**Paper framing document — narrative, core mathematics, experimental evidence,
and claim boundaries.**

> 이 문서는 논문 초안 작성의 기준 문서다. 서술은 논문에 그대로 옮길 수 있도록
> 영어로, 주장 경계와 주의사항은 명시적으로 적는다. 모든 수치는
> `out/{unicycle,arm}/` 의 실제 실험 결과와 일치한다 (2026-07-02 기준).

> ⚠️ **논문 인용 금지 (2026-07-03):** 이 문서의 "published BK-MPPI numbers
> report 14/35" 언급(§1, §8.3)은 논문에 넣지 말 것. 해당 arm 결과는 BK-MPPI
> 논문(v1)에서 **드롭되어 출판되지 않으므로** 인용할 대상이 존재하지 않는다.
> 두 논문은 무인용 동시 제출 후 camera-ready에서 상호 인용하는 전략이며,
> optimizer ablation은 내부 통제 비교(MPPI operating point, 32/35)로만
> 서술한다 (2026-07-03 main.tex §IV "What the Anneal Contributes" 반영 완료).

---

## 1. One-paragraph framing

The per-step cost of sampling-based control is dominated by the rollout
model: at every control step, hundreds of candidate control sequences must be
propagated over the horizon. Koopman lifting replaces this propagation with
matrix–vector products, decoupling rollout cost from dynamics complexity — but
a **linear** lift acts on the input through a single constant matrix, and
therefore cannot represent the **configuration-dependent input gain** that
first-order coupled robots exhibit (a heading rotation $R(\theta)$ for a
vehicle, a Jacobian $J(q)$ for an arm). We show this failure is structural and
total in closed loop (0/55 trials across two tasks), that it can be repaired
only by *hand-supplying* the coupling analytically (a "split" model, 55/55) —
which is inadmissible in the model-free setting — and that a **bilinear** lift
learns the coupling from data at the same rollout cost class and matches the
true-dynamics oracle (55/55). Embedded in Model-Based Diffusion, whose
annealed multi-iteration update contains the MPPI update as its
single-iteration, fixed-noise special case, the bilinear rollout additionally
resolves the high-DOF reliability problem that bilinear-Koopman MPPI left
open: on a 7-DOF reaching task it succeeds in 35/35 trials where our
controlled MPPI-equivalent reaches 31/35 and the published BK-MPPI numbers
report 14/35.

---

## 2. Problem setup

Discrete-time control-affine system with unknown dynamics:

$$
s_{t+1} = f(s_t, u_t), \qquad s_t \in \mathcal{X} \subset \mathbb{R}^n,\quad
u_t \in \mathcal{U} = [u_{\min}, u_{\max}]^m .
$$

We assume the *model-free, real-time* setting: $f$ is unknown or too costly
to integrate at the sampling rate; a true-dynamics rollout is available only
as an **oracle** reference, not as a deployable controller. Only interaction
data $\mathcal{D} = \{(s_i, u_i, s_i^+)\}$ is available.

Receding-horizon control: at every step, given the current state $s_t$,
choose a control sequence $U = (u_0, \dots, u_{T-1}) \in \mathbb{R}^{mT}$
minimizing a task cost over the predicted trajectory,

$$
J(U; s_t) \;=\; \sum_{k=1}^{T} c\big(\hat{s}_k, u_{k-1}\big) + \phi(\hat{s}_T),
\qquad \hat{s}_0 = s_t,\;\; \hat{s}_{k+1} = \hat{f}(\hat{s}_k, u_k),
$$

apply the first control $u_0$ to the true system, and replan. The decision
space has dimension $mT$ (arm: $7 \times 15 = 105$). The rollout model
$\hat f$ is the design choice under study.
(The running cost $c$ may additionally depend on the previously *executed*
control through a slew term $\|u_0 - u_{\text{prev}}\|^2$, as in the unicycle
task; this only shifts $J$ by a $U$-dependent constant structure and does not
affect anything below.)

**Tracked output and Koopman state.** Each task defines observable features
$b(s)$ used both as the Koopman training state and in the cost:

- unicycle: $b = [x, y, \sin\theta, \cos\theta - 1] \in \mathbb{R}^4$,
- arm: $b = [q, \mathrm{ee}(q)] \in \mathbb{R}^{10}$ (joints + end-effector).

---

## 3. Model-Based Diffusion over control sequences

### 3.1 Target distribution and annealed smoothing

Define the Boltzmann target over control sequences,

$$
p_0(U) \;\propto\; \exp\!\big(-J(U)/\alpha\big),
$$

and its Gaussian-smoothed family (the "noised" marginals of a diffusion),

$$
p_{\sigma}(U) \;=\; \big(p_0 * \varphi_{\sigma}\big)(U)
\;=\; \mathbb{E}_{\varepsilon \sim \mathcal{N}(0, \sigma^2 I)}
\big[\, p_0(U - \varepsilon) \,\big].
$$

MBD performs approximate reverse diffusion: starting from a warm-started $U$,
it descends the annealing ladder $\sigma_1 > \sigma_2 > \dots > \sigma_S$
(we use a linear schedule; unicycle $0.9 \to 0.2$, arm $1.2 \to 0.3$, $S=5$).

### 3.2 Score estimation (Tweedie) and the Monte-Carlo update

By Tweedie's formula, the score of the smoothed target is a denoising mean
shift:

$$
\nabla_U \log p_{\sigma}(U)
\;=\; \frac{\mathbb{E}\big[\,U_0 \,\big|\, U\,\big] - U}{\sigma^{2}},
\qquad
\mathbb{E}[U_0 \mid U]
= \frac{\int U'\, e^{-J(U')/\alpha}\, \varphi_\sigma(U - U')\, dU'}
       {\int e^{-J(U')/\alpha}\, \varphi_\sigma(U - U')\, dU'} .
$$

Because the smoothing kernel $\varphi_\sigma(U - U')$ is itself the density
of the proposal $U' \sim \mathcal{N}(U, \sigma^2 I)$, the posterior mean is a
ratio of expectations under the proposal, and the self-normalized
Monte-Carlo estimator with $N$ candidates $U_k = U + \sigma\,\varepsilon_k$,
$\varepsilon_k \sim \mathcal{N}(0, I)$, is the cost-weighted mean

$$
\widehat{\mathbb{E}}[U_0 \mid U] = \sum_{k=1}^{N} w_k\, U_k,
\qquad
w_k = \frac{\exp\!\big(-(J(U_k) - J_{\min})/\alpha\big)}
           {\sum_j \exp\!\big(-(J(U_j) - J_{\min})/\alpha\big)} .
$$

*(Implementation note: candidates are projected onto the box
$\mathcal{U}^T$ before evaluation. Formally this targets $J$ extended by
$+\infty$ outside the box under a censored proposal and introduces a
boundary bias in $\widehat{\mathbb{E}}[U_0 \mid U]$; this projection is
standard in MPPI implementations and is applied identically to every method
compared.)*

The update at annealing stage $s$ is a preconditioned score step with
optional Langevin noise:

$$
U \;\leftarrow\; U + \eta_s\, \widehat{\nabla_U \log p_{\sigma_s}}(U)
       \;+\; \sqrt{2\eta_s}\,\xi,
\qquad
\boxed{\;\eta_s = \eta\, \sigma_s^{2}\;}
$$

**The step size is tied to the noise level.** With $\eta = 1$ and no Langevin
noise this reduces *exactly* to the weighted-mean update
$U \leftarrow \sum_k w_k U_k$ (verified to $6 \times 10^{-16}$ in code). A
fixed absolute step $\eta_s \equiv \eta$ makes the effective movement
$\eta/\sigma_s^2$ collapse at large $\sigma$ and was found to stall on the
arm; it is kept only as a debug mode.

### 3.3 MPPI as the single-stage special case

The classic MPPI update is

$$
U \;\leftarrow\; U + \sum_k w_k\, \sigma \varepsilon_k
\;=\; \sum_k w_k U_k ,
\qquad w_k \propto \exp(-S(U_k)/\lambda),
$$

i.e. **one** weighted-mean step at a **fixed** noise scale. In our
parameterization this is precisely $(S{=}1,\ \sigma_1 = \sigma_{\text{fix}},\
\eta{=}1,\ \text{no Langevin noise})$. MBD generalizes MPPI along two axes —
number of inner iterations $S$ and the annealing schedule
$\sigma_1 \to \sigma_S$ — while sharing the same rollout backend, cost,
horizon, warm start, and clipping. *We therefore do not compare "MBD vs MPPI"
as different tools; we ablate one update-design axis inside a single stack*
(Sec. 8.3).

---

## 4. Koopman rollout models

### 4.1 Lift and decoder

A shared deep lift with an exact-decoder convention. With $d = \dim b$:

$$
z = \Psi_\theta(b) = \begin{bmatrix} b \\ \psi_\theta\big(\tau(b)\big) \end{bmatrix}
\in \mathbb{R}^{r},
\qquad
C z = z_{1:d} = b
\quad (\text{decode by slicing}),
$$

where $\psi_\theta$ is a tanh MLP and $\tau$ a fixed task input map
(unicycle: $\tau = \mathrm{id}$, $4 \to 64 \to 64 \to 4$, $r = 8$;
arm: $\tau(b) = [\sin q, \cos q] \in \mathbb{R}^{14}$,
$14 \to 96 \to 96 \to 10$, $r = 20$).

### 4.2 Linear vs bilinear lifted dynamics

$$
\textbf{DK (linear):}\quad z^+ = A z + B u ,
$$

$$
\textbf{BK (bilinear):}\quad
z^+ = A z + B_0 u + \sum_{i=1}^{m} u_i B_i z
\;=\; M(u)\, z + B_0 u,
\qquad
M(u) = A + \sum_i u_i B_i .
$$

The bilinear form is the minimal correct class for control-affine systems:
the input's effect on the lifted state, $\partial z^+ / \partial u_i =
B_0 e_i + B_i z$, **depends on the state through $B_i z$**, whereas the
linear form's $\partial z^+/\partial u = B$ is constant. Rollout cost is the
same matrix–vector regime for both.

### 4.3 Why the linear lift must fail (representability)

Let the tracked output increment be first-order in the input,
$\Delta y = \Phi(s)\, u\, \Delta t + O(\Delta t^2)$ with configuration-
dependent gain $\Phi(s)$ (unicycle: $\Phi = [\cos\theta, 0; \sin\theta, 0;
0, 1]$; arm: $\Phi = J(q)$ on the end-effector rows). In the linear lift the
input enters the output only through **constant** matrices: the one-step
map is $CB$, and more generally every input-to-output Markov parameter
$C A^{k} B$ is state-independent, so the entire input–output response of a
linear lift is LTI regardless of the lifted drift.

**Proposition (irreducible error of a constant gain).** Let inputs satisfy
$\mathbb{E}[u u^\top] = \lambda I$ independently of the state
$s \sim \rho$. Then

$$
\arg\min_{X}\; \mathbb{E}_{s,u}\big\|(\Phi(s) - X)\,u\big\|^2
= \mathbb{E}_{s\sim\rho}\big[\Phi(s)\big],
\qquad
\min_{X}\; \mathbb{E}_{s,u}\big\|(\Phi(s) - X)\,u\big\|^2
= \lambda\; \mathbb{E}_{s\sim\rho}\big\|\Phi(s) -
\mathbb{E}[\Phi]\big\|_F^2 .
$$

*Proof.* $\mathbb{E}_u\|(\Phi - X)u\|^2
= \mathrm{tr}\big((\Phi - X)\,\mathbb{E}[uu^\top]\,(\Phi - X)^\top\big)
= \lambda \|\Phi(s) - X\|_F^2$; minimizing the expectation over $s$ of a
quadratic in $X$ gives the mean, with the variance as the residual. $\square$

The best constant surrogate is the **mean gain**, and the irreducible
mean-squared error equals the **dispersion of the gain along the visited
distribution** — a model-class property no training can remove. The unicycle
is the extreme case: parking sweeps the heading over the full circle,
$\mathbb{E}[(\cos\theta, \sin\theta)] \approx 0$, so the optimal constant
input channel for the speed command is *zero* — the model discards the
actuator. The bilinear class instead realizes the state-dependent gain
$u \mapsto C\big(\sum_i u_i B_i z\big)$, whose columns $C B_i z$ are linear
in $z$: the heading rotation $R(\theta)$ is exactly expressible since
$\sin\theta$ and $\cos\theta$ are (affine) coordinates of $z$ (the constant
offset absorbed by $B_0$); the arm Jacobian $J(q)$ has entries that are
*multilinear* in $\{\sin q_i, \cos q_i\}$ across joints, so exact
realization requires the learned features $\psi_\theta$ to supply the
cross-products — which the deep lift learns in practice (open-loop gap
$1.5\times$, closed loop $100\%$ vs $0\%$, Sec. 8.1).

**Empirical signature.** Open-loop $H{=}15$ prediction RMSE (validation):
unicycle DK $0.082$ vs BK $0.0020$ ($41\times$); arm DK $0.086$ vs BK
$0.059$ (modest $1.5\times$, because $J(q)$ has a non-zero mean unlike the
sign-flipping heading gain) — yet closed loop amplifies this structural
deficit into $0\%$ vs $100\%$ (Sec. 8.1): the win is representability
amplified by control, not raw prediction ratio.

### 4.4 The split baseline: what "linear works" actually requires

The MPPI-DK paper (Hao et al., arXiv:2603.05385) reports $100\%$ success —
but in every one of its experiments the input enters state-independently and
the first-order coupling is handled *analytically outside the learned model*
(surface vehicle: learned body-frame dynamics, analytic rotation integration;
quadruped: body-frame acceleration inputs with $\cos\theta,\sin\theta$ state
features). We reproduce these conditions as `dk_mbd_split`:

- **unicycle:** fit $[\Delta p_{\text{body}}; \Delta\theta] = G u$ by least
  squares, where $\Delta p_{\text{body}} = R(-\theta_k)(p_{k+1} - p_k)$;
  roll out by analytic $SE(2)$ composition
  $p^+ = p + R(\theta)\,(Gu)_{1:2},\ \theta^+ = \theta + (Gu)_3$.
  (On a kinematic vehicle the learned part degenerates to the integrator:
  the fitted $G \approx [\,\mathrm{diag}(\Delta t)\,; \cdot\,]$ — the
  decomposition, not the model class, carries the coupling.)
- **arm:** linear deep Koopman on $q$ only ($z^+ = Az + Bu$ over lifted $q$),
  with the analytic forward kinematics applied to the decoded joints when
  evaluating the cost.

`dk_mbd_split` succeeds ($55/55$) — confirming the baseline is not
under-tuned and that the published MPPI-DK results are consistent — but it
**requires exact analytic structure ($R(\theta)$, FK) that the model-free
setting withholds**. The bilinear lift achieves the same success with no such
structure.

---

## 5. Training: shared multi-step loss

Both DK and BK (and the arm split model) are trained by the same $H$-step
loss on random-rollout snippets ($H = 15$):

$$
\mathcal{L}(\theta, A, B_\bullet)
= \frac{1}{H} \sum_{h=1}^{H}
\Big[ \big\| C \hat{z}_h - b_h \big\|^2
+ \gamma \big\| \hat{z}_h - \Psi_\theta(b_h)^{\text{sg}} \big\|^2 \Big],
\qquad
\hat z_0 = \Psi_\theta(b_0),\;\;
\hat z_{h+1} = F(\hat z_h, u_h),
$$

with $\gamma = 0.1$ and stop-gradient on the latent target. (In the
implementation both terms are per-component *means* rather than sums;
$\gamma$ is stated in that convention — the difference is a constant factor
absorbed into $\gamma$ and the learning rate.) **Zero
initialization** $B_i = 0$ starts BK exactly at the linear model, adding
bilinear terms only where the data demands them (this stabilizes the $m$
per-channel matrices; the arm uses additionally cosine LR decay, weight decay
$10^{-4}$, and gradient clipping at norm $2$). Datasets are generated once
(fixed data seed) and shared across all learned models and training seeds;
one-step regression in place of the $H$-step loss yields rollouts that
diverge before the planning horizon.

---

## 6. Error tube (model-error-aware cost)

One-step lifted residuals on the training set,
$r_k = \Psi_\theta(b_{k+1}) - F(\Psi_\theta(b_k), u_k)$, are covered by a
proportional bound $\|r_k\| \le c_x \|z_k\| + c_u \|u_k\|$ fitted at
**quantile coverage** (least-squares fit of $(\hat c_x, \hat c_u)$,
rescaled by the empirical $(1-\delta)$-quantile of the ratios
$\|r_k\| / (\hat c_x\|z_k\| + \hat c_u\|u_k\|)$, $\delta = 10^{-3}$).

> Changed 2026-07-02 (was: worst-case linprog full coverage). Reason: the
> FR3 dataset's rest-to-motion transients — one-step residuals a
> velocity-free lifted state cannot represent — are rare outliers
> (99.9%-quantile ratio ≈ 0.03, max ≈ 0.32, identical across dk/bk/split,
> hence a data property). Worst-case fitting set $c_x = 0.32$, the tube
> \eqref-recursion grew exponentially over $T = 15$, and the tube term
> dominated the anneal: bk_mbd oscillated 0.23 m from a target it reaches
> at 0.024 m with the quantile fit. All runners
> (unicycle/arm/franka/run_all) now use
> `fit_tube_constants(..., quantile=0.999)`; unicycle/arm multiseed
> results are unchanged under the switch.

During planning the scalar tube is propagated along each candidate with the
precomputed norm bound
$\bar m(u) = \|A\|_2 + \sum_i |u_i|\, \|B_i\|_2 \;\ge\; \|M(u)\|_2$:

$$
e_{k+1} = \big(\bar m(u_k) + c_x\big)\, e_k + c_x \|\hat z_k\| + c_u \|u_k\|,
\qquad e_0 = 0,
$$

which maintains $e_k \ge \|\delta_k\|$, $\delta_k := \Psi_\theta(b_k) -
\hat z_k$, by induction: since $F(z, u) = M(u) z + B_0 u$ is linear in $z$,
the error obeys exactly $\delta_{k+1} = M(u_k)\,\delta_k + r_k$; applying
the residual bound at the true lifted state and
$\|z_k\| \le \|\hat z_k\| + \|\delta_k\|$ gives

$$
\|\delta_{k+1}\|
\;\le\; \big(\|M(u_k)\|_2 + c_x\big)\|\delta_k\|
+ c_x \|\hat z_k\| + c_u \|u_k\| ,
$$

and $\bar m \ge \|M(u)\|_2$ closes the induction (this is where the $+c_x$
in the contraction factor comes from).

and the planner minimizes the tube-augmented cost

$$
\tilde J(U) = J(U) + \beta_e \sum_{k} e_k .
$$

$\beta_e$ scales with the task cost magnitude (unicycle $10^{-2}$, arm
$2\times 10^{-4}$); $\beta_e = 0$ recovers plain BK-MBD, and the fitted
constants are empirical proxies (per-seed unicycle:
$c_x \in [2.7, 4.1]\times 10^{-3}$, $c_u \in [3.3, 5.7]\times 10^{-3}$).

---

## 7. Why annealing beats a fixed noise scale (mechanisms)

With a learned surrogate the evaluated cost is $\hat J(U) = J(U) + e(U)$,
where the model-error term $e$ inherits high-frequency structure in $U$ from
error accumulation along rollouts.

**(i) Gaussian homotopy / graduated non-convexity.** Each annealing stage
descends the effective landscape
$J_\sigma := -\alpha \log\big(e^{-\hat J/\alpha} * \varphi_\sigma\big)$.
By Jensen's inequality $J_\sigma \le \hat J * \varphi_\sigma$, with the
high-temperature expansion

$$
J_\sigma \;=\; \hat J * \varphi_\sigma
\;-\; \tfrac{1}{2\alpha}\,\mathrm{Var}_{\varphi_\sigma}\!\big(\hat J\big)
\;+\; O(\alpha^{-2}),
$$

so to leading order the stage-$\sigma$ landscape is the Gaussian-smoothed
cost. Smoothing attenuates the Fourier components of the surrogate-error
term as

$$
\widehat{(e * \varphi_\sigma)}(\omega)
= \hat e(\omega)\, \exp\!\big(-\tfrac{1}{2}\sigma^2 \|\omega\|^2\big) .
$$

Early stages ($\sigma = 1.2$) therefore select the basin on a landscape in
which the surrogate's spurious fine-scale minima are exponentially
suppressed; late stages ($\sigma = 0.3$) refine within the chosen basin,
where local model bias is small. A fixed $\sigma$ must serve both roles with
one scale.

**(ii) Precision floor of a fixed scale.** The floor is a *finite-sample*
effect, not a property of the idealized iteration: for convex $J$ and
$N \to \infty$ the fixed-$\sigma$ weighted-mean fixed point is the exact
minimizer, but with $N$ samples the estimator
$\sum_k w_k U_k$ carries Monte-Carlo noise of order
$\sigma / \sqrt{N_{\mathrm{eff}}}$ per axis
($N_{\mathrm{eff}} = 1/\sum_k w_k^2$, the effective sample size), and in
receding horizon the control is *executed after a single update*, so this
noise enters the plant directly. At $\sigma \equiv 1.2$ against a $0.05\,$m
reach threshold — and on a non-convex surrogate whose fine-scale error is
not smoothed away near convergence — the final approach stalls just above
threshold. *This is exactly the observed failure signature* (Sec. 8.3):
every fixed-$\sigma$ failure ends at $0.050$–$0.066\,$m — marginal stalls,
not divergence — and annealing to $\sigma_S = 0.3$ removes all of them
(every trial strict, $< 0.025\,$m).

**(iii) Sample efficiency (weak effect).** Annealing also spreads the
proposal–target mismatch across stages (as in annealed importance sampling),
but empirically this is not the binding constraint: $5\times$ more samples in
a single fixed-$\sigma$ stage recovers only $+1$ trial (Sec. 8.3, config b).

---

## 8. Experimental evidence

Fairness: all methods share the task, cost, horizon, sample count, diffusion
schedule, action bounds, warm start, clipping, closed-loop protocol, and
per-trial planning seed ($\text{seed} \times 10 + \text{case}$); DK and BK
additionally share the dataset, lift architecture, dimension, and training
budget, differing **only** in the lifted dynamics class. The oracle runs on
every trial.

### 8.1 Main comparison (multi-seed)

(Numbers below are from the 2026-07-02 reruns with the quantile tube fit;
they match the earlier linprog-fit runs within noise on unicycle/arm.)

**Unicycle parking** (5 training seeds × 4 start–goal pairs = 20 trials;
park $< 0.3$ m):

| method | success | strict | final error [m] | plan [ms/step] |
|---|---|---|---|---|
| MBD-true (oracle)      | 20/20 (100%) | 20/20 | 0.070 ± 0.008 | 81 |
| DK-MBD (linear)        | **0/20 (0%)** | 0/20 | 2.383 ± 1.706 | 43 |
| DK-MBD-split           | 20/20 (100%) | 20/20 | 0.066 ± 0.010 | 58 |
| **BK-MBD (ours)**      | **20/20 (100%)** | 18/20 | 0.071 ± 0.026 | 95 |

**7-DOF arm reaching** (5 seeds × 7 targets = 35 trials; reach $< 0.05$ m,
strict $< 0.025$ m):

| method | success | strict | final EE error [m] | plan [ms/step] |
|---|---|---|---|---|
| MBD-true (oracle)      | 35/35 (100%) | 35/35 | 0.023 ± 0.001 | 113 |
| DK-MBD (linear)        | **0/35 (0%)** | 0/35 | 0.729 ± 0.399 | 43 |
| DK-MBD-split           | 35/35 (100%) | 35/35 | 0.023 ± 0.001 | 49 |
| **BK-MBD (ours)**      | **35/35 (100%)** | 35/35 | 0.024 ± 0.001 | 69 |

**Franka FR3 reaching, MuJoCo full physics** (FR3 + Franka Hand, TCP
control point, gravity-compensated joint-velocity servo interface, 50 ms
control period = 25 physics substeps; 5 seeds × 7 targets = 35 trials;
reach $< 0.05$ m, strict $< 0.025$ m):

| method | success | strict | final TCP error [m] | plan [ms/step] |
|---|---|---|---|---|
| MBD-true (oracle, threaded MuJoCo rollouts) | 35/35 | 35/35 | 0.023 ± 0.002 | **4094** |
| DK-MBD (linear)        | 13/35 (37%) | **7/35 (20%)** | 0.054 ± 0.022 | 37 |
| DK-MBD-split           | 35/35 (100%) | 35/35 | 0.023 ± 0.001 | 56 |
| **BK-MBD (ours)**      | **35/35 (100%)** | 35/35 | 0.023 ± 0.002 | **64** |

Failure modes of DK-MBD: unicycle — drives off along spirals (constant $B$
averages the heading gain to zero); arm — either diverges ($>1$ m) or
jitters without converging near the target (constant $B$ cannot track
$J(q)$); FR3 — approaches but stalls near the target (smoother servo
dynamics soften, but do not repair, the constant-gain error: only 7/35
strict).

### 8.2 Rollout-cost scaling (how to state the speed claim)

Koopman rollout cost is fixed by the lift dimension, independent of dynamics
complexity; oracle rollout cost scales with the dynamics. On the cheap RK4
unicycle the oracle is *faster* (81 vs 95 ms/step); on the 7-DOF arm with
per-step forward kinematics BK-MBD is already $1.7\times$ faster (69 vs 113);
on the MuJoCo FR3 (the native MBD setting) the gap is **64×** (64 ms vs
4.1 s per 50 ms control step) — i.e., the surrogate moves the planner from
two orders of magnitude beyond real time to inside the control period.
**Claim the scaling, not a uniform speedup** (now demonstrated at the
engine end by the FR3).

### 8.3 Optimizer ablation: MPPI as an operating point

Same trained BK checkpoints, same task/cost/tube; only the update schedule
varies (arm, 35 trials each):

(2026-07-02 rerun on the quantile-tube stack; strict counts added.)

| config | structure | reach | strict | final err [m] | ms/step |
|---|---|---|---|---|---|
| (a) MPPI-equivalent | $S{=}1 \times 800$, $\sigma \equiv 1.2$ | 32/35 (91%) | 25/35 | 0.029 ± 0.013 | 18 |
| (b) equal compute   | $S{=}1 \times 4000$, $\sigma \equiv 1.2$ | 32/35 (91%) | 28/35 | 0.028 ± 0.008 | 24 |
| (c) iterations only | $S{=}5 \times 800$, $\sigma \equiv 1.2$ | 34/35 (97%) | 32/35 | 0.027 ± 0.017 | 71 |
| (d) **annealed (ours)** | $S{=}5 \times 800$, $\sigma: 1.2 \to 0.3$ | **35/35 (100%)** | **35/35** | **0.024 ± 0.001** | 69 |

FR3 replication of the same sweep: **all four configs 35/35 strict**
((a) 17, (b) 22, (c) 64, (d) 64 ms/step) — the 2.5 cm strict tolerance
sits above the precision floor there, so the dial matters on the arm but
not on the FR3; report as scope, not contradiction.

Seven of the eight failures across (a)–(c) are marginal stalls at
$0.050$–$0.079$ m, six of them on the farthest target (the eighth stalls
at $0.124$ m); never divergence — the fixed-$\sigma$ precision-floor
mechanism of Sec. 7(ii). Strict success climbs monotonically
($25 \to 28 \to 32$) but only annealing reaches $35/35$, at the same
rollout budget as (c). The published BK-MPPI results
(14/35, seed counts $[5,1,3,2,3]$) sit well below our controlled
MPPI-equivalent (32/35, $[6,6,7,6,7]$); our per-seed BK open-loop RMSE is
uniform ($0.059$–$0.062$), so we attribute the published gap to
training-draw variance in a protocol with one training run and one
closed-loop evaluation per cell and no model-quality diagnostic — cite the
published number as context, and make quantitative claims only within the
unified stack.

### 8.3b Mechanism: open-loop gap vs closed-loop failure (2026-07-02)

Open-loop prediction error of the *actual control checkpoints* (5 seeds,
held-out data), decomposed into coupled rows (EE/TCP) vs joint rows:

| task | EE/TCP ratio dk/bk @H15 | joints | closed loop |
|---|---|---|---|
| unicycle | 40x @H1 (0.082 vs 0.002 RMSE; near-zero-mean gain) | — | dk diverges 0/20 |
| arm | **1.40x** | coincide | dk 0/35 vs bk 35/35 (31x final err) |
| FR3 | **1.33x** | coincide | dk 7/35 strict vs bk 35/35 |

Punchline: closed-loop failure is NOT proportional to open-loop error. The
mean-gain model's residual is a *structural bias* (always same direction per
configuration), the annealed search selects candidates that exploit exactly
that bias, and receding-horizon replanning re-commits every step. Open-loop
scores under-report the deficit in proportion to how benign the visited gain
distribution is (non-zero-mean Jacobian → tiny open-loop gap, total
closed-loop failure). Figure: paper/figs/pred_error_horizon.png; script:
experiments/analysis_figs.py.

### 8.3c Tube validation (2026-07-02)

Quantile-fitted (0.999) tube, bk seed-0 FR3 model, 2000 held-out rollouts x
40 steps (80,000 steps, 2.7x planning horizon): **violations 20/80,000 =
0.025%** — one-step 99.9% coverage transfers through the propagation.
Conservatism: median e_k/||delta_k|| ≈ 1.8e3 @H15 (growth factor
mbar+c_x > 1 with ||A||=1.004, sum||B_i||=1.24) → tube is a *relative
ordering signal* (soft trust region), not a certificate. Figure:
paper/figs/tube_validation.png.

### 8.3d Regime delimitation (2026-07-02)

- **Pendulum** (input state-independent → Prop-1 penalty = none): fixed
  r=8, encoder width sweep 8..256, 3 seeds. **Bilinear @32 neurons =
  accuracy linear reaches only @256 (8x width gap).** Closed-loop MBD
  swing-up with control-grade width-64 models (20k snippets, K=15, 220
  epochs), 5 planning seeds, horizon 50: **DK-MBD 5/5** (final angle 0.14
  rad), BK-MBD 3/5 (succeeds at horizon 40; open-loop BK is *better* at
  every horizon — H15 0.400 vs 0.717 — so this is planner stochasticity on
  a marginal underactuated task, not model quality). Framing: where the
  coupling is absent, bilinear buys compactness, NOT closed-loop
  advantage; linear machinery is fully serviceable → sharpens the
  first-order-coupling claim. True-dynamics MBD sanity: swings up in 3.5 s
  with the same loop/cost.
- **Quadrotor** (coupling on acceleration = second-order): full-state gap
  2.3x @H12, but velocity rows converge to 1.4x @H12 (short-horizon
  bilinear advantage on velocity is real — Δv ∝ T·R e₃ is a state-input
  product — but attitude-prediction error dominates both classes at longer
  horizons). Linear lift serviceable on control-relevant rows → delimits
  the BK-MBD regime; consistent with linear-lift literature (body-frame /
  acceleration inputs). Figure: paper/figs/quad_boundary.png; script:
  experiments/quad_boundary.py.

### 8.4 What resolves the "left open" problem

BK-MPPI attributes its arm unreliability to the $m$ per-channel bilinear
matrices being hard to optimize. In our stack the models are uniformly good
across seeds (open-loop RMSE $0.059$–$0.062$), zero-init + cosine decay +
gradient clipping suffice for stable *training*, and the residual
unreliability is a *planning* phenomenon at fixed $\sigma$ — removed by the
annealed update. Together: reliable high-DOF bilinear Koopman control =
(zero-init multi-step training) + (annealed multi-iteration planning).

---

## 9. Claim boundaries (what we assert, and what we do not)

**We claim:**
1. Structural failure/unreliability of the linear lift under first-order
   coupling (Sec. 4.3 argument + 0/55 closed loop on the analytic testbeds
   + 7/35 strict on the FR3), and that it is a model-class property, not
   baseline under-tuning (split control: 90/90).
2. The split repair requires analytic structure unavailable model-free;
   the bilinear lift needs none and matches the oracle (90/90, oracle-level
   final errors on all three testbeds).
3. Within a unified stack, annealed multi-iteration updates account for
   $91\% \to 100\%$ reach and $71\% \to 100\%$ strict (final error
   $0.029 \to 0.024$) over the MPPI-equivalent operating point on the arm,
   at $\sim 3.8\times$ the per-step cost of the 1-step variant; a
   speed/reliability dial, not a free lunch. **Scope**: the benefit appears when the tolerance approaches
   the sampling precision floor — on the FR3 all four operating points
   succeed strictly (35/35 each; tolerance far above the floor), which we
   report as consistent with the same mechanism, not as a counterexample.
4. Rollout-cost decoupling from dynamics complexity (scaling claim,
   Sec. 8.2; engine-end instance: 64× on the FR3).

**We do not claim:**
- A 60-point improvement over *published* BK-MPPI numbers (cross-stack;
  cited as context only).
- A uniform wall-clock speedup over the oracle (false on cheap dynamics).
- Certified tube bounds (constants are empirical quantile-coverage proxies
  — $\delta = 10^{-3}$ — with no generalization guarantee; Sec. 6 records
  why worst-case fitting is actively harmful).
- That annealing is necessary on every task (see claim 3 scope; FR3 sweep).
- Anything about second-order/weakly coupled regimes (pendulum, quadrotor)
  — out of scope here; the linear lift remains well matched there.

**Honest footnotes to carry:** the initial lifted state at each replan uses
the measured/computed current output ($b(s_t)$, incl. FK at the current
joint state) — sensor-level information, identical across all learned
methods; the arm environment uses the analytic kinematic chain (verified to
$4\times10^{-16}$ against MuJoCo); MBD is run in receding horizon, so the
"trajectory optimizer vs real-time controller" distinction dissolves — at
62 ms/step (16 Hz) BK-MBD is itself real-time-capable, and the MPPI update
is recovered as its $S{=}1$ special case.

---

## 10. Suggested paper skeleton

1. **Introduction** — rollout model as the bottleneck; Koopman promise; the
   linear lift's structural cost; contributions (bilinear rollout in MBD,
   split-controlled baseline analysis, annealing ablation, tube-augmented
   cost).
2. **Preliminaries** — problem setup (Sec. 2), MBD update with
   $\eta_s = \eta\sigma_s^2$ and MPPI as special case (Sec. 3).
3. **Method** — bilinear Koopman rollout + representability proposition
   (Sec. 4.1–4.3), multi-step training (Sec. 5), error tube (Sec. 6).
4. **Why annealing** — mechanisms (Sec. 7), stated as hypotheses that the
   ablation then tests.
5. **Experiments** — main tables, split baseline, optimizer ablation,
   scaling discussion (Sec. 8), figures: unicycle trajectories,
   arm final-pose panels, arm error bars, ablation table.
6. **Related work** — MBD; MPPI-DK (complementary: weakly coupled regime);
   BK-MPPI (prior work whose reliability gap we close); bilinear
   Koopman/EDMD control; SafEDMD-style bounds.
7. **Limitations** — empirical tube constants; kinematic tasks only (no
   contact); receding-horizon compute dial; OOD penalty left as ablation.
