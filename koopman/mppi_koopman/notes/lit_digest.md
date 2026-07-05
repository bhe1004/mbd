# Literature Digest — MPPI × Koopman

두 리뷰 논문(`ref/`) 정독 정리. 연구 방향 설계용 레퍼런스.

- **[Koopman]** Strässer et al., *"An overview of Koopman-based control: From error bounds to closed-loop guarantees"*, Annual Reviews in Control 61 (2026) 101035.
- **[MPPI]** Park, Jang, Kim, *"A Systematic Survey of Model Predictive Path Integral Control: From Stochastic Theory to Real-Time Robotics"*, ISA Transactions (2026, submitted).

> **핵심 관찰**: 두 리뷰는 서로를 전혀 언급하지 않음. [Koopman]에 "MPPI/path integral" 0회, [MPPI]에 "Koopman/EDMD/DMD/lifted" 0회. 교집합이 리뷰 레벨에서 공백.

---

## Part A. Koopman 리뷰 핵심

### A.1 Setup & notation
- 연속시간 자율계: `ẋ = f(x)`, 이산: `x⁺ = F(x)`.
- Koopman 작용소(관측함수 ψ에 작용): `(K^t ψ)(x̂) = ψ(x(t;x̂))` — 상태가 아니라 **관측함수를 선형 전파**.
- Generator: `d/dt ψ = ⟨∇ψ, f⟩` (transport PDE). 유한차원 근사 = dictionary Ψ=(ψ₁..ψ_M) 위로 Galerkin 사영.
- Control-affine 계 (WLOG): `ẋ = f(x) + G(x)u`, `G=(g₁..g_m)`.

### A.2 LINEAR vs BILINEAR predictor (이 논문의 중심 주장)
- **Linear (EDMDc)**:  `Ψ(x⁺) ≈ A Ψ(x) + B u`,  `x = C Ψ`.
- **Bilinear (bilinear EDMDc)**:  `Ψ(x⁺) ≈ A Ψ(x) + B₀ u + Σᵢ uᵢ (Bᵢ Ψ(x))`.
- **논문의 결론(boxed)**: 입력이 원계에 선형으로 들어가도 lifting의 Lie 도함수에서 **상태 의존 항**이 생김 → *선형 유한차원 표현은 비선형 제어계를 일반적으로 정확히 못 담음*. 정확한 lifted 표현은 **최소한 bilinear**이어야 함 (Iacob–Tóth–Schoukens 2024). ⇒ **"bilinear이 옳은 system class"**.

### A.3 식별 (최소제곱)
- EDMD: `min_K ‖Ψ_Y − K Ψ_X‖_F`, 해 `K=(Ψ_Y Ψ_Xᵀ)(Ψ_X Ψ_Xᵀ)⁻¹`.
- EDMDc: `min_{A,B} ‖Ψ_Y − AΨ_X − BU‖_F`.
- bilinear EDMDc: 위에 Kronecker 항 `Ψ_U=(u₁⊗Ψ(x₁) …)` 추가.
- kEDMD(커널): dictionary = 커널 특성 `k(xⱼ,·)`; RKHS pointwise 평가로 **uniform(L∞) bound** 가능. Wendland(컴팩트 지지) 커널이 Koopman 불변성·유계성 보장.
- Deep Koopman: NN dictionary. **완전한 오차해석 없음 → 아직 폐루프 보장 불가**.

### A.4 오차 bound (이 논문의 강조점)
오차 = (i) projection error(유한 dictionary) + (ii) estimation error(유한 data).
- 확률적 유한데이터 operator bound: `P(‖K_d^M − P_V K|_V‖_F ≤ ε) ≥ 1−δ`, 필요 데이터 `d=O(M²/(δε²))`.
- 커널 uniform bound: `‖K̂−K‖_{H→L∞} ≤ C·h^{k+1/2}`, fill distance `h`.
- **★ Proportional error bound (중심 결과)**: bilinear surrogate의 잔차 r에 대해
  ```
  ‖r(x,u)‖ ≤ c_x ‖Ψ(x)‖ + c_u ‖u‖     ∀(x,u)
  ```
  **원점(평형)에서 0으로 사라짐** → robust 안정화의 핵심.
  - SafEDMD: 확률적, `c ∈ O(1/√(δd) + Δt²)` w.p. 1−δ.
  - kernel bilinear kEDMD: **결정론적**, full residual, `c ∈ O(1/√(2Nd)+Δt²)`, Δt→0·d→∞에서 →0.
- **알고리즘 무관**: proportional bound를 주는 *어떤* 방법이든 폐루프 보장 가능.

### A.5 폐루프 보장
proportional bound가 있는 uncertain bilinear 계 → **robust control 문제**로 환원:
- Feedback: LFR+robust, SafEDMD LMI gain-scheduling `μ(x)=(I−K_w(I_m⊗Ψ))⁻¹KΨ`, SOS(전역) → **지수 안정화** (Lyapunov sublevel set = RoA).
- MPC: Korda–Mezić 선형 Koopman MPC(보장 없음) → bilinear EDMDc MPC는 **cost controllability + proportional bound** 로 *terminal 없이* practical asymptotic stability, terminal 조건 추가 시 asymptotic stability. 커널 MPC도 동일 라인.
- **보장 체인**: proportional error bound + (cost controllability / Lipschitz / terminal) ⇒ 원 비선형계의 practical/asymptotic stability + recursive feasibility.

### A.6 Open problems
입력-출력 데이터만으로 보장 / 데이터 요구량·보수성 축소 / dictionary 선택 / lifted→원계 스케일러빌리티·noisy data. **bilinear 데이터기반 MPC로 nonlinear-MPC 연산비 회피** 명시적 권고.

---

## Part B. MPPI 리뷰 핵심

### B.1 두 가지 유도 (등가)
**(1) Information-theoretic (Williams)** — 일반 이산계 (control-affine 불필요):
- 모델: `x_{t+1}=F(x_t,v_t)`, `v_t~N(u_t,Σ)`.
- 비용: `S(V)=φ(x_T)+Σ q(x_t)`.
- 자유에너지: `F=−λ log E_p[exp(−S/λ)]`, Jensen → `F ≤ E_q[S]+λ·KL(q‖p)`.
- 최적분포: `q*(V)=(1/η)exp(−S/λ)p(V)`; `U*=argmin KL(q*‖q)=E_{q*}[V]`.
- **★ 업데이트**: `u_t ← u_t + Σ_k w(V_k) ε_t^(k)`,  `w(V_k)=softmax(−S̃(V_k)/λ)`.

**(2) Path-integral (Kappen)** — control-affine SDE 필요:
- `dx=f(x)dt+G(x)u dt+B(x)dw`; Cole–Hopf `V=−λ log Ψ`.
- **★ matching condition**: `B Bᵀ = λ G R⁻¹ Gᵀ` → HJB가 선형 PDE화, Feynman–Kac `Ψ=E_Q[exp(−S/λ)]`.
- 같은 업데이트식으로 귀결. 등가 조건 `Σ=λR⁻¹`.

### B.2 가정·한계
- **IT 유도**: F measurable, S finite 만 필요 → **임의(비-affine·비선형) dynamics 허용**. 현대 MPPI가 IT를 쓰는 이유.
- **PI 유도**: control-affine + matching condition (노이즈가 **제어 채널로만** 진입). 단순화판은 G 정방·가역(=#입력=#제어상태) → 과/저구동 배제.
- Gaussian 제안분포는 비선형계의 비-Gaussian q*를 잘 못 덮음.
- **형식적 폐루프 안정성 정리 없음**; receding-horizon 재계획은 경험적 보정일 뿐.
- 하드 제약 보장 없음(soft); λ 민감; **risk-neutral**; 좋은 모델 의존.
- **샘플 효율이 제어차원 m·horizon T에 지수적으로 악화** (궤적공간 R^{m×T}); humanoid 전신제어에서 심각.

### B.3 변형 taxonomy (4 테마)
1. **Sampling**: mean-adaptation(planner/learning/SVGD guided), covariance-adaptation(colored/TC/DV/MPOPI), non-Gaussian(Log-MPPI, normalizing flow), sampling-space 변환(o-MPPI, SMPPI).
2. **Constraint**: penalty(soft), chance/CVaR(CC-MPPI, GP-MPPI, U-MPPI, RA-MPPI), safety filter(CBF/CLF/HJ: Shield-MPPI 등), constraint-aware sampling(π-MPPI 등).
3. **Framework**: hierarchical(상/하위 결합), multi-MPPI(parallel/PE-MPPI/contingency).
4. **Robustness**: 불확실성(parameter free-energy, sensitivity tube, GP/ensemble variance cost, U-MPPI), 외란(**Tube-MPPI**, **Robust-MPPI=실계 free-energy 증가에 형식적 bound**, ZSG-MPPI).

### B.4 사용 dynamics 모델 (Koopman 질문)
- 해석/기구학, 학습 NN(LSTM), GP, GMM-GMR, jump-diffusion, **물리엔진 rollout(MuJoCo/Isaac, 최고비용)**.
- rollout 연산비가 핵심 병목: CPU 순차 → 수 Hz, GPU/CUDA → 수백 Hz.
- **Koopman/EDMD/DMD/lifted-linear: 전혀 등장 안 함.** 가장 가까운 미래과제: "저차원 latent **motion manifold** 학습으로 샘플 배분" (= Koopman이 들어갈 자리, 단 명명 안 됨).

### B.5 Open problems
1. **일반 폐루프 안정성 정리 부재** (terminal ingredient 없음 + soft averaging이 매 스텝 비용감소 미보장).
2. 샘플효율 m·T 지수악화 → **학습된 저차원 manifold 샘플링** (이론: latent 샘플링이 MPPI 추정 품질 보존하는 조건).
3. Sim-to-real gap → 샘플링에 domain randomization 내장, predictive state forwarding.

---

## Part C. 두 리뷰를 잇는 연구 빈틈 (설계 시드)

| 구분 | [Koopman] 리뷰 | [MPPI] 리뷰 | 결합 빈틈 |
|---|---|---|---|
| 모델 클래스 | bilinear이 "옳은 class" | rollout 모델은 NN/GP/physics, **Koopman 없음** | **bilinear Koopman을 MPPI rollout으로** |
| 입력 진입 | linear는 일반적으로 부정확 | IT-MPPI는 **임의 dynamics 허용**(convex·affine 불필요) | MPPI는 bilinear lifted를 *공짜로* 수용 → MPPI-DK의 linear 한계 극복 |
| 모델 오차 | **proportional bound** `‖r‖≤c_x‖Ψ‖+c_u‖u‖` (원점서 0) | MPPI는 risk-neutral, 오차 무시 | bound를 **MPPI cost/공분산에 주입** → risk-aware Koopman-MPPI |
| 폐루프 보장 | bound+cost-controllability ⇒ practical stability | **안정성 정리 부재** | Koopman bound 기반 MPPI 안정성 논증 |

**선행연구 경계**: MPPI-DK (arXiv 2603.05385)가 *linear* deep-Koopman를 MPPI rollout에 사용(속도 목적). bilinear·error-bound·보장은 미개척.

### 후보 연구 각도
- **(A)** Bilinear Koopman + MPPI — "MPPI라서 convex 불필요 → 더 정확한 bilinear lifted 사용 가능". MPPI-DK(linear) 대비 모델 우위.
- **(B)** Error-bound 기반 risk-aware MPPI — proportional bound를 stage cost/샘플 공분산(CVaR·tube)에 주입. 두 리뷰 직접 연결.
- **(C)** Koopman-MPPI 폐루프 안정성 이론 — bound + cost-controllability를 sampling 세팅으로 이식.
