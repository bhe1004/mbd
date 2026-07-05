# Formulation — (A) Bilinear Koopman-MPPI + (B) Error-bound Risk-aware MPPI

연구 정식화 초안. 근거: [[lit_digest]] (`notes/lit_digest.md`), `ref/` 6편.
작업명(가칭): **RB-KMPPI** — *Robust Bilinear Koopman MPPI*.

---

## 0. 한 줄 아이디어
MPPI는 샘플링 기반이라 **convexity가 불필요** → Koopman-MPC가 QP를 위해 강제하는 *linear* 제약을 벗고
**bilinear lifted 모델**을 rollout에 쓸 수 있다 (A). 동시에 Koopman의 **proportional error bound**를
MPPI 비용에 **error-tube 패널티**로 주입해 risk-neutral MPPI를 **모델오차-인지형**으로 만든다 (B).

---

## 1. 문제 정의 (State / System)

제어-affine 비선형계 (Koopman 리뷰 식 14):
```
ẋ = f(x) + G(x) u ,   G=(g₁ … g_m),   x∈X⊆ℝⁿ, u∈𝕌⊆ℝᵐ, 0∈int𝕌, f(0)=0
```
샘플시간 Δt 이산화:  `x_{k+1} = F(x_k, u_k)` (해석식 미지, 데이터 D={(x_k,u_k,x_{k+1})}만 보유).

목표(후퇴구간):
```
min_U  J = E[ φ(x_T) + Σ_{k=0}^{T-1} q(x_k,u_k) ]   s.t. (soft) h(x_k)≥0
```

## 2. Lifting (관측함수 / 상태 정의)

dictionary `Ψ: ℝⁿ→ℝᴺ`, `Ψ(x)=[x ; ψ_nl(x)]`, 제약:
- `Ψ(0)=0` (shifted lifting; 평형에서 0)
- **sandwich**: `ℓ‖x‖ ≤ ‖Ψ(x)‖ ≤ L‖x‖`  (proportional bound 성립 전제, SafEDMD/kernel)

lifted 상태 `z=Ψ(x)∈ℝᴺ`,  역사영 `x = C z` (첫 n좌표가 x면 `C=[Iₙ 0]` 고정).

## 3. Bilinear Koopman surrogate (학습 모델)

**참 dynamics** (Koopman 리뷰 식 16/23, bilinear EDMDc):
```
z_{k+1} = A z_k + B₀ u_k + Σ_{i=1}^m u_{k,i} (B_i z_k) + r_k
```
**★ proportional error bound** (Koopman 리뷰 식 21/22, SafEDMD/kernel):
```
‖r_k‖ = ‖r(x_k,u_k)‖ ≤ c_x ‖z_k‖ + c_u ‖u_k‖        (원점에서 0)
```
- `c_x, c_u`는 **식별과 동시에 데이터에서 산출** (SafEDMD: 확률적 `c∈O(1/√(δd)+Δt²)`; kernel: 결정론적). → **손으로 튜닝하는 값이 아님** = 셀링포인트.

**식별**: `min_{A,B₀,B_{1..m}} ‖Z_Y − A Z_X − B₀U − (B₁…B_m)Z_U‖_F`, `Z_U=(u₁⊗z₁ … u_d⊗z_d)` (Kronecker).

**Nominal rollout 모델** (MPPI가 굴리는 것):
```
ẑ_{k+1} = M(u_k) ẑ_k + B₀ u_k ,   M(u) := A + Σ_i u_i B_i
```
→ 입력의존 행렬 `M(u)`지만 여전히 **matvec만** (= MPPI-DK linear과 같은 GPU 병렬 비용대,
   샘플당 스텝당 `m+1` matvec). MPPI-DK의 속도 이점 유지 + 정확도 향상.

---

## 4. (A) Bilinear Koopman-MPPI

IT-MPPI은 `F` measurable·`S` finite만 요구(임의 dynamics 허용) → bilinear `M(u)` **무수정 수용**.
사이클당:
1. warm-start `U=(u_0..u_{T-1})`
2. `k=1..K`: 샘플 `ε_t^k~N(0,Σ)`, `v_t^k=u_t+ε_t^k`; `z_0=Ψ(x_meas)`에서 bilinear로 `ẑ^k` rollout
3. 비용 `S^k = φ(Cẑ_T^k) + Σ_t q(Cẑ_t^k, v_t^k) + λ Σ_t u_t^⊤Σ⁻¹ε_t^k`
4. `w^k = softmax(−(S^k−ρ)/λ)`,  `u_t ← u_t + Σ_k w^k ε_t^k`,  apply `u_0`

(또는 비용을 lifted 공간에서 직접: `q_z(z,u)=‖z−z_ref‖²_{Q_z}+‖u‖²_R`.)

---

## 5. (B) Error-bound 주입 → Risk-aware

**(B-1) per-step 오차크기** (bound를 nominal ẑ로 평가):
```
δ_k := c_x ‖ẑ_k‖ + c_u ‖v_k‖
```

**(B-2) 전파 오차 튜브** (샘플당 스칼라 재귀, 비용 무시 가능):
```
e_0 = 0
e_{k+1} = (‖M(v_k)‖ + c_x) e_k + c_x‖ẑ_k‖ + c_u‖v_k‖
```
→ `e_k`는 참–nominal lifted 편차의 상계: `‖z_k−ẑ_k‖ ≤ e_k`, 따라서 `‖x_k−x̂_k‖ ≤ ‖C‖ e_k`.
   `‖M(v_k)‖`은 SVD 없이 `‖A‖+Σ_i|v_{k,i}|‖B_i‖` (사전계산 norm) 으로 상계 → 스칼라 연산.

**(B-3) 주입 (세 방식, 조합 가능)**:
- **(i) risk-penalized stage cost** (메인):
  ```
  S̃^k = φ(Cẑ_T^k)+β_T e_T^k + Σ_t [ q(Cẑ_t^k,v_t^k) + β e_t^k ] + (control terms)
  ```
  softmax가 **인증오차가 커지는 궤적을 down-weight** → bound가 원점서 0이므로 **인증영역으로 정규화**.
- **(ii) constraint back-off** (안전): `h(x_t)≥0` 을 `h(Cẑ_t) ≥ γ‖C‖ e_t` 로 tightening 후 soft-penalty.
- **(iii) CVaR risk measure**: per-step 비용 스프레드 `~ Lip(q)·‖C‖e_k` 로 보고 `E→CVaR_α` (RA-MPPI 연결).

**β→0 이면 (A)로 환원** → (B)는 knob으로 제어되는 엄밀한 일반화.

---

## 6. 보장으로의 다리 (이론, 경량 — 추후 角도 C)

인증집합 `Ω_c={x: 튜브 e_k 유계}`. 주장: error-tube 패널티가 cost-weighting을 `‖r‖` 작은 영역으로 유지
→ bilinear surrogate의 **cost controllability**(Koopman 리뷰) 와 결합 시 실계에 **practical stability / ISS**,
ultimate bound `∝ (c_x,c_u)`. = Koopman-MPC 보장 체인을 **sampling-based MPPI로 이식**. (정식 증명은 별도.)

---

## 7. 실험 설계

**Tier 0 — 검증(해석/기지 bound)**: Van der Pol, pendulum/cartpole swing-up.
  → 튜브 `e_k`가 참 lifted 오차를 실제로 상계하는지, bilinear vs linear 예측오차 비교.
**Tier 1 — bilinear 필수계**: nonholonomic mobile robot(unicycle/Dubins), surface vehicle.
  → 입력이 bilinear로 진입; **bilinear-KMPPI ≫ linear MPPI-DK** 추종성능; error-aware가 제약위반 감소.
**Tier 2 — 확장성/샘플효율**: planar/3D quadrotor.
  → lifted rollout 속도 + 샘플효율(성능 vs K) + 외란/도메인랜덤 하 강건성.
**Tier 3 — (옵션) 하드웨어**: wheeled robot 또는 quadrotor; 혹은 MPPI-DK 따라 quadruped task-space.

**Baselines**:
(a) MPPI w/ 참모델 (oracle 상한), (b) **MPPI-DK** (linear Koopman), (c) bilinear-Koopman-**MPC** (QP/iLQR),
(d) bilinear-KMPPI **w/o bound** (=우리 A, ablation), (e) **ours full (A+B)**.

**Metrics**: closed-loop cost, tracking RMSE, success rate, **constraint-violation rate**,
강건성(파라미터오차·외란·데이터량↓ 하), **rollout wall-time & 제어주파수**, **샘플효율 곡선(cost vs K)**,
open-loop 예측오차 vs horizon.

---

## 8. 기여 목록 (Contributions)

- **C1 (모델)**: MPPI rollout에 **bilinear Koopman** 최초 적용. "샘플링이라 convexity 불필요 →
  Koopman-MPC를 linear에 묶던 제약 해제, bilinear 정확도를 linear급 GPU 비용으로." (vs MPPI-DK linear)
- **C2 (오차인지)**: Koopman **proportional error bound를 MPPI 비용에 주입**(error-tube/risk-aware).
  risk-neutral MPPI에 인증 모델오차를 넣은 최초 사례; 원점-소멸 bound가 안정화 정규화로 작동.
- **C3 (이론)**: bound + cost-controllability → 실계 practical-stability/bounded-error 논증.
  Koopman-MPC 보장을 sampling-based로 이식.
- **C4 (실증)**: 위 baseline/metric로 Tier 0–2(+옵션 HW) 벤치마크.

### vs MPPI-DK — 차별성 (gatekeeper)
선행 MPPI-DK (arXiv 2603.05385, 2026-03): *linear* deep-Koopman `g⁺=Ag+Bu` 를 **속도 목적**으로 MPPI rollout에. 모델오차 무시, 보장 없음.

| 축 | MPPI-DK | RB-KMPPI (ours) |
|---|---|---|
| 모델클래스 | linear deep-Koopman | **bilinear** (Iacob–Tóth: 정확한 lifted의 최소클래스) |
| 클래스 선택이유 | 속도 | sampling ⇒ convexity 불필요 ⇒ bilinear 해금 (동일 matvec 레짐, convex solve 없음) |
| 모델오차 | **무시** (risk-neutral) | **인증 proportional bound 주입** (tube §1 / CVaR §2) |
| 인증 bound 가용성 | ✗ (NN 인증無; linear 잔차는 product항 `~|u|‖Ψ‖` 라 깔끔한 proportional 아님) | ✓ SafEDMD/kernel **bilinear** → 원점소멸 `c_x,c_u` |
| 폐루프 보장 | 없음 (deep-Koopman 오차해석 부재) | practical/asymptotic, ultimate bound ∝(c_x,c_u) (§3,§5) |
| 목표 | 가속 | 정확도+강건성+보장 (속도 **우위주장 안 함**, 아래 비용분석 참조) |

**⚙ 연산비용 (raw 속도 정직하게)**: rollout matvec — linear `Aẑ+Bu` = **1 matvec** vs bilinear `Aẑ+Σᵢuᵢ(Bᵢẑ)` = **(m+1) matvec** ≈ **(m+1)× (iso-N)**.
tier별: pendulum m=1→2×, unicycle m=2→3×, quadrotor m=4→5×, Go1 m≈12→~13×(많이 느림). ⇒ **"raw 속도 동급" 주장 금지.**
방어 3단: (a) **동일 복잡도 레짐**(dense matvec·GPU 병렬·convex solve無 → 실 baseline physics/NMPC 대비 같은 *수십–수백× 빠른 범주*).
(b) **★ iso-accuracy 역전**: 비용∝N², linear은 bilinear 진실 근사에 더 큰 N 필요. 교차 `N_lin/N_bi ⋛ √(m+1)`
→ linear이 √(m+1)배 이상 차원 부풀려야 정확도 맞으면 **iso-accuracy에선 bilinear이 더 빠름** (셀링포인트는 accuracy-per-compute Pareto).
(c) wall-clock < (m+1)× (m+1 matvec 독립 → GPU 배치융합; 단 과장 금물). ⇒ 실험서 **(m+1)× raw 느려짐 + iso-accuracy Pareto 둘 다 보고.**

**결합논리 (incremental 방어)**: C2(오차주입)·C3(보장) 은 C1(bilinear)의 **인증 proportional bound에 올라탐** —
linear deep-Koopman은 (i) 잔차가 깔끔한 proportional 형태 아님, (ii) deep-NN 인증오차해석 부재
⇒ **C2/C3 를 MPPI-DK에 이식 불가**. 패키지 C1+C2+C3 는 분해 불가 한 묶음.
⚠ **C1 단독은 얇음** → 항상 패키지로, 속도 우위로 팔지 말 것(bilinear가 더 느림). 자세한 유도: [[derivation]].

---

## 9. 열린 설계 결정 (확정 필요)
- dictionary 선택: 수동(다항/RBF) vs deep-Koopman(NN) — deep는 bound 이론 미비(주의).
- 비용 평가 공간: original `x=Cz` vs lifted `z` 직접.
- `c_x,c_u` 출처: SafEDMD(확률) vs kernel(결정론) vs 실측 잔차 상위분위수(실용 surrogate).
- 구현 스택: JAX/PyTorch(GPU 병렬 rollout) vs CUDA.
