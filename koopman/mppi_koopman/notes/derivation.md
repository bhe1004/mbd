# Derivation — RB-KMPPI 정밀 수학 유도

[[formulation_AB]] (`notes/formulation_AB.md`) 의 (B) 오차주입·(C) 보장 부분을 엄밀화.
구성: **§1 오차 튜브 재귀(완전 증명)** → **§2 비용오차 & CVaR 주입** → **§3 안정성 정리 스케치** → **§4 미결 항목**.
근거: [[lit_digest]], `ref/` (Strässer et al. proportional bound / cost-controllability, MPPI 리뷰 free-energy).

---

## 0. 표기 (Notation)

| 기호 | 의미 |
|---|---|
| `x∈ℝⁿ, u∈ℝᵐ` | 원 상태·입력. `0∈int𝕌, f(0)=0` |
| `z=Ψ(x)∈ℝᴺ` | lifted 상태. **shifted**: `Ψ(0)=0` |
| sandwich | `ℓ‖x‖ ≤ ‖z‖ = ‖Ψ(x)‖ ≤ L‖x‖`,  `0<ℓ≤L` |
| `C∈ℝ^{n×N}` | 역사영 `x=Cz`,  `c_C:=‖C‖` (`C=[Iₙ 0]`면 `c_C=1`) |
| `M(u):=A+Σ_{i=1}^m u_i B_i` | 입력의존 lifted 천이행렬 |
| `v_k` | rollout에서 실제 적용/샘플된 입력 (`v_k=u_k+ε_k`) |

**참(true) lifted dynamics** (bilinear EDMDc + 잔차):
```
z_{k+1} = M(v_k) z_k + B₀ v_k + r_k                              (0.1)
```
**Nominal rollout** (MPPI가 굴리는 것, 잔차 0):
```
ẑ_{k+1} = M(v_k) ẑ_k + B₀ v_k ,     ẑ_0 = z_0 = Ψ(x_meas)        (0.2)
```
**★ proportional error bound** (Strässer, SafEDMD/kernel; 원점서 소멸):
```
‖r_k‖ = ‖r(x_k,v_k)‖ ≤ c_x ‖z_k‖ + c_u ‖v_k‖                     (0.3)
```
> ⚠️ (0.3)의 `‖z_k‖`은 **참** lifted 상태. rollout에선 `ẑ_k`만 알고 `z_k`는 모름 →
> §1의 튜브가 이 간극(`z_k vs ẑ_k`)을 명시적으로 흡수한다. 이게 재귀에 `c_x e_k` 항이 생기는 이유.

연산비용용 행렬노름 상계 (SVD 불필요, 오프라인 1회 `‖A‖,‖B_i‖` 사전계산):
```
‖M(v)‖ ≤ m̄(v) := ‖A‖ + Σ_{i=1}^m |v_i| ‖B_i‖                     (0.4)
```

---

## 1. 오차 튜브 재귀 (Error-tube recursion)

목표: 참–nominal lifted 편차 `Δ_k := z_k − ẑ_k` 를 **샘플당 스칼라** `e_k` 로 상계하는,
rollout 중 `ẑ_k`만으로 계산 가능한 재귀를 유도한다.

### Lemma 1 (일스텝 오차 동역학)
참(0.1)과 nominal(0.2)은 **같은 입력열** `v_k` 를 쓴다. 차를 빼면
```
Δ_{k+1} = z_{k+1} − ẑ_{k+1}
        = [M(v_k)z_k + B₀v_k + r_k] − [M(v_k)ẑ_k + B₀v_k]
        = M(v_k) Δ_k + r_k .                                      (1.1)
```
`B₀v_k` 항이 정확히 상쇄됨에 유의 (입력 affine 항은 오차에 기여 안 함). ∎

### 정의 (스칼라 튜브)
```
e_0 = 0
e_{k+1} = (m̄(v_k) + c_x) e_k + c_x‖ẑ_k‖ + c_u‖v_k‖               (1.2)
```
(0.4)에 의해 `m̄(v_k) ≥ ‖M(v_k)‖`. 더 타이트하게 쓰려면 `m̄(v_k)→‖M(v_k)‖` 로 대체 가능
(여전히 아래 정리 성립; `m̄`는 스칼라 사전계산용 보수적 상계).

### Theorem 1 (튜브가 lifted 오차를 상계)
모든 `k≥0` 에서  `‖Δ_k‖ ≤ e_k`,  따라서  `‖x_k − x̂_k‖ ≤ c_C e_k`.

**증명 (k에 대한 귀납).**
- *기저*: `Δ_0 = z_0 − ẑ_0 = 0` ⇒ `‖Δ_0‖ = 0 = e_0`. ✓
- *귀납단계*: `‖Δ_k‖ ≤ e_k` 가정. (1.1)에 삼각·부등식:
  ```
  ‖Δ_{k+1}‖ ≤ ‖M(v_k)‖‖Δ_k‖ + ‖r_k‖.                            (a)
  ```
  proportional bound (0.3) 와 `‖z_k‖ = ‖ẑ_k + Δ_k‖ ≤ ‖ẑ_k‖ + ‖Δ_k‖`:
  ```
  ‖r_k‖ ≤ c_x‖z_k‖ + c_u‖v_k‖ ≤ c_x‖ẑ_k‖ + c_x‖Δ_k‖ + c_u‖v_k‖.  (b)
  ```
  (a)+(b):
  ```
  ‖Δ_{k+1}‖ ≤ (‖M(v_k)‖ + c_x)‖Δ_k‖ + c_x‖ẑ_k‖ + c_u‖v_k‖.       (c)
  ```
  계수 `(‖M(v_k)‖+c_x) ≥ 0` (단조성) + `‖M(v_k)‖ ≤ m̄(v_k)` + 귀납가정 `‖Δ_k‖≤e_k`:
  ```
  ‖Δ_{k+1}‖ ≤ (m̄(v_k)+c_x) e_k + c_x‖ẑ_k‖ + c_u‖v_k‖ = e_{k+1}. ✓
  ```
원좌표: `‖x_k − x̂_k‖ = ‖C Δ_k‖ ≤ ‖C‖‖Δ_k‖ ≤ c_C e_k`. ∎

> **핵심**: (b)에서 모르는 `‖z_k‖` 을 알고 있는 `‖ẑ_k‖` + 튜브 `e_k` 로 치환한 것이
> 재귀계수의 `+c_x` 와 소스항 `c_x‖ẑ_k‖` 를 동시에 만든다. 튜브는 **온라인 계산 가능**
> (`ẑ_k, v_k` 와 사전계산 norm만 필요) — 이게 MPPI 비용에 바로 꽂을 수 있는 이유.

### Prop 1 (닫힌형 + 균일 상계)
`a_k := m̄(v_k)+c_x`,  `b_k := c_x‖ẑ_k‖ + c_u‖v_k‖ ≥ 0` 로 두면 (1.2)는 선형 시변 재귀
`e_{k+1}=a_k e_k + b_k`, `e_0=0`. 닫힌형:
```
e_k = Σ_{j=0}^{k-1} ( Π_{l=j+1}^{k-1} a_l ) b_j        (빈 곱 = 1)   (1.3)
```
*해석*: 각 스텝의 잔차 소스 `b_j` 가 이후 증폭계수곱 `Π a_l` 만큼 키워져 누적.

**균일 상계 (contractive 가정)**: 만약 관련 입력영역에서 `a_k ≤ ā < 1` (즉 `m̄(v_k) < 1−c_x`),
그리고 `b_k ≤ b̄` 면
```
e_k ≤ b̄ Σ_{i=0}^{k-1} ā^i ≤ b̄ / (1−ā) =: e_∞ .                  (1.4)
```
`b̄`는 `c_x,c_u`에 비례 ⇒ **모델 완전(`c_x,c_u→0`)이면 튜브 →0**. (0.3)의 원점소멸이
여기서 ultimate-bound 소멸로 직결.

### Remark 1 (보수성 & 가중노름 정련)
`m̄(v)`(절댓값 합)·연산자노름 contractivity (1.4)는 **보수적** (대부분 안정계는 연산자노름 수축 아님).
정련: `∃P≻0, ρ<1: M(v)ᵀP M(v) ⪯ ρ²P  ∀v∈𝕍` 이면 `‖·‖_P:=‖P^{1/2}·‖` 에서
`‖Δ_{k+1}‖_P ≤ ρ‖Δ_k‖_P + ‖P^{1/2}r_k‖`, 잔차를 `‖P^{1/2}r_k‖ ≤ √κ(P)(c_x‖z‖+c_u‖v‖)` 로 환산
(`κ(P)=조건수`). 균일상계 `e_∞^P = √κ(P)·b̄/(1−ρ)`. 실험에선 P를 식별후 LMI로 구해 타이트화 가능.
(메인 유도는 가독성 위해 Euclidean 유지; 실측 잔차 분위수로 `c_x,c_u` 대체하는 실용판도 가능.)

---

## 2. 비용오차 & CVaR 주입 (Risk-aware)

튜브 `e_k` → 비용 불확실성으로 환산 → MPPI softmax에 risk-aware 비용으로 주입.

### Lemma 2 (스테이지 비용 오차)
`q(·,u)` 가 `x`에 Lipschitz (상수 `L_q`), `φ` Lipschitz (`L_φ`) 가정. 한 샘플 궤적에 대해
참 비용 `S` vs nominal `Ŝ`:
```
|q(x_k,v_k) − q(x̂_k,v_k)| ≤ L_q‖x_k−x̂_k‖ ≤ L_q c_C e_k          (2.1)
|φ(x_T) − φ(x̂_T)|        ≤ L_φ c_C e_T
```
⇒ 전체 비용오차 균일상계
```
|S − Ŝ| ≤ ΔS := c_C ( L_φ e_T + Σ_{k=0}^{T-1} L_q e_k ).          (2.2)
```
(2차형 비용 `q=‖x‖²_Q` 면 국소 Lipschitz `L_q(x̂)=2‖Q‖(‖x̂_k‖+c_C e_k)` 로 더 타이트.) ∎

### 2.1 주입식 (A): worst-case = additive penalty
CVaR 신뢰수준 `α→0` (최악) 한계에서 `S_CVaR = Ŝ + ΔS`. (2.2) 대입하면 정확히
formulation (B-3-i)의 **risk-penalized stage cost** 가 복원됨:
```
S̃^k = φ(Cẑ_T^k) + β_T e_T^k + Σ_{k} [ q(Cẑ_k^k, v_k^k) + β e_k^k ] + (control)   (2.3)
      with   β = L_q c_C ,   β_T = L_φ c_C .
```
즉 (i) additive penalty 와 (iii) CVaR 는 **같은 대상의 두 한계** — penalty는 CVaR의 `α→0` 코너.
`β` 가 손튜닝이 아니라 `L_q·c_C` 로 **유도**됨에 주목.

### 2.2 주입식 (B): 분포형 CVaR_α  (α∈(0,1))
**규약**: `CVaR_α(Y)` = 상위 `α` 최악 꼬리의 평균 (α작을수록 risk-averse).

잔차를 영평균·등방 surrogate `r_k ≈ 0, Cov(r_k) ⪯ s_k² I`, `s_k := c_x‖ẑ_k‖+c_u‖v_k‖` 로 모델링하고
(1.1)을 2차모멘트로 전파 (`Δ_k`–`r_k` 독립 surrogate):
```
P_0 = 0 ,   P_{k+1} = M(v_k) P_k M(v_k)ᵀ + s_k² I .              (2.4)
```
스테이지 비용 분산 (1차 전개): `σ_{q,k}² ≈ (∇_x q)ᵀ C P_k Cᵀ (∇_x q)`, 누적 `σ_S² = Σ_k σ_{q,k}² + σ_{φ,T}²`.
Gaussian surrogate 하 CVaR:
```
CVaR_α(S^k) = Ŝ^k + κ(α) σ_S^k ,   κ(α) = ϕ(Φ⁻¹(1−α)) / α .       (2.5)
```
(`ϕ,Φ` 표준정규 pdf/cdf. `α→0⁺` 면 `κ→∞` → worst-case (2.3)와 정성적 일치.)

> `P_k` 는 `M P Mᵀ` 전파라 `e_k`(노름 튜브)보다 덜 보수적 (방향성 반영). 단 잔차 독립성은
> **모델링 가정**(진짜 잔차는 결정론적) — 그래서 (2.4)는 surrogate로 명시. 보수적 보장은 §1 `e_k`,
> 평균적 risk-shaping은 (2.5) `P_k`, 라는 역할 분담.

### 2.3 MPPI 업데이트 (risk-aware)
어느 주입이든 비용을 `S_risk^k ∈ {S̃^k (2.3),  CVaR_α(S^k) (2.5)}` 로 두고
```
w^k = softmax_k( −(S_risk^k − ρ)/λ ) ,   ρ=min_k S_risk^k ,
u_t ← u_t + Σ_k w^k ε_t^k .                                       (2.6)
```
⇒ **인증 모델오차가 큰 궤적을 softmax가 down-weight**. 이게 "process noise가 아니라 *Koopman 모델오차*
에서 위험이 나오는 RA-MPPI" — 두 리뷰를 잇는 새 다리. `β=0` (또는 `α→1`) 이면 (A) 순수 KMPPI로 환원.

---

## 3. 안정성 정리 스케치 (Practical stability)

주장: **proportional bound + bilinear surrogate의 cost controllability + (2.3) tube penalty**
⇒ 실 폐루프의 **practical asymptotic stability**, ultimate bound `∝ (c_x,c_u)`.
Koopman-MPC 보장체인(Strässer)을 sampling-based MPPI로 이식. *형식 증명 아닌 스케치* (개구간 §4 명시).

### 가정
- **(A1) Cost controllability** (Grüne류, Koopman 리뷰): nominal lifted 문제의 최적가치함수
  `V_T(z)` 가 `α₁‖z‖² ≤ V_T(z) ≤ γ ℓ*(z)`, `ℓ*(z)=min_u q_z(z,u) ≥ α‖z‖²`. (terminal 없는 MPC 표준조건.)
- **(A2) MPPI 준최적성 간극**: (2.6)이 내는 입력의 가치가 최적대비 `V_T^{MPPI}(z) ≤ V_T(z) + Δ_opt`,
  `Δ_opt = Δ_opt(K,λ,Σ)` 유한 (샘플수 K↑·제안분포 적합도↑ 에서 ↓). ← **이 부분이 진짜 개구간**.
- **(A3) Lipschitz 가치함수**: `|V_T(z)−V_T(z')| ≤ L_V‖z−z'‖`.
- **(A4) 모델오차**: (0.3) proportional bound + (sandwich) 성립.

### Lyapunov 하강 부등식
nominal `V_T` 를 후보로. 표준 MPC 1스텝 하강 (nominal):
`V_T(ẑ_{k+1}) − V_T(z_k) ≤ −(1−1/γ) ℓ*(z_k)`. 실 다음상태는 `z_{k+1}=ẑ_{k+1}+ (z_{k+1}-ẑ_{k+1})`,
편차는 **1스텝 잔차** `‖z_{k+1}−ẑ_{k+1}‖ = ‖r_k‖ ≤ c_x‖z_k‖+c_u‖v_k‖` (튜브 1스텝, §1). (A3):
```
V_T(z_{k+1}) ≤ V_T(ẑ_{k+1}) + L_V‖r_k‖ ≤ V_T(ẑ_{k+1}) + L_V(c_x‖z_k‖+c_u‖v_k‖).
```
합치고 (A2) 준최적성 간극 `Δ_opt` 흡수, `ℓ*≥α‖z‖²`, 입력 `‖v_k‖≤L_u‖z_k‖+e_u` (제어법 유계) 가정:
```
V_T(z_{k+1}) − V_T(z_k)
   ≤ −(1−1/γ) α‖z_k‖²  +  L_V(c_x + c_u L_u)‖z_k‖  +  L_V c_u e_u + Δ_opt .   (3.1)
                └ 음의 2차(안정화) ┘   └ 모델오차 1차(원점서 소멸) ┘  └ 상수 floor ┘
```

### 결론 (practical asymptotic stability)
(3.1) 우변은 `‖z_k‖` 가
```
‖z_k‖ > R := [ L_V(c_x+c_u L_u) + √(...) ] / [2(1−1/γ)α]  ~  O(c_x, c_u, √Δ_opt)   (3.2)
```
밖이면 음수 ⇒ `V_T` 감소 ⇒ 궤적은 sublevel set `{V_T ≤ b(R)}` 로 유한시간 진입 후 **불변**.
sandwich (`‖z‖≥ℓ‖x‖`) 로 원좌표 ultimate bound `‖x‖ ≲ R/ℓ`. 따라서:

> **(Sketch) Thm 3.** (A1)–(A4) 하에서 RB-KMPPI 폐루프는 반지름 `O(c_x,c_u,√Δ_opt)` 의
> ultimate bound 로 **practically asymptotically stable**. `c_x,c_u→0` (모델완전) & `Δ_opt→0`
> (샘플 충분) 이면 반지름→0 ⇒ asymptotic stability. terminal ingredient 추가 시 `1/γ` 항 제거되어
> 더 강한 보장.

**(2.3) tube penalty의 역할**: penalty `βe_k` 가 softmax 가중을 잔차소스 `c_x‖ẑ‖+c_u‖v‖` 작은 영역으로
유지 ⇒ (3.1)의 1차 교란항 실효계수를 **수축** = `R` 축소. 즉 penalty는 ultimate bound 를 줄이는
정규화로 작동 (formulation §6 주장의 정량화).

### Remark 3 (대안 증명경로: Robust-MPPI free-energy)
MPPI 리뷰의 Robust-MPPI 는 *실* 자유에너지 증가에 형식적 bound 를 줌.
잔차 `r_k` 를 외란으로 보고 `F_real − F_nom ≤ g(c_x,c_u)` (proportional bound로 `g` 산출) 를 세우면
(A2) 의 sampling 간극을 우회해 free-energy 차분으로 직접 ISS 를 논할 수 있음 — §4 과제.

---

## 4. 미결 항목 (해야 할 정식화)

1. **(A2) MPPI 준최적성 간극 `Δ_opt(K,λ,Σ)` 의 명시적 bound** — 가장 약한 고리. **→ §5 에서 착수.**
2. **가중노름 정련** (Remark 1) 으로 (1.4)·(3.1) 의 연산자노름 보수성 제거 → P-LMI 조건으로. **→ §6 에서 착수.**
3. **CVaR (2.4) 잔차 독립 surrogate 정당화** — 실측 잔차 자기상관 측정 후 보정항 or 결정론적 over-bound. **→ §7-R2 에서 격리.**
4. **cost controllability (A1) 의 bilinear-Koopman 검증** — `γ` 를 식별모델에서 수치추정/인증.
5. **(3.2) `R` 상수 명시화** — `L_V, α, γ, L_u, e_u` 를 sandwich(`ℓ,L`)·식별량으로 환원.
6. 수치검증: Tier 0 (Van der Pol/pendulum) 에서 `e_k` 가 참 `‖Δ_k‖` 상계인지, `R` 예측이 실 ultimate bound 와 맞는지.

---

## 5. (A2) MPPI 준최적성 간극 `Δ_opt(K,λ,Σ)` 정량화  [open item 1 — 약한 고리]

§3 (A2)의 가장 약한 고리를 free-energy/SNIS 로 정량화.

### 5.1 Setup
IT-MPPI: 제안분포 `q=N(û, Σ̄)`, `Σ̄=blkdiag(Σ,…,Σ)` (T스텝), warm-start 평균 `û`.
목표분포 `q*(V) ∝ exp(−S(V)/λ) q(V)`. 적용입력
```
U_MPPI = Σ_{k=1}^K w̃_k V_k ,   w̃_k = exp(−S(V_k)/λ)/Σ_j exp(−S(V_j)/λ) ,  V_k~q
```
는 `μ* := E_{q*}[V]` 의 **self-normalized importance sampling (SNIS)** 추정량.
참 결정론적 최적 `U° := argmin_U J(z,U)` 로의 간극을 둘로 분해:
```
‖U_MPPI − U°‖ ≤ ‖U_MPPI − μ*‖   +   ‖μ* − U°‖
                └ 유한표본(SNIS) ┘   └ 온도·Gaussian 편향 ┘
```

### 5.2 유한표본 항 (SNIS)
표준 SNIS MSE bound (Agapiou et al. 2017). weight `w(V)=exp(−S(V)/λ)`:
```
E‖U_MPPI − μ*‖² ≤ (ρ_χ / K)·tr(Σ_{q*}) ,   ρ_χ := 1+χ²(q*‖q) = E_q[w²]/(E_q[w])² = K/ESS
```
- `ρ_χ` = 정규화가중 2차모멘트 = `K/ESS` (`ESS`=effective sample size).
- **λ 의존**: λ↓ → `w=exp(−S/λ)` 첨예 → 가중분산↑ → `ρ_χ↑` (ESS↓, weight degeneracy).
- **Σ 의존**: Σ↑ → 탐험폭↑ → q* 지지 커버↑(편향↓) 그러나 `tr(Σ_{q*})↑` (per-sample 분산↑).
⇒ `E‖U_MPPI−μ*‖ ≤ √( ρ_χ(λ,Σ)·tr(Σ_{q*}) / K )`  = **`O(1/√K)`**, 계수 = `ρ_χ`.

### 5.3 편향 항
`μ*=E_{q*}[V]` = softmax 가중평균. λ→0 에서 q* 는 지지 내 `argmin S` 로 집중 → `μ*→U°`
(단 `U°` 가 q 지지 안, 즉 Σ 충분히 넓어야). ⇒ `b(λ,Σ):=‖μ*−U°‖` 는 **λ→0 에서 →0**;
LQ-Gaussian 특수경우 `b=O(λ)` (q* Gaussian, 평균이 최적쪽으로 O(λ) 이동).

### 5.4 값 간극 Δ_opt
`U°` 내부최적(`∇_U J(z,U°)=0`) + J `L_∇`-smooth ⇒ `Δ_opt ≤ (L_∇/2)E‖U_MPPI−U°‖²`.
입력제약 활성(`U°` 경계) ⇒ J `L_J`-Lipschitz ⇒ `Δ_opt ≤ L_J E‖U_MPPI−U°‖` (**보수적, 권장**). 결합:
```
E[Δ_opt]  ≲  C·( b(λ,Σ)²  +  ρ_χ(λ,Σ) tr(Σ_{q*}) / K )        (smooth, 내부최적)
          ≲  C·( b(λ,Σ)   +  √( ρ_χ tr(Σ_{q*}) / K ) )         (Lipschitz, 제약활성)
```

### 5.5 §3 으로 환류 + 트레이드오프
(3.2) `R ∝ √Δ_opt` ⇒ ultimate bound 가 **compute budget 의 명시적 함수**:
```
R(K,λ,Σ) ∝ √( [c_x,c_u 항]  +  Δ_opt(K,λ,Σ) ) ,   Δ_opt → 0  as  K→∞, λ→λ*(K)→0
```
- **λ 트레이드오프**: λ↓ ⇒ 편향 `b↓` 이나 `ρ_χ↑` (분산↑) ⇒ **최적 `λ*(K)` 존재**, K↑ 에 따라 λ*↓ 허용.
- **K**: 항상 개선(`1/K` or `1/√K`). **Σ**: 탐험–분산 절충.
⇒ "샘플 충분 + 온도 적정 → 실 폐루프 asymptotic stability" 가 **정량 명제**로. **약한 고리 닫힘.**

### Remark 5 (Robust-MPPI 경로 — U-space smoothness 회피, 메인승격 후보)
대안: 실 자유에너지 `F_real ≤ E_{q*}[S]+λ KL(q*‖q)` 에서 모델오차 기여를 proportional bound 로 직접:
```
F_real − F_nom ≤ L_F·E_{q*}[ Σ_k‖r_k‖ ] ≤ L_F Σ_k ( c_x‖z_k‖ + c_u‖v_k‖ )
```
(Williams Robust-MPPI 템플릿). 값함수 smoothness `L_∇` 가정 없이 **cost-space ISS 직접** → §3 보다 깔끔.
§7-R3 (penalty=인증 상계 최소화) 와 정합 ⇒ **장기적으로 이 경로를 메인 증명으로 승격 권장.**

---

## 6. 가중노름 튜브 정련 (P-LMI)  [open item 2]

§1 (1.4)·§3 (3.1) 의 연산자노름 contractivity (`m̄(v)<1−c_x`, 대부분 안정계 위반) 를 *진짜 수축률* 로 교체.

### 6.1 P-노름 튜브
`P≻0`, rate `ρ<1` 가 `M(v)ᵀP M(v) ⪯ ρ²P  ∀v∈𝕍` (즉 `‖M(v)ξ‖_P ≤ ρ‖ξ‖_P`, `‖ξ‖_P:=√(ξᵀPξ)`)
를 만족한다 하자. (1.1) `Δ_{k+1}=M(v_k)Δ_k+r_k` 에 P-노름:
```
‖Δ_{k+1}‖_P ≤ ρ‖Δ_k‖_P + ‖r_k‖_P .
```
잔차 환산 (`κ_P:=√(λ_max(P)/λ_min(P))`, `‖z_k‖≤‖ẑ_k‖+‖Δ_k‖`, `‖Δ_k‖≤‖Δ_k‖_P/√λ_min(P)`):
```
‖r_k‖_P ≤ √λ_max(P)( c_x‖z_k‖+c_u‖v_k‖ ) ≤ κ_P c_x‖Δ_k‖_P + √λ_max(P)( c_x‖ẑ_k‖+c_u‖v_k‖ ) .
```
⇒ **P-튜브** (`ē_0=0`):
```
ē_{k+1} = (ρ + κ_P c_x) ē_k + √λ_max(P)( c_x‖ẑ_k‖ + c_u‖v_k‖ ) ,
‖Δ_k‖_P ≤ ē_k ,   ‖x_k−x̂_k‖ ≤ c_C ē_k/√λ_min(P) .
```

### 6.2 균일상계 & 보수성 비교
contractive 조건이 **`ρ + κ_P c_x < 1`** 로 완화 (연산자노름 `m̄<1−c_x` 대신 *실제 수축률* ρ):
```
ē_∞ = √λ_max(P)·b̄ / ( 1 − ρ − κ_P c_x ) .
```
- ρ = M 의 스펙트럼적 수축(=실 안정성) 반영 → `m̄`(절댓값 합)보다 훨씬 작게 가능.
- 대가: `κ_P` 가 `c_x` 에 곱해짐 + 소스에 `√λ_max(P)`. P ill-conditioned 면 `κ_P c_x` 가 지배.
- ⇒ **ρ 최소화 ↔ κ_P 최소화 절충** 을 LMI 목적에 동시 반영.

### 6.3 인증 LMI (Schur + 꼭짓점)
`M(v)=A+Σv_iB_i` 가 v에 affine ⇒ `M(v)ᵀP M(v)⪯ρ²P` 를 Schur 보수로 **v-affine** 화:
```
[ ρ²P        M(v)ᵀ P ]
[ P M(v)        P     ] ⪰ 0 .
```
𝕍 가 박스(폴리토프)면 affine성 → **꼭짓점 `v^(j)` 에서만** 검사 충분:
```
find P≻0,  minimize ρ²  (또는 ρ + κ_P c_x)
s.t.  [ ρ²P , M(v^(j))ᵀP ; P M(v^(j)) , P ] ⪰ 0   ∀ vertex j ,
      I ⪯ P ⪯ κ_P² I    (κ_P 제어) .
```
ρ 에 대해 **bisection (quasi-convex SDP)**. 식별 직후 1회 오프라인 → `ρ,κ_P,λ_max(P)` 확정,
온라인 튜브 `ē_k` 는 여전히 스칼라 재귀.

### 6.4 §3 환류
Lyapunov 후보를 `V_P(z)=zᵀPz` (또는 P-가중 값함수) 로 → (3.1) 하강 교란계수가 `‖M‖` 대신 `ρ`
→ `R` 의 연산자노름 보수성 제거, 1차교란항 스케일을 `√λ_max(P)·κ_P` 로 명시화.

### 6.5 수치검증 (verify/plmi_tube.py, Docker)
- **합성 bilinear계** (비정규 안정 A, `‖A‖₂=2.3>1, spec.rad=0.85`, **c_x=0.001 통제**): Euclid 튜브 폭발(`e_max≈1e12`)
  vs P-LMI 튜브 유계·수축 (`coef=ρ+κ_P c_x=0.96<1`, ρ는 *최소화 아닌* coef 최소화로 선택), **≈4×10¹¹배 tighter**. ⇒ **§6 메커니즘 검증.**
- **실계** (pendulum_sdg/VdP, vanilla 최소제곱 EDMDc): (i) vanilla A 가 `spec.rad>1` (물리적 안정계인데도!) → P-LMI infeasible;
  (ii) 고유값 클리핑으로 안정화해도 `c_x≈0.25` → `coef=ρ+κ_P c_x≈1.7–2.7 > 1` → **비수축**.
- **정량 요구**: 수축 튜브엔 `c_x ≲ (1−ρ)/κ_P ≈ 0.001–0.01` (vanilla 대비 ~25–250×↓) 필요 = **SafEDMD/kernel `c_x=O(1/√d+Δt²)` 영역**.
- **★ 본질적 긴장 (§6.2 caveat 실측확인)**: `‖A‖₂>1` 을 만드는 **비정규성**이 동시에 수축메트릭 P를 ill-conditioned(`κ_P↑`)로 만들어
  `κ_P c_x` 를 키움. ⇒ RB-KMPPI 보장체인은 **SafEDMD급 식별(작은 c_x)이 필수**, vanilla EDMD로는 튜브가 수축하지 않음.
  → 이는 formulation §9 "`c_x,c_u` 출처" 결정과 [[ref-mppi-koopman-literature]] 의 SafEDMD 선택을 **실증적으로 정당화**.

---

## 7. Adversarial 검토 (증명 빈틈 점검)  [open item 3]

§1–§6 자기 red-team. **코어 결과는 견고**; 아래는 조건/주의. (✓ 확인 · ⚠ 조건부 · ✗ surrogate · ↑ 보강)

**✓ 견고 (재확인)**
- **Lemma 1 / Theorem 1 / Prop 1**: 귀납·닫힌형(k=1,2 검산)·균일상계 통과. 단조성(계수≥0)·`‖M‖≤m̄` 치환 방향 정확. 차원정합 OK (`e_k`~‖z‖, `βe_k`~비용).
- **Lemma 2**: 같은 `v^k` → 제어비용 상쇄, 상태비용만 차 → `|S−Ŝ|≤ΔS` 유효.
- **§2.1 worst-case = CVaR(α→0)**: 규약 일치(α↓=averse, esssup), penalty `β=L_q c_C` **유도값** 확인.

**⚠ 조건부 (명시 필요)**
- **R1 (regional·probabilistic bound)**: (0.3) 은 **인증영역** `x∈X_cert,u∈𝕌` 에서만, SafEDMD는 **w.p.≥1−δ**.
  ⇒ Theorem 1 은 "궤적이 X_cert 이탈 전까지" 조건부 + horizon union bound로 **w.p.≥1−Tδ**.
  → 실험서 이탈 모니터, kernel(결정론) 버전이면 δ 제거.
- **R4 (입력 소멸 가정)**: §3 의 `‖u_0‖≤L_u‖z‖+e_u`. `e_u=0`(원점서 입력→0, 안정화제어) ⇒ 교란 원점서 완전소멸 ⇒ **asymptotic**.
  `e_u>0` ⇒ floor `∝c_u e_u` ⇒ **practical** 만. (`‖u‖≤u_max` 상수 floor 쓰면 asymptotic 잃음 — affine 모델이 옳음.)
- **R5 (값 간극 차수)**: §5.4 — 내부최적이면 quadratic, 제약활성이면 Lipschitz 선형판(**보수적 채택 권장**).

**✗ surrogate (격리)**
- **R2 (공분산 튜브 2.4 비엄밀)**: `P_{k+1}=MP_kMᵀ+s_k²I` 는 잔차를 영평균·독립 가정한 **surrogate** —
  진짜 `r_k` 는 결정론적·`Δ_k` 상관. ⇒ `P_k` 는 상계 아님(과소추정 가능). **안전제약엔 §1 `e_k`(엄밀 상계)만,
  risk-shaping 엔 `P_k`** 역할분담 엄수. (또는 결정론적 over-bound 으로 대체 — open.)

**↑ 강화 (버그 아님, 논거 보강)**
- **R3 (penalty = 인증 상계 최소화)**: (2.3) `S̃=Ŝ+βe_k` 는 `βe_k≥|S−Ŝ|` (Lemma 2) ⇒ **`S̃≥S`**, 참비용의 인증 상계.
  MPPI가 S̃ 최소화 = 참비용 상계 최소화 = **minimax/robust**. ⇒ 상계 위 descent 가 참비용 하강 함의.
  **Remark 5 (free-energy) 와 결합해 메인 증명으로 승격 권장.**
- **R6 (연산자노름 보수성)**: (1.4) `m̄<1−c_x` 과보수 → **§6 P-LMI 가 정식 해소** (`ρ<1−κ_P c_x`).

**종합**: 코어 정리(튜브·penalty 유도) 정확. 보강 3곳 = {R1 regional 명시, R2 공분산 surrogate 격리, Δ_opt(§5,λ-trade)}.
메인 증명경로는 **Remark 5(free-energy) + R3(인증 상계) 결합** 이 §3 의 `L_∇`-smoothness 를 피하며 가장 깔끔. **→ §8 에서 승격.**

---

## 8. Free-energy 기반 안정성 정리 (Remark 5 승격, `L_V`/`L_∇`-free)

§3 은 실 다음상태 교란을 `L_V‖r‖` (값함수 Lipschitz) 로 흡수했다. 여기선 **Robust-MPPI free-energy** 로
그 가정을 제거하고 proportional bound 를 **free-energy mismatch** 로 직접 주입한다. ← 권장 메인 증명.

### 8.1 Free-energy 값함수
base(무제어 노이즈) 분포 `p`, 온도 λ. 입력열 V 의 T-horizon 비용을 `S_•(V;z)`
(•=real 참모델 / nom 표면모델, 같은 V·같은 `z_0=z`) 라 할 때:
```
F_•(z) := −λ log E_{V~p}[ exp(−S_•(V;z)/λ) ] .
```
`F_nom` = MPPI softmax 가 실제로 최소화하는 양; `F_real` = 참 폐루프 비용 지배.

### 8.2 ★ Lemma (free-energy 모델오차 bound — novel core, 엄밀)
샘플지지 위(입력클립 `v∈𝕌`·`x∈X_cert`)에서 `ΔS_max(z):=sup_V |S_real(V;z)−S_nom(V;z)|` 라 하면
```
|F_real(z) − F_nom(z)| ≤ ΔS_max(z) ,                                  (8.1)
ΔS_max(z) ≤ c_C( L_φ e_T + L_q Σ_{k=0}^{T-1} e_k )   [Lemma 2 + 튜브 §1]  (8.2)
```
**증명 (8.1)**: 점별 `S_real ≥ S_nom − ΔS_max` ⇒ `e^{−S_real/λ} ≤ e^{ΔS_max/λ} e^{−S_nom/λ}`.
`E_p` 단조 + `−λ log(·)` 단조 ⇒ `F_real ≥ F_nom − ΔS_max`. 대칭(`S_real ≤ S_nom+ΔS_max`) ⇒ 역부등식 ⇒ (8.1). ∎
- **핵심**: 값함수 Lipschitz(`L_V`) **불요** — 비용 Lipschitz(`L_q,L_φ`, Lemma 2 에서 이미 가정)만으로 충분.
- (8.2)의 `e_k` 는 §1 튜브 ⇒ `ΔS_max=O(c_x,c_u)`, **원점서 소멸** (ẑ→0,v→0 이면 e_k→0).

### 8.3 Thm (free-energy ISS — practical stability)
가정: (A1) 표면모델 cost-controllability `V̂_T(z)≤γℓ*(z)`, `ℓ*≥α‖z‖²`; (A2′) free-energy 온도·표본 gap
`|F_nom(z)−V̂_T(z)| ≤ δ_λ+δ_K`, `δ_λ=O(λ)`(log-sum-exp 온도편향), `δ_K=O(√(ρ_χ/K))`(§5 SNIS); (A4) proportional bound.
그러면 `F_real` 이 실 폐루프 ISS-Lyapunov:
```
F_real(z⁺) − F_real(z) ≤ −(1−1/γ) ℓ*(z) + 2ΔS_max(z) + 2(δ_λ+δ_K) .    (8.3)
```
*증명 스케치*: (A1)+Williams Robust-MPPI 하강 ⇒ `F_nom(ẑ⁺)−F_nom(z) ≤ −(1−1/γ)ℓ*(z)+δ_λ+δ_K`;
실/표면 상태 양끝에 (8.1) 두 번 적용 (`F_real(·)≤F_nom(·)+ΔS_max(·)`) ⇒ (8.3). 값함수 연속성 대신 (8.1)이 교란 흡수. ∎
**결론**: `ΔS_max(z) ≤ σ‖z‖+d` (`σ,d=O(c_x,c_u)`, `e_u=0`이면 `d→0`) ⇒ (8.3) 우변은
`‖z‖ > R_F := O(c_x,c_u,λ,K^{-1/2})/[(1−1/γ)α]` 밖에서 음수 ⇒ **practical asymptotic stability**, ultimate bound `R_F`.
`c_x,c_u→0, λ→0, K→∞` 이면 `R_F→0`. **`L_V`(§3)·`L_∇`(§5) 가정 모두 제거.**

### 8.4 §3 대비 (무엇이 엄밀해졌나)
| | §3 (Lyapunov-값함수) | §8 (free-energy) |
|---|---|---|
| 모델오차 흡수 | `L_V‖r‖` (값함수 Lipschitz) | **(8.1) free-energy bound** (비용 Lipschitz만) |
| MPPI 준최적성 | `Δ_opt` via `L_∇` (값함수 smooth) | `δ_λ+δ_K` via free-energy gap (smooth 불요) |
| 남는 가정 | A1 + L_V + L_∇ + A2 | **A1 + A2′(free-energy gap) + A4** |

**Novel = Lemma 8.2** (proportional bound → free-energy 다리). 하강·controllability 는 Williams/Grüne 인용.
R3(penalty=인증 상계)와 결합 시 `S̃≥S` ⇒ `F_real` 이 상계 위에서 하강 → 참비용 하강 자동.
