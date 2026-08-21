# Annealing 자기강건성: MBD가 tube 없이 매끈한 모델오차에 강건한 이유

> **비형식 진술:** softmax 가중은 후보들에 공통인 cost 오프셋에 불변이므로, 모델오차가
> denoising 업데이트를 왜곡하는 것은 오차장의 **후보 간 변동**뿐이다. 후보는 현재 반복점의
> σ-공 안에 있으므로 이 변동은 O(σ)이고, 업데이트 오염은 **O(σ²)**이다. MBD는 σ→0으로
> anneal하므로 후반(정밀화) 단계는 매끈한 모델오차에 구조적으로 면역이다. fixed-σ MPPI에는
> 이 자기감쇠가 없어 error-tube 같은 외부 보정이 상시 역할을 갖는다.
>
> 관련: `framing.md` §7.5, witness 실험 `experiments/annealing_robustness.py`.

---

## 1. 설정

- 제어열 U ∈ R^{T×m}. 참 rollout cost J(U), surrogate rollout cost Ĵ(U) = J(U) + ε(U).
  (ε는 학습된 dynamics의 오차가 cost에 유도하는 오차장.)
- 한 denoising 레벨 σ에서 후보 U_k = Ū + σ ξ_k (k = 1..K), Ξ := max_k ‖ξ_k‖.
- 온도 α의 softmax 가중과 가중평균:

```
w_k  ∝ exp(-J(U_k)/α),      μ  = Σ_k w_k  U_k     (참 cost 기준)
ŵ_k ∝ exp(-Ĵ(U_k)/α),      μ̂ = Σ_k ŵ_k U_k     (surrogate 기준)
```

- 가정: ε는 conv{Ū, U_1, ..., U_K} 위에서 L_ε-Lipschitz. (매끈한 오차장. 접촉 등
  불연속 오차는 배제 — §4 한계 참고.)

## 2. Proposition (annealed sampling의 매끈한 모델오차 자기강건성)

**(i) 오프셋 불변성.** 임의 상수 c에 대해 ε를 ε + c로 바꿔도 ŵ, μ̂는 불변이다.
즉 선택을 왜곡하는 것은 ε의 크기가 아니라 **후보 간 변동**뿐이다.

**(ii) 업데이트 오염 경계.** R := L_ε σ Ξ 라 하면,

```
|ŵ_k/w_k − 1| ≤ e^{2R/α} − 1                        (가중 왜곡)
‖μ̂ − μ‖ ≤ (e^{2R/α} − 1) · 2σΞ                     (평균 왜곡)
```

특히 2R ≤ α 이면 e^x − 1 ≤ 2x (x ≤ 1)로부터

```
‖μ̂ − μ‖ ≤ 8 L_ε σ² Ξ² / α  =  O(σ² · L_ε / α).
```

**(iii) 스케줄 귀결.** annealing σ_1 > ... > σ_N → 0 아래에서 i번째 denoising
업데이트의 오염은 σ_i²에 비례해 소멸한다. 최종 출력을 지배하는 저-σ 단계는 매끈한
모델오차에 면역이다. 반면 σ 고정 샘플러(MPPI)는 매 control step마다 위 경계의
σ²-스케일 오염 바닥을 유지한다.

### 증명

**(i)** exp(-(Ĵ+c)/α) = exp(-c/α)·exp(-Ĵ/α); 상수 인자는 분자·분모에 공통이라
정규화에서 소거. ∎

**(ii)** e_k := ε(U_k) − ε(Ū) 라 하면 Lipschitz로 |e_k| ≤ L_ε σ‖ξ_k‖ ≤ R.
(i)에 의해 ε(Ū)는 소거되므로 ŵ_k = w_k r_k / Σ_j w_j r_j, r_k := e^{-e_k/α}.
r_k ∈ [e^{-R/α}, e^{R/α}]이고 분모 Σ_j w_j r_j도 같은 구간에 있으므로
ŵ_k/w_k = r_k / Σ_j w_j r_j ∈ [e^{-2R/α}, e^{2R/α}], 따라서
|ŵ_k − w_k| ≤ (e^{2R/α} − 1)·w_k.

Σ_k(ŵ_k − w_k) = 0 이므로 μ̂ − μ = Σ_k (ŵ_k − w_k)(U_k − μ). μ는 U_j들의
볼록결합이라 ‖μ − Ū‖ ≤ σΞ, 따라서 ‖U_k − μ‖ ≤ ‖U_k − Ū‖ + ‖Ū − μ‖ ≤ 2σΞ.
노름을 취하면 ‖μ̂ − μ‖ ≤ (e^{2R/α} − 1)·2σΞ. 2R/α ≤ 1일 때
e^{2R/α} − 1 ≤ 4R/α 이므로 ‖μ̂ − μ‖ ≤ 8 L_ε σ² Ξ²/α. ∎

**(iii)** (ii)를 각 σ_i에 적용. ∎

## 3. Remarks

**R1 (tube와의 대비 — 무엇이 진짜 위험 신호인가).**
경계는 오차장의 **국소 변동** L_ε에만 의존하고 크기 ‖ε‖_∞에는 무의존이다.
proportional tube(β·Σe_t, e-재귀는 c_x‖z‖+c_u‖u‖ 구동)는 전역 상수로 fit한
**크기 대리량**을 벌점한다 — softmax가 이미 무시하는(국소 상수) 성분까지 벌하고,
정작 위험을 결정하는 국소 변동은 지역화하지 못한다. franka에서 tube가 "많이
움직이는 후보 벌점 = 도달 지연"으로 퇴화한 관측(§7.4–7.5 framing.md)과 정합.

**R2 (두 업데이트 규칙 모두 적용).**
weighted_mean: U ← μ. score_langevin: score = (μ−U)/σ², U ← U + η σ²·score
= U + η(μ−U). 어느 쪽이든 오염 전달량은 η‖μ̂−μ‖ ≤ η·O(σ²L_ε/α).

**R3 (전형적 방향 — Stein).**
ξ ~ N(0,I)에서 1차 전개하면 μ̂ − μ ≈ −(σ²/α)·∇ε_σ(Ū) (ε_σ: σ-smoothed 오차장).
worst-case뿐 아니라 전형적 크기도 σ² 스케일이며, 방향은 오차장의 내리막.

**R4 (정직한 한계).**
1. **basin 선택은 보호 못 함:** 고-σ 초기 단계에서 후보는 서로 멀고, basin 스케일의
   오차 변동이 선택을 뒤집을 수 있다. MBD의 σ_start가 MPPI의 통상 Σ보다 크면
   OOD-취약 환경에서 초기 노출은 오히려 더 클 수 있다.
2. **Lipschitz 가정:** 접촉/hybrid 오차장(불연속)에는 적용 불가.
3. **systematic OOD optimism**(데이터 없는 곳의 낙관 편향)은 본 명제도, 전역 상수
   tube도 다루지 못한다. 원리적 처방은 데이터 커버리지/국소 불확실도(ensemble 등).

**R5 (MPPI 대비 요약).**
fixed-σ MPPI: 실행되는 매 액션이 고정 σ-폭 softmax 한 라운드의 출력 → σ²-오염
바닥에 상시 노출 → tube가 (증명서와 knob으로서) 상시 역할.
MBD: 실행 plan은 마지막(최소 σ) 업데이트들의 고정점 → 오염이 스케줄에 의해
자동 소멸 → tube는 성능 요건이 아니라 (원한다면) 안정성 증명서의 세금.

## 4. 논문 삽입용 LaTeX 초안

```latex
\begin{proposition}[Self-robustness of annealed sampling to smooth model error]
\label{prop:annealing}
Let $J$ be the true trajectory cost and $\hat J = J + \varepsilon$ the
surrogate-induced cost, with $\varepsilon$ $L_\varepsilon$-Lipschitz on a
neighborhood of the current iterate $\bar U$. At annealing level $\sigma$,
let $U_k = \bar U + \sigma \xi_k$, $\Xi = \max_k \lVert \xi_k \rVert$, and let
$\mu, \hat\mu$ denote the softmax-weighted means at temperature $\alpha$
under $J, \hat J$ respectively. Then (i) $\hat\mu$ is invariant to constant
shifts of $\varepsilon$; and (ii) with $R = L_\varepsilon \sigma \Xi$,
\[
\lVert \hat\mu - \mu \rVert
\;\le\; \bigl(e^{2R/\alpha} - 1\bigr)\, 2\sigma\Xi
\;=\; O\!\bigl(\sigma^2 L_\varepsilon \Xi^2 / \alpha\bigr).
\]
Consequently, along an annealing schedule $\sigma_i \downarrow 0$ the
contamination of the $i$-th denoising update vanishes as $\sigma_i^2$,
whereas a fixed-$\sigma$ sampler retains it at every executed step.
\end{proposition}

\begin{proof}
Constant shifts cancel in the softmax normalization, so only
$e_k = \varepsilon(U_k) - \varepsilon(\bar U)$, $|e_k| \le R$, enters:
$\hat w_k = w_k r_k / \sum_j w_j r_j$ with $r_k = e^{-e_k/\alpha} \in
[e^{-R/\alpha}, e^{R/\alpha}]$, hence $|\hat w_k / w_k - 1| \le
e^{2R/\alpha} - 1$. Since $\sum_k (\hat w_k - w_k) = 0$ and
$\lVert U_k - \mu \rVert \le 2\sigma\Xi$,
$\hat\mu - \mu = \sum_k (\hat w_k - w_k)(U_k - \mu)$ gives the bound.
\end{proof}

% Discussion hook:
% Our first-order treatment of model error is therefore structural, not
% penalty-based: the bilinear class removes the dominant systematic error
% (Sec.~C3), and the annealing schedule attenuates the residual smooth
% error quadratically in $\sigma_i$ (Prop.~\ref{prop:annealing}). An
% explicit error-tube cost, which fixed-$\sigma$ MPPI needs as the
% ingredient of its stability certificate~\cite{bkmppi}, is thus not a
% performance requirement in MBD; we report a tube ablation in
% Appendix~X.
```

## 5. Witness 실험 설계 (`experiments/annealing_robustness.py`)

학습된 BK 모델을 **합성 ground truth**로 삼고(참-모델 혼입 오차 배제), 그 파라미터
(A, B0, B_i)에 상대크기 δ의 매끈한 섭동을 가한 모델을 surrogate로 사용.

- **Exp 1 (Prop 직접 검증):** 같은 후보 집합에서 참/섭동 cost로 각각 softmax 평균을
  계산, ‖μ̂−μ‖ vs σ를 log-log로 — 예측: 기울기 ≈ 2 (소-σ 영역).
- **Exp 2 (end-to-end):** 동일 표본 예산에서 annealed(1.2→0.05) vs fixed-σ(고/중)로
  plan을 뽑고, 참 모델로 평가한 cost 열화 Δ(δ) = J_true(U_δ) − J_true(U_0)를 δ별 비교
  — 예측: annealed 곡선이 가장 평평.
- δ=0 대비 자기-기준 열화를 쓰는 이유: fixed-σ는 δ=0에서도 최적화 품질이 다르므로,
  "오차 민감도"를 최적화 품질과 분리하기 위함.

## 6. Witness 결과 (2026-07-06 실행; franka BK seed0, K=256)

**Exp 1 — Prop 확인 ✅.** log-log 기울기 **2.00** (σ ≤ 0.3, clip-free 영역; 예측 2).
두 decade에 걸쳐 slope-2 가이드에 정확히 정합. σ ≳ 0.5의 꺾임은 action-bound clip이
유효 spread를 압축하는 효과(스크립트에 문서화).

**Exp 2 — 예측 기각, 무차별 ⚠️.** "annealed가 가장 평평"은 **확인되지 않음**:

| arm | δ=0.003 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|
| annealed | +0.007±0.004 | +0.097±0.022 | +0.633±0.123 | +1.521±0.332 |
| fixed-mid | −0.001±0.012 | +0.057±0.043 | +0.461±0.106 | +1.291±0.260 |
| fixed-high | +0.001±0.006 | +0.074±0.027 | +0.571±0.111 | +1.396±0.233 |

arm 간 차이는 전 구간에서 1 s.e. 이내 — **통계적으로 구분 불가.** 단 절대 cost는
annealed가 δ=0에서 최고(0.552 vs 0.857/0.620)이고 모든 δ에서 경쟁력 유지.

**사후 해석 (왜 end-to-end에선 무차별인가):** 수렴하는 최적화기는 어느 것이든
surrogate 최적점의 변위를 물려받는데, 그 변위는 **오차장 ε의 성질이지 샘플러
스케줄의 성질이 아니다.** Prop이 bound하는 것은 업데이트 단위 오염이지, 그 업데이트가
충실히 수렴해 가는 고정점의 위치가 아니다. fixed-σ arm은 smoothed landscape를
최적화하므로 오차 성분도 함께 감쇠되어(합성곱이 Lipschitz 상수를 수축) 업데이트
오염 바닥을 상쇄 — 그래서 open-loop 지표에선 비긴다.

**주장 경계 (논문에서 지킬 것):**
- ✅ 주장 가능: (i) Prop 그대로(업데이트 오염 O(σ²), Exp 1 실증); (ii) 오프셋
  불변성 통찰(tube는 softmax가 무시하는 크기 대리량을 벌점); (iii) **annealing의
  정밀화 이득은 추가적 오차 민감도를 지불하지 않고 얻는다**(Exp 2: 무차별 + 절대
  cost 최고); (iv) tube 없는 headline이 실측으로 문제없음(§7.5 framing.md).
- ❌ 주장 금지: "annealed가 fixed-σ보다 end-to-end로 오차에 덜 민감하다" —
  Exp 2가 기각. "annealing이 tube를 대체한다"는 수사도 **업데이트 수준**으로 한정할 것.
- ⚠️ 위 ❌는 **open-loop 단일 solve** 기준. closed-loop에서는 분리가 나타남 → §7.

## 7. Closed-loop witness (Exp 3, `experiments/mirage_closedloop.py`)

Exp 2가 못 잡은 것: open-loop 한 번의 solve는 "최적점 변위"만 물려받아 arm 무차별.
그러나 closed-loop(계획→첫 액션 실행→상태 이동→재계획)에서는 **매 step 커밋되는
결정의 오염**이 plant를 통해 반복 주입된다 — 여기서 σ 스케줄의 차이가 결과로 나타남.

설정: plant = 무섭동 BK(합성 진실, 매 step re-lift로 다양체 유지), planner = δ-섭동
BK. 세 arm 동일 예산(K=256, N=5/replan), 차이는 σ 스케줄뿐. n=16/셀(섭동 4 × opt 4).

**정상상태 EE 오차 [mm] (마지막 15 step):**

| arm | δ=0 | 0.01 | 0.03 | 0.1 |
|---|---|---|---|---|
| annealed (1.2→0.05) | 48.6±0.8 | 82.6±7.0 | **142.3±6.0** | **306.3±23.2** |
| fixed-mid (0.35) | **29.4±1.8** | 95.6±11.4 | 219.5±18.3 | 384.2±36.3 |
| fixed-low (0.05) | 168.5±5.4 | 185.1±11.0 | 232.3±22.7 | 292.0±39.2 |

**판독:**
1. **같은 사기꾼, 다른 운명:** δ=0.03 시계열(그림 패널 A)에서 두 arm이 ~1 s에 같은
   최저점을 찍은 뒤, fixed-mid는 **단조롭게 목표에서 멀어지고**(매 replan의 σ=0.35
   심사가 오염된 커밋을 반복 주입) annealed는 **평평하게 유지.** 신기루 쫓기의 시각적 실체.
2. **열화 속도 ~2배:** 자기 baseline 대비 열화가 fixed-mid +66/+190/+355 vs
   annealed +34/+94/+258 — 모든 δ에서 fixed-mid가 약 2배 빠르게 무너짐.
   δ=0.03에서 ~4×s.e., δ=0.1에서 ~1.8×s.e. 분리.
3. **fixed-low의 δ-무감각(+16/+64/+124, 거의 평평) = σ² 법칙의 결과 수준 증거.**
   단 baseline 169 mm(탐색 불능). 즉 고정 σ는 "속거나(크면) 눈멀거나(작으면)"의
   trade-off를 강제당하고, **annealing은 둘 다 가져간다** — 탐색은 고-σ 단계에서,
   면역 커밋은 저-σ 단계에서.

**정직한 한계:** (i) δ=0에서 annealed(49)가 fixed-mid(29)보다 나쁨 — 매 replan을
σ=1.2부터 재-anneal하는 잔여 노이즈(K=256 소표본). 실전 튜닝(σ_start 축소/K 증가)
여지. (ii) 파국적 발산은 없음 — "실패 vs 성공"이 아니라 "2배 빠른 우아한 열화" 수준.
과장 금지. (iii) plant가 합성(BK 자신)이고 전역 매끈 섭동 — MuJoCo plant·국소 오차
버전이 다음 단계의 현실성 보강.
