# 논문 프레이밍: 기반(Established) vs 주장(Contribution)

> **핵심 제목(가안):** 관측 기반 bilinear Koopman surrogate를 통한 실시간 Model-Based Diffusion Trajectory Optimization
>
> **한 줄 요지:** MPC는 convexity 때문에 bilinear를 국소선형으로 *격하*시켜 피드백으로 버티지만, MBD는 그 격하를 **할 이유도 없고(convexity 불요) 하면 안 된다(open-loop).** 이 지점이 논문의 심장이다.

---

## 1. 기반 (Established — 인용해서 딛고 서는 것)

증명할 필요가 없는 확립된 사실. 증명하려 들면 오히려 novelty가 약해 보인다. "확립된 사실"로 담담하게 인용한다.

| 코드 | 내용 | 출처 | 역할 |
|------|------|------|------|
| **B1** | Model-based diffusion(MBD) 프레임워크 자체. sampling 기반 trajectory optimization, MPPI와의 연결, rollout으로 score 추정. | Pan et al. (MBD) | 이 구조를 **그대로 채택**. denoising도 동일 수행. |
| **B2** | control-affine 시스템에서 bilinear Koopman이 linear보다 우월(존재성 정리 + soft manipulator 실증). | Bruder et al. (RA-L 2021), Li et al. 등 | **"linear로는 부족, bilinear가 낫다"는 내 주장이 아니라 이들의 결과.** |
| **B3** | soft robot을 Koopman으로 관측 기반 모델링/제어 가능. 명시적 dynamics 없이 lifting으로 제어. | Bruder 원조 soft robot Koopman 계열 | soft robot을 대상으로 삼는 정당성의 토대. |
| **B4** | bilinear를 실시간 최적화에 쓰는 표준 트릭 = **z[0] 고정 국소선형화**. convexity 유지 위해 bilinear를 국소 linear로 격하하고 피드백으로 버팀. | Bruder 식 45–46, Li 식 12 | 기반이자 **내가 넘어서는 대상** (C3와 짝). |

---

## 2. 주장 (My Contributions — 기반 위에 세우는 것)

각 주장이 어느 기반에 얹혀 있는지 `[ ]`로 표시.

### C1. MBD의 rollout 병목을 학습된 bilinear surrogate로 대체해 실시간화 — `[B1+B2+B3 위에]`
참 dynamics rollout 자리에, 관측 기반으로 학습한 bilinear Koopman surrogate(**deep Koopman bilinearization**)를 꽂아 20 ms/step 확보.
**novelty는 "bilinear가 낫다"(=B2)가 아니라, 그 bilinear surrogate를 MBD의 rollout 엔진으로 통합한 결합에 있다.**

### C2. 이 결합이 특히 soft robot에서 값지다 — `[B3 위에, DIAL-MPC와 대비]`
DIAL-MPC는 GPU 병렬 full-order 시뮬로 실시간을 달성 → 단, **빠른 병렬 시뮬레이터가 있을 때**의 이야기.
본 연구는 **병렬 시뮬 없이 학습된 surrogate로** 같은 목표 달성. soft robot이 이 각도가 필요한 대표 영역임을 주장.

### C3. ★핵심 주장★ MBD에서는 bilinear를 "격하 없이 full로" 써야 하고, 동시에 "공짜에 가깝게" 쓸 수 있다 — `[B4를 정면으로 넘어섬]`
두 개의 독립된 이유가 맞물린다:

- **꼭 필요하다 (정확도):** MBD는 open-loop rollout이라 중간 피드백으로 z[0]를 리프레시할 수 없음 → B4의 국소선형화 트릭이 성립 안 함 → 모델오차가 horizon 따라 누적 → **full bilinear 전파 필수.**
- **써도 된다 (비용):** MBD는 QP를 풀지 않고 forward rollout으로 비용만 평가 → convexity 불요 → bilinear H항(몇 번의 추가 행렬곱)을 그대로 굴려도 linear와 비용 거의 동일.

> **한 줄 요약:** MPC는 convexity 때문에 bilinear를 국소선형으로 격하시키고 피드백으로 버티지만,
> **MBD는 격하할 이유도 없고(convexity 불요) 격하하면 안 된다(open-loop).**

### C4. 정직한 trade-off 명시: training-free를 포기하는 대신 실시간을 산다 — `[B1의 성질을 의도적으로 교환]`
원조 MBD의 training-free 장점을 학습된 surrogate로 대체 → "일회성 오프라인 학습 비용 ↔ 온라인 실시간"의 교환임을 스스로 밝힘. (리뷰어의 "MBD 장점 버렸다" 선제 차단하는 방어적 기여.)

---

## 3. 한 장으로 보는 구조

```
[B1 MBD 구조] ──────────────┐
[B2 bilinear>linear 이론]───┤
[B3 soft robot Koopman]─────┼──> C1 학습된 bilinear surrogate를 MBD rollout에 통합 (실시간화)
                            │
[B3] ───────────────────────┼──> C2 병렬 시뮬 없이 실시간 (DIAL-MPC 대비, soft robot 정당화)
                            │
[B4 z[0]고정 국소선형화]────┴──> C3 ★MBD는 full bilinear를 필수적으로(open-loop)
                                     & 저렴하게(convexity 불요) 쓴다  ← 논문의 심장
                                 └─> C4 training-free 상실을 오프라인 비용으로 정당화
```

---

## 4. 초록/서론용 한 문단

> Model-based diffusion(MBD)은 참 dynamics의 rollout으로 score를 추정하는데 **[B1]**, 이 비용이 빠른 미분가능 시뮬레이터가 없는 soft robot 같은 영역에서 실시간 계획을 가로막는다. 우리는 이 rollout을 관측 기반으로 학습한 bilinear Koopman surrogate로 대체한다 **[C1]**. control-affine 시스템에서 bilinear가 linear보다 필요하다는 것은 알려져 있으나 **[B2]**, 기존 MPC들은 convexity를 위해 bilinear를 초기 lifting state에 고정해 국소선형으로 격하시켜 왔다 **[B4]**. 우리의 관찰은, **MBD가 (i) open-loop rollout이라 그 격하가 오차 누적을 일으켜 성립하지 않으며, (ii) QP를 풀지 않아 convexity가 불필요하므로 full bilinear를 linear와 사실상 같은 비용으로 전파할 수 있다**는 것이다 **[C3]**. 이로써 병렬 시뮬 없이 **[C2]** 20 ms/step 실시간을 달성하며, 이는 일회성 오프라인 학습을 대가로 한다 **[C4]**.

---

## 5. 리뷰어 예상 질문 → 내장된 답

| 예상 질문 | 답 |
|-----------|-----|
| "bilinear 새로운 거 아니잖아?" | B2는 **기반**. 내 주장은 C3(MBD 안에서의 full bilinear 필요성·저비용성). |
| "DIAL-MPC가 이미 실시간인데?" | C2. 병렬 시뮬 없이 학습된 surrogate로 달성하는 다른 각도. |
| "왜 MPC처럼 국소선형화 안 하고?" | C3의 open-loop 논거. z[0] 고정이 rollout 오차 누적을 부름. |
| "MBD의 training-free 장점을 버린 것 아닌가?" | C4. 의도된 trade-off(오프라인 1회 비용 ↔ 온라인 실시간)로 명시. |

---

## 6. 다음 단계 (실험 설계로 연결)

C3를 실제로 증명할 실험:
1. **예측오차 vs horizon 곡선** — 국소선형(z[0] 고정)이 발산하고 full bilinear가 참 dynamics를 따라가는 그림.
2. **downstream 계획 성공률** — linear-MBD 실패 vs bilinear-MBD 성공.
3. **조건 sweep** — 속도 · 곡률 · horizon 길이를 변화시켜, bilinear 이득이 이들에 비례해 커짐을 입증.

---

## 7. 현황 업데이트 (2026-07-06)

### 7.1 선행연구 재확인 결과

| 논문 | 확인 내용 | C3 위협도 |
|------|-----------|-----------|
| **dk-mppi** (MPPI-DK, Hao et al., arXiv:2603.05385) | error tube **없음**(단어 출현 0회). linear DKO로 MPPI 가속만. 기여 3개 전부 연산 가속. | 낮음 — linear·receding-horizon·tube 없음, 오히려 공백 |
| **bk-mppi** ("Bilinear by Default", arXiv 익명) | ★위협★ "sampling은 convexity 불요 → bilinear가 옳은 기본값" = **C3의 '써도 된다' 축을 선점.** Prop 1(상수 입력행렬은 config-dependent gain 표현 불가)로 '필요하다' 축도 표현력 관점에서 선점. error tube가 기여 3번으로 명시(식 6·9, Prop 2, Thm 1). | **높음** — 반드시 대비 인용 + C3 재포지셔닝 필요 |

bk-mppi가 스스로 인정한 약점(= 우리에게 열린 틈):
- MPPI이지 **MBD 아님** ("diffusion" 언급 0회) → open-loop 누적 논거·annealing 결합은 무주공산.
- **soft robot 아님** ("soft" 언급 0회) → B3 각도 그대로 유효.
- tube가 반쪽: certified 상수 c_x, c_u를 deep dictionary가 못 줘서 **LS proxy**, ‖A‖≈1 적분기 lift라 40-step에서 **~10³× conservative**(구조적), 역할을 conservatism knob으로 자체 격하, hardware는 placeholder.


### 7.3 error tube 거취 (결정)

- **독립 기여로 내세우지 않는다.** bk-mppi 기여 3번이 선점. 우리 구현도 사실상 동일물.
- **method 레벨에서 정직하게 명시만 한다** (썼으면 썼다고 쓴다): 기여 격상 여부는 나중에 판단 — method 문단 마지막의 "we do not claim this as a contribution" 한 문장만 빼면 격상 구조로 전환 가능하게 작성해 둠 (문안 초안 있음).
- **구현 차이 1개도 함께 공개**: 배치 전파는 매 step 정확한 ‖M(u)‖₂ 대신 상계 `m̄(u) = ‖A‖₂ + Σ_i|u_i|·‖B_i‖₂`를 사용(더 conservative, 배치 비용 절감). `bk_mbd/tube.py` docstring에도 bk-mppi 계열 코드(`mppi_koopman/verify/tube_deep.py`)를 따랐다고 명시돼 있음.

### 7.4 cost-sensitivity tube 파일럿 (부정적 결과, 폐기)

score 신뢰도 관점에서 tube를 cost 민감도로 가중하는 변형(`β·Σ L_t·e_t`, `L_t = ‖∂c/∂b(b̂_t)‖`)을 `--tube-mode cost-sens`로 구현·시험함 (`bk_mbd/tube.py::cost_sensitivity_torch`, `experiments/realtime_franka.py`).

- 오버헤드 +1.8 ms/plan(~9%)으로 실시간엔 문제없으나, franka reaching에서 **plain보다 도달이 느림**(β=1.0 완주 실패, β=0.01·0.1 도달하되 ~5.7 s vs plain ~1.2 s).
- kill-switch 조건("plain 대비 유의미한 이득 없으면 폐기")에 걸려 **폐기.** 논문 결과에 사용하지 않으므로 기술 의무 없음(원하면 각주 1줄). 코드는 기본값 `plain`으로 비활성 상태로 남김(부정적 파일럿 기록).

### 7.5 no-tube 실측 + annealing 자기강건성 관찰 (tube 불요의 이론적 근거)

**실측 (franka lockstep, target 0, seed 0):** `--tube-mode none`(fitting·전파 완전 생략) 모드 추가 후 3-모드 비교 —

| tube-mode | plan latency | strict reach |
|---|---|---|
| none | 16.7 ms | **0.80 s** |
| plain (β=2e-4) | 19.0 ms | ~1.2 s |
| cost-sens (β=0.01) | 20.7 ms | 5.7 s |

tube 없는 쪽이 가장 빠르고 가장 잘 도달. bk-mppi 자신의 ablation("β>0은 nominal 성능을 내주는 conservatism knob")과 일치.

**이론적 관찰 (→ `plan/annealing_robustness.md`에 Prop으로 정식화):**
tube의 정체는 optimizer's curse(모델이 낙관적으로 틀린 후보가 뽑히는 선택 편향)에 대한 pessimism 보험인데, **MBD의 annealing 구조가 이 보험의 일을 대신한다:**

1. **softmax 상수-불변성:** 모델오차 ε(U)는 후보 간 **변동**으로만 선택을 왜곡. 공통 오프셋은 정확히 소거.
2. **σ 자기감쇠:** 후보는 σ-공 안에 있으므로 매끈한 ε의 후보 간 변동 ~ σ·Lip(ε) → 한 denoising 업데이트의 오염 = **O(σ²·Lip(ε)/α)**. MBD는 σ→0으로 anneal하니 후반(정밀화) 단계는 매끈한 모델오차에 구조적으로 면역. **MPPI는 σ 고정 → 오염 바닥이 매 step 유지 → tube가 상시 역할을 가짐.**
3. **증명서 구조:** bk-mppi의 tube는 Thm 1(closed-loop 안정성)의 재료. MBD 서사(최적화 점근 정확성)에는 tube를 요구하는 정리가 없음.

**정직한 한계:** (i) 매끈한/거친 오차에만 성립 — OOD systematic optimism엔 둘 다 무방비(tube도 전역 상수라 못 막음). (ii) MBD의 고-σ 초기 단계는 오히려 데이터에서 멀리 후보를 뿌려 OOD-취약 환경에선 더 노출될 수 있음 — tube가 MBD에서 값할 이론적 자리는 이 초기 단계뿐. (iii) 문헌의 정리가 아니라 자작 관찰 → 논문엔 Remark/Prop으로 자체 정식화 필요.

**7.3 결정에 미치는 영향:** "headline 파이프라인에서 tube 제거(기본 none) + appendix ablation"이 유력해짐 — 이러면 bk-mppi 기여 3번과의 겹침·기술 의무가 동시에 소멸하고, "tube 없이도 잘 됨"은 C1(surrogate 품질)의 방증이 됨. C3와의 연결: *모델오차에 대한 우리의 1차 처방은 벌점(tube)이 아니라 모델 클래스를 옳게(full bilinear); 잔여 매끈한 오차는 annealing이 흡수.* 최종 확정은 witness 실험(`experiments/annealing_robustness.py`) 결과 확인 후.

**Witness 실험 결과 (실행 완료 → `plan/annealing_robustness.md` §6–7, `out/annealing_robustness/`, `out/mirage_closedloop/`):**
- **Exp 1 ✅:** 업데이트 오염 ‖μ̂−μ‖의 log-log 기울기 **2.00** — Prop의 O(σ²) 예측 정확히 실증. 커밋 시점 오염 비대칭도 곡선에서 직접 읽힘: MBD 마지막 심사(σ=0.05) 대비 fixed-σ 심사(0.31/1.2)가 **38×/287×** 더 오염.
- **Exp 2 ⚠️:** **open-loop** end-to-end 열화는 annealed vs fixed-σ **무차별** — 수렴하는 최적화기는 어느 것이든 surrogate 최적점의 변위(ε의 성질)를 물려받기 때문. 단 절대 cost는 annealed가 전 δ에서 최고~동률.
- **Exp 3 ✅ (closed-loop, `mirage_closedloop.py`, n=16/셀):** 같은 섭동에서 fixed-σ(MPPI 아날로그)는 **~2배 빠르게 열화**(+66/+190/+355 vs annealed +34/+94/+258 mm), δ>0 전 구간 annealed 우위(δ=0.03에서 ~4×s.e. 분리). 시계열: 두 arm이 같은 최저점 도달 후 fixed-mid만 단조 이탈(신기루 쫓기), annealed는 유지. fixed-low(σ=0.05 고정)는 δ-무감각이지만 탐색 불능(169 mm baseline) — **고정 σ는 "속거나 눈멀거나", annealing은 둘 다 가져감.** 정직한 한계: δ=0에선 fixed-mid가 근소 우위(재-anneal 노이즈), 파국적 발산은 아님(과장 금지).
- **주장 경계 확정:** "annealing이 tube를 대체" = 업데이트 수준(Prop+Exp1) + **closed-loop 결과 수준(Exp 3)**까지 주장 가능. open-loop 단일 solve의 강건성 우위(Exp 2)는 주장 금지. tube-free headline 근거: (i) 실측(none 최고), (ii) 오프셋-불변성 통찰, (iii) bk-mppi 자신의 knob 규정, (iv) Exp 1·3의 메커니즘+결과 증거.