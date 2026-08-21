# 논문 요약본 (섹션별 흐름) — 리뷰어 시점 digest

> **한 줄 takeaway:** Model-Based Diffusion(MBD)은 강력하지만 오프라인 경로
> 최적화기에 머물렀다. 우리는 이를 **bilinear Koopman 대리모델**로 실시간 제어기로
> 만든다 — DIAL-MPC가 GPU로 달성한 diffusion-MPC 실시간을 **GPU 없이** 낸다. 그리고
> 대리모델 오차 하에서 **어닐링**이 단일-스테이지 샘플링·convex-MPC 대비 왜 유리한지를
> 실험으로 보인다.

*Title:* Koopman-Accelerated Model-Based Diffusion for Real-Time Robot Control

---

## Abstract
- MBD는 제어 스텝당 수천 롤아웃 때문에 오프라인 궤적 최적화에 국한된다(`pan2024model`).
  DIAL-MPC(`xue2025full`)는 이를 실시간화했으나 **GPU 병렬 full-order 시뮬레이션**에 의존.
- 우리는 시뮬레이터를 **학습된 bilinear deep Koopman 대리모델**로 대체해, MBD의 어닐링
  스케줄을 **GPU 없이 제어율(≥20 Hz)에서** 구동한다.
- 핵심 관찰: 대리모델 오차 하에서 노이즈 스케줄은 **탐색 반경**과 **정밀도 하한**이라는
  두 레버를 제어하며, 어닐링만이 둘을 동시에 준다 — 단일-스테이지 샘플링에는 없다.
- FR3(MuJoCo), 드론, 실기에서 검증: 오라클과 동등한 정확도, 비볼록 과제에서 convex
  Koopman-MPC 대비 일방적 우위, 시뮬레이터 없이 온로봇 재식별만으로 실시간 표적 추종.

## 1. Introduction — 문제 → 선행 → 기여
- **필요성:** 샘플링 MPC는 저렴한 구조화 롤아웃이 필요. MBD는 annealed coarse-to-fine의
  강한 최적화기지만 롤아웃 비용 때문에 **오프라인**.
- **선행(3자 구도):**
  - DIAL-MPC(`xue2025full`): diffusion-MPC 실시간, 단 **GPU + full-order 시뮬**.
  - Koopman 대리모델은 CPU에서 롤아웃을 저렴하게; 상수-입력 linear lift는 형상 의존
    게인에 부적합 → **bilinear**(`bruder2021bilinear`,`iacob2024`). 학습 Koopman으로
    샘플링 가속은 linear+MPPI(`mppidk2026`), bilinear+MPPI(CPU 실시간)도 존재.
  - convex Koopman-MPC(`korda2018mpc`,`bruder2021bilinear`)는 실시간이나 비볼록 실패.
  - **공백:** MBD의 *어닐링*을 **GPU 없이** 제어율에서 돌린 방법 없음; 대리모델 오차 하
    어닐링 효용 미연구; 실시간 Koopman 제어기에서 샘플링 vs convex-MPC 비교 부재.
- **기여:** (i) **GPU 없는 실시간 MBD** — 같은 과제에서 병렬-시뮬 rate를 CPU로;
  (ii) **대리모델 오차 하 어닐링** — reach/precision 두 레버 + 실패 서명; (iii) **샘플링
  > convex Koopman-MPC**(비볼록); (iv) **실기 배치** — 시뮬 없이 온로봇 데이터 재식별로
  실시간 표적 추종.

## 2. Preliminaries
- **2A 문제:** 실시간 receding-horizon 제어. 참-동역학 롤아웃은 **rate에 비해 너무 비싼
  오라클**(sim, 배치 불가)이거나 배치에선 시뮬 없이 데이터로 재식별(HW) — 동기는
  "미지 동역학"이 아니라 **"rate로 못 돌림"**(sim의 model-free 필요성은 주장하지 않음).
- **2B MBD 리뷰**(`pan2024model`): Boltzmann target, Tweedie score = 비용가중평균,
  어닐링 ladder, 업데이트. — 우리가 가속하는 대상.
- **2C Koopman lifting(배경):** deep lift + linear/bilinear 예측기 *형식*
  `z⁺=Az+B₀u+ΣuᵢBᵢz`와 gain-variation 직관을 표준 배경으로
  (`iacob2024`,`bruder2021bilinear`,`korda2018mpc`,`lusch2018deep`). — MBD와 대칭인 리뷰.

## 3. Method
- **3A Bilinear rollout:** lift/rollout 형식 제시(문헌 공통, canonical 인용). "왜 bilinear":
  형상 의존 Jacobian → 상수 B로 불가 → state-의존 입력 채널(bilinear)이 최소 구조적 처방.
- **3B Multi-step training:** H-step rollout loss + latent-consistency(deep-Koopman 표준),
  zero-init bilinear로 linear에서 출발해 데이터가 요구하는 곳만 bilinear.
- **3C Error tube (차용):** 비례 오차 경계 → 후보별 온라인 tube 재귀. `kim2026bilinear`
  (+`safedmd2024`/`strasser2026`) 인용. 성능 기여가 아니라 "검증된 비관성"의 랭킹 페널티로
  한정(abstract·기여에 없음).
- **3D 실시간 MBD 루프:** 1회 리프트 → 리프트 공간에서 후보 전파 + tube 누적 → 가중 →
  업데이트 → 첫 입력 실행 → warm-start 후 재계획. — MBD를 제어기로 만드는 통합. MBD의
  어닐링 ladder는 이 루프의 스테이지로 그대로 포함(별도 정당화·정리 없음).

## 4. Experiments — 2 표 + 3 소절
**공통:** 학습 **seed 5개**(E2·E3의 통계 주장; 학습 분산). **E4는 예외** — single-model·
vary-target(최적화기 격리, 타깃 N개). 성공률 paired McNemar, 정착시간 paired Wilcoxon.
플랫폼 **기본 MuJoCo FR3**(kinematic은 물리-격리·정밀도 측정에만).

- **4.1 E1 — 실시간(헤드라인 표):** {**병렬-시뮬 MBD(GPU/MJX)** ← 우리 과제에 우리가 구현
  (cf. DIAL-MPC `xue2025full`), **MBD-true 오라클(CPU)**, **BK-MBD(ours, CPU)**} ×
  {제어율, latency median/p95/worst, deadline-miss, 성공률}, ours N-sweep. → *"GPU 없이
  병렬-시뮬 rate 달성"*. (DIAL-MPC(CPU)=MBD-true라 중복 제거.) 옵티마이저 latency는
  optimizer/plant-inclusive 분리 보고(중앙 17 ms·최악 40 ms·50 ms 0-miss 계열). *[병렬-시뮬 측정 예정]*
- **4.2 E2 — 롤아웃 클래스·충실도·용량(메인 표):** {오라클, BK-MBD, DK linear, DK-split,
  MLP, linear-large} × {reach, strict, final err, held-out 예측오차, ms/step}. 한 표가
  네 결론 흡수 — 오라클 동등(BK 35/35) · bilinear 필요(linear 실패, DK-split 회복) · 용량
  아님(large-linear 실패) · **MLP는 예측오차↓인데 reach↓(26/35)** → open→closed 증폭.
  criterion(`kim2026bilinear`)은 여기서만 인용. 보조: kinematic 1행(물리 제거해도 linear 실패).
- **4.3 E3 — 어닐링 under surrogate error(소절, 이 논문의 핵심 finding):** **2요인 설계**로
  "surrogate error 축"을 실제 변수로 — **행=오차 수준**(참-동역학 오라클 / BK 정상 / BK
  1/10-데이터 / BK 편향주입) × **열=스케줄**(1×wide·narrow, 5×wide·narrow·**reverse**·anneal).
  주장: **anneal−fixed 격차가 오차↑에 따라 단조 증가**(오라클에선 작고 열화될수록 벌어짐).
  **reverse-anneal(0.3→1.2)** = 순서 인과 통제(구조 vs 순서). 두 레버 프레이밍은 여기서
  **가벼운 remark**로. + dispersion 직접 측정(E3b, 기존) + 재계획빈도 정합. surrogate-error
  특유 신호 = narrow-fixed가 대리모델 **가짜 극소점**에 잠김(오라클엔 없음). kinematic 사용.
  *(오라클 행=E2 오라클 재사용, 1/10=기존, 신규는 편향주입 행 + reverse 열뿐.)*
- **4.4 E4 — 샘플링 vs convex Koopman-MPC(소절, 완료):** 드론 원형 window, 벽 뒤·창 아래
  35 타깃. **BK-MBD 35/35 vs QP-MPC 2/35, 집행 위반 0(양쪽 안전정지), paired McNemar
  p≈2e-10** — QP는 near-side 분기에 commit해 정지, 샘플링은 전역 가중으로 통로 homotopy
  재선택. (강화 QP 변형으로 straw-man 방어.)
- **4.5 E5 — 하드웨어(소절):** 시뮬레이터 無 + 온로봇 25분 재식별만으로 FR3 이동표적
  실시간 추종(속도별 오차, deadline-miss 0) + **모바일 유니사이클 ㄷ자**(실기 비볼록,
  실시간 이중계상) + resolved-rate 베이스라인. *[실기 예정]*

## 5. Conclusion
- **기여 재진술:** GPU 없는 실시간 MBD; 대리모델 오차 하 어닐링의 두 레버; 샘플링의 비볼록
  우위; 시뮬 없는 온로봇 재식별로 실기 배치.
- **한계:** tube는 인증 아닌 경험적 랭킹 페널티; 접촉 없는 과제; **sim 실험은 참-모델이
  알려진(다만 너무 비싼) 과제** — 진짜 미지-동역학(연질·케이블·마찰지배) 플랜트로의 확장은
  향후(모델-프리 필요성은 현재 HW 재식별로만 시연).

---

## 리뷰어가 받는 인상 (self-check)
- **신규성 위치가 명확**: 실시간(vs DIAL-MPC의 GPU) + 어닐링(vs bk_mppi의 MPPI) + 비볼록
  (vs convex-MPC). tube/multistep/criterion은 차용으로 인용 → "선행 재탕" 방어.
- **실험이 조밀**: 파편화된 표들이 E1·E2 두 표로 통합, 소절은 어닐링·비볼록·실기만.
- **주장↔증거 정합**: abstract의 모든 주장이 E1–E5 중 하나에 매핑, tube는 기여에서 빠짐.
- **정직성**: rate는 측정 후 결정(20 Hz 실시간 하한), latency 열 분리, 1 seed 한계 명시.

---

## 표 골격 (채울 값 — 값은 비움)

### E1 — 실시간
**E1a. 방법 비교** · MuJoCo FR3
| Method | Rollout | Compute | Rate [Hz] | Latency med [ms] | p95 [ms] | worst [ms] | deadline-miss (/K) | Reach |
|---|---|---|---|---|---|---|---|---|
| 병렬-시뮬 MBD (cf. DIAL-MPC) | full-order sim | GPU (MJX) |  |  |  |  |  |  |
| MBD-true (oracle) | full-order sim | CPU |  |  |  |  |  |  |
| BK-MBD (ours) | bilinear Koopman | CPU |  |  |  |  |  |  |

**E1b. Ours N-sweep** · CPU
| N (samples) | worst latency [ms] | max rate 0-miss [Hz] | Reach | Strict |
|---|---|---|---|---|
| 800 |  |  |  |  |
| 400 |  |  |  |  |
| 200 |  |  |  |  |

### E2 — 롤아웃 클래스·충실도·용량 · MuJoCo FR3, 1 seed, N 타깃
**E2a. 메인**
| Method | Rollout class | Reach (/N) | Strict (/N) | Final err [m] | Pred err @H [m] | ms/step |
|---|---|---|---|---|---|---|
| MBD-true (oracle) | true sim |  |  |  |  |  |
| BK-MBD (ours) | bilinear |  |  |  |  |  |
| DK | linear |  |  |  |  |  |
| DK-split | linear + analytic FK |  |  |  |  |  |
| MLP | nonlinear |  |  |  |  |  |
| Linear-large | linear (large lift) |  |  |  |  |  |

**E2b. 물리 격리(확인 1행)** · kinematic FR3
| Method | Reach (/N) | Strict (/N) |
|---|---|---|
| BK-MBD |  |  |
| DK (linear) |  |  |

### E3 — 어닐링 · kinematic FR3, 1 cm 스트레스
**E3a. 2요인: 오차 수준 × 스케줄** (셀 = reach@1cm /N; 주장: anneal−fixed 격차가 위→아래로 단조↑)
| Surrogate (오차↑) | 1×wide | 1×narrow | 5×wide | 5×narrow | 5×reverse | 5×anneal | anneal−best-fixed 격차 |
|---|---|---|---|---|---|---|---|
| 참-동역학 오라클 |  |  |  |  |  |  |  |
| BK 정상 |  |  |  |  |  |  |  |
| BK 1/10 데이터 |  |  |  |  |  |  |  |
| BK 편향주입 |  |  |  |  |  |  |  |

*보조 지표(정착시간·final err)는 승리 스케줄(anneal) 중심으로 별도 소표 or 본문.*

**E3b. Dispersion(정밀도 하한 검증)**
| 상태 | σ 설정 | 측정 dispersion | frozen-weight 예측 |
|---|---|---|---|
| settled | 1.2 |  |  |
| settled | 0.3 |  |  |
| cold-start | fixed-wide |  |  |
| cold-start | anneal |  |  |
| cold-start | narrow |  |  |

### E4 — 드론 non-convex · 완료(값: `drone_window_stats.py`)
**E4a. 플래너 비교** · 35 타깃
| Planner | reach@2.5cm (/35) | reach@1cm (/35) | viol | safe-stall (/35) | settling steps (med) | ms/step |
|---|---|---|---|---|---|---|
| BK-MBD |  |  |  |  |  |  |
| QP-MPC |  |  |  |  |  |  |
| QP-MPC (강화 변형) |  |  |  |  |  |  |

**E4b. Paired 대비**
| both | BK-only | QP-only | neither | McNemar p |
|---|---|---|---|---|
|  |  |  |  |  |

### E5 — 하드웨어
**E5a. FR3 이동표적 추종** · 20 에피소드
| target speed [cm/s] | ours TCP err mean±sd [cm] | p95 [cm] | tracking lag [ms] | deadline-miss (/K) | linear-rollout err [cm] |
|---|---|---|---|---|---|
| 4 |  |  |  |  |  |
| 8 |  |  |  |  |  |
| 12 |  |  |  |  |  |

요약: `__/20` 에피소드가 마지막 10 s 동안 5 cm threshold 유지.

**E5b. 모바일 유니사이클 ㄷ자**
| Planner | reach | viol | settling steps |
|---|---|---|---|
| BK-MBD |  |  |  |
| QP-MPC / resolved-rate |  |  |  |
