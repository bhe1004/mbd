# 논문 구성 계획 — Koopman-Accelerated MBD for Real-Time Control

## 배경 (왜 재구성하는가)
`paper/main.tex`는 **같은 저자들의 선행 논문** `kim2026bilinear`(bk_mppi,
"Bilinear by Default / Linear or Bilinear", Lee & Kim — bib 주석: *중립적 3인칭으로
인용, 절대 "our previous work" 금지*) 위에 세워져 있다. 그 선행 논문은 이미 다음을
포함한다: bilinear **필요성 기준**(Prop 1 / gain-variation), **error tube**(Prop 2,
동일 재귀식), **multi-step loss**(eq 14), **free-energy 안정성**(Prop 3), MPPI,
그리고 **드론 + 유니사이클** 실험.

따라서 현재 초안의 실제 리스크는 **신규성 귀속(attribution)**이다: *차용한* 구성요소
(error tube가 abstract + contribution #1에 있음)를 전면에 내세우고, 정작 bk_mppi가
갖지 못한 유일한 것 — **MPPI가 아니라 MBD(annealed diffusion)를, 그것도 실시간으로** —
을 과소 강조하고 있다. 모의 ICRA 리뷰가 독립적으로 지적한 약점:
- **W1** 헤드라인 속도향상이 약한 베이스라인 대비(GPU 없이 CPU MuJoCo)
- **W2** tube가 사실상 null result인데 abstract/기여에 과대 대표
- **W3** FK가 알려진 접촉없는 도달 과제뿐이라 "학습 대리모델이 필요하다"가 미시연
- **W4** 벽 과제의 QP-MPC 베이스라인이 straw-man
- **W5** compute-matched 비교 부재
- **W6** 이론(Prop 1/Remark 1)이 서사 대비 얇음
- **W7** latency 표 자기모순(64 ms vs 50 ms 데드라인)
- **W8** 신규성이 저자 자신의 프리프린트에 크게 의존(점진적)
- **W9** 성공률 통계 처리 느슨(McNemar 필요)

**이 계획의 목표:** 문제 → 사전연구 → 이론 → 제안방법 → 실험이 깔끔히 매칭되고,
신규성이 **실시간 MBD + 대리모델 오차 하 어닐링 + 샘플링 vs convex-MPC**에 집중되며,
bk_mppi와 공유하는 모든 요소는 인용 처리(주장 아님), 실험은 하나의 일관된
seed/플랫폼 프로토콜로 정리, 유효한 리뷰 지적은 각각 구체 실험/수정으로 매핑.

**확정된 사용자 결정:**
- W3 → **새 실험 안 함**(torque/soft/미지툴 모두 제외). 대신 **범위 축소**: sim은 참-모델이
  알려진(다만 rate에 너무 비싼) 과제로 한정, model-free 필요성은 **E5 HW 온로봇 재식별로만**
  시연, 미지-동역학 플랜트는 한계로 명시.
- 모바일 유니사이클 ㄷ자 장애물 → 하드웨어 실시간 캡스톤으로 포함(이중 계상).
- W1 → MJX/GPU 오라클 측정 추가.
- Section C → 드론 window가 secy FR3 전신-팔 벽을 **완전 대체**.

---

## 1. 포지셔닝(핵심 축): 차용 vs 신규
2축 포지셔닝(세 선행연구 누구도 차지하지 않은 교집합에 우리 논문이 위치):

| 구성요소 | 출처 | 역할 |
|---|---|---|
| MBD(Boltzmann target, 어닐링, Tweedie) | `pan2024model` | 우리가 **실시간화**하는 오프라인 최적화기 |
| **DIAL-MPC**(diffusion-annealing MPC, **full-order 시뮬 롤아웃·GPU 병렬**, legged에서 실시간) | `xue2025full` | **실시간 baseline**: diffusion-MPC를 *GPU로* 실시간화; 우리는 surrogate로 **GPU 없이** 동일 rate |
| bilinear rollout, error tube, multi-step loss, 필요성 기준 (+ Koopman surrogate로 CPU 실시간이지만 **MPPI** 단일 스테이지) | `kim2026bilinear` | **차용 — 인용, 절대 기여로 주장 X**; 이것 대비 차별 = **annealing** |
| convex Koopman-MPC(linear QP / frozen-z bilinear SQP) | `korda2018mpc`, `bruder2021bilinear` | Section C가 non-convex에서 이기는 실시간 convex 대안 |

교집합: **annealing(MBD/DIAL-MPC) + Koopman surrogate(bk_mppi) = GPU 없는 실시간
diffusion-MPC** — 셋 중 누구도 갖지 못한 조합.

**이 논문의 신규(이것 외엔 없음):** (i) **GPU 없는 실시간 MBD** — DIAL-MPC
(`xue2025full`)의 diffusion-MPC rate를 *같은 과제*에서 CPU로 달성(DIAL-MPC는
GPU 병렬 full-order 롤아웃 필요); (ii) 대리모델 오차 하 어닐링 — reach/precision 이중
레버 + 실패 서명 분석, 단일 스테이지 MPPI(`kim2026bilinear`)엔 없음; (iii) 샘플링 >
convex Koopman-MPC(non-convex); (iv) 실기 배치(시뮬 없는 온로봇 데이터 재식별).
**철칙:** tube / multistep / 기준은 "adopted from [kim2026bilinear]"로만,
contribution 목록·abstract에 절대 넣지 않음. 주의: "GPU-free"는 DIAL-MPC 대비 차별
(bk_mppi도 CPU); bk_mppi 대비 차별은 annealing.

## 2. Introduction 흐름 (필요성 → 선행 → 기여)
1. **필요성** — 샘플링 MPC는 저렴하고 구조화된 롤아웃이 필요; MBD는 강력한 어닐링
   최적화기지만 제어 스텝당 수천 롤아웃 때문에 **오프라인** 경로 최적화에 국한
   (`pan2024model`).
2. **선행** — (a) DIAL-MPC(`xue2025full`)가 이미 diffusion-MPC를 실시간화, 단
   **GPU 병렬 full-order 시뮬레이션**으로. (b) Koopman 리프트는 CPU에서 롤아웃을
   저렴하게; 상수 입력 선형 리프트는 형상 의존 게인에 부적합 ⇒ bilinear 필요(기준
   `kim2026bilinear`, 차용); bilinear+MPPI는 이미 CPU 실시간(`kim2026bilinear`),
   linear+MPPI(`mppidk2026`), bilinear+convex-MPC(`bruder2021bilinear`). (c) **공백:**
   MBD의 *어닐링* 스케줄을 **GPU 없이** 제어율에서 돌린 방법 없음; *대리모델* 오차 하
   어닐링 효용 미연구; 실시간 Koopman 제어기 안에서 샘플링이 convex MPC를 non-convex
   에서 이기는지 미시연.
3. **기여** — §1의 신규 4항, FR3(kinematic / MuJoCo / 하드웨어) + 드론 +
   모바일에서 검증.

## 3. Preliminaries — 이론은 하되 층위로 나눔
세 층위를 구분("Koopman도 MBD처럼 이론 설명 필요하지 않나" 해소):
- **확립된 배경 → 여기서 설명, 표준 인용:** 3A 문제; 3B MBD 리뷰
  (`pan2024model`); 선택적 **3C Koopman lifting** — linear/bilinear 예측기 *형식*
  `z⁺=Az+B₀u+ΣuᵢBᵢz` + gain-variation 직관을 교과서적 배경으로
  (`iacob2024`,`bruder2021bilinear`,`korda2018mpc`). → MBD와 대칭인 Koopman 리뷰 확보.
- **설계 결정 정당화 → Method(§4A):** *왜 bilinear를 고르는가*.
- **최근 한 논문의 특정 정리 → 인용, 재유도 X:** 필요성 **기준**
  (`kim2026bilinear`의 gain-variation proposition) — 간결한 귀속 요약만. 완전한
  Preliminaries 정리로 두지 않음(확립된 배경이 아니며, 앞쪽 전면 재유도는 W8
  자기중복 인상을 키움).

## 4. Method — 중복을 어떻게 서술할까
**원칙:** *공통 formulation*은 원전(canonical) 인용/표준 서술(kim2026bilinear 불필요),
*특정 결과*만 kim2026bilinear 인용. **개념이 공통이어도 문장·식·표기는 자기 표현으로
재작성**(텍스트 자기표절 방지).
- **4A Bilinear rollout — kim2026bilinear 인용 X:** 리프트 + linear/bilinear 형식
  (`z⁺=Az+B₀u+ΣuᵢBᵢz`, exact-decoder slicing)은 문헌 공통 → `bruder2021bilinear`,
  `iacob2024`, `korda2018mpc`, `lusch2018deep` 인용. "왜 bilinear"는 qualitative 자기
  서술(형상 의존 Jacobian → 상수 B 부적합) + `bruder2021bilinear`. 정량 criterion을
  실제 쓸 때만 kim2026bilinear(→ 실험으로 국소화, §5 C3/C4).
- **4B Multi-step training — kim2026bilinear 인용 X:** H-step rollout loss +
  latent-consistency는 deep-Koopman 학습 표준 관행 → 일반 서술(필요시 `lusch2018deep`).
  자기 표기로 재작성.
- **4C Error tube — 인용 유지:** 비례 경계 → 온라인 tube 재귀는 kim2026bilinear의 특정
  구성(certified bound 기반) → `kim2026bilinear` + `safedmd2024`/`strasser2026` 인용.
  **강등** — abstract·contribution에서 제거; 차용한 후보 랭킹 페널티("검증된 비관성")로
  유지. (W2, W8)
- **kim2026bilinear 의존 국소화:** Method 전면이 아니라 (i) tube(4C) + (ii) criterion을
  실제 검증·사용하는 실험(DK-split 통제조건·mechanism 분해, §5 C3/C4)에서만 인용.
- **4D 대리모델 오차 하 어닐링 — 유일한 신규 방법 절.** reach/precision floor 이중 레버를
  MBD 스케줄에 적용(대리모델 오차 하). 단일 스테이지 MPPI(= bk_mppi 동작점)엔 없는
  coarse-to-fine으로 명시. W6대로 `Prop reach`/`Remark floor`는 remark로 강등(또는 강화),
  정리로 과장 X.
- MBD 실시간 루프(1회 리프트 → 리프트 공간 전파 → 가중 → 업데이트)를 MBD를 제어기로
  만드는 통합으로.

## 5. 실험 — 통합 구조 (2 표 + 3 집중 소절)
**전 실험 공통 프로토콜(실험 intro에 한 번):**
- 학습 **seed 1개**(고정 모델). 통계력은 **타깃 N개(예 35)**의 many-target 시행에서 확보
  (bk_mppi / drone window와 동일한 single-model·vary-target 프로토콜). 성공률 비교는
  타깃 단위 **paired McNemar**, 정착시간은 paired Wilcoxon.
- 플랫폼 **기본 = MuJoCo FR3**. **kinematic FR3는 정말 필요한 곳만**: (i) E2의 물리-격리
  확인 1행, (ii) E3의 깨끗한 정밀도 측정. 드론 = 통제된 non-convex testbed(E4),
  모바일 = HW non-convex 캡스톤(E5).
- **묶을 수 있는 건 한 표로**(다중 baseline 동시 비교), 상세 설명·개별 비교가 필요한
  것만 별도 소절.

### E1 — 실시간 (헤드라인 표) · MuJoCo FR3(+드론) [C2 / W1,W5,W7]
한 표에 다중 baseline 동시: **{DIAL-MPC(GPU), DIAL-MPC(CPU), BK-MBD(ours, CPU),
MBD-true 오라클}** × **{제어율/주기, latency median/p95/worst, K주기 deadline-miss,
성공률}**; ours는 **N-sweep(800/400/200)**으로 어느 rate가 0-miss인지. → "GPU 없이
DIAL-MPC rate 달성"을 한눈에. (rate는 측정 후 결정, ≥20 Hz 실시간 하한)

### E2 — 롤아웃 클래스 & 충실도 & 용량 (메인 비교 표) · MuJoCo FR3 [C1+C3+capacity+C4 / W8]
한 표에 다중 baseline 동시: **{MBD-true 오라클, BK-MBD(ours), DK linear, DK-split,
MLP, linear-large}** × **{reach, strict, final err, held-out 예측오차, ms/step}**.
1 seed, N 타깃.
- **한 표가 4개 주장을 흡수**: 오라클 동등(C1) · bilinear 필요(C3) · 용량 아님(large/MLP) ·
  **MLP가 예측오차↓인데 reach↓**(open→closed 증폭, C4) → 별도 mechanism 소절/그림 불필요
  (1–2문장 + 채널별 오차는 작은 패널).
- 보조: **kinematic 1행**("물리 제거해도 linear 실패 → 결합 탓"). criterion(kim2026bilinear)은
  여기서만 인용. (model-free 시연은 별도 sim 실험 없이 E5 HW로만 — W3 범위 축소.)

### E3 — 대리모델 오차 하 어닐링 (집중 소절) · kinematic FR3 [C5 / W5,W6,D7,D8]
스케줄 ablation 표(1×800 wide / 1×4000 / 5×800 wide·narrow·anneal × reach@1cm·steps·err)
+ dispersion 직접측정 + **재계획빈도 정합** 통제. 상세 mechanism(2 레버) 설명 필요 → 소절.
kinematic이 정당화되는 곳(물리 노이즈 없는 정밀도 측정). 1 seed.

### E4 — 샘플링 vs convex Koopman-MPC (non-convex) (집중 소절) · 드론 [C6 / W4,W9,D2]
드론 window, **BK-MBD vs QP-MPC(+강화 QP 변형)**, 35 타깃, paired McNemar
(35/35 vs 2/35, viol 0, p≈2e-10). 기하 설명 필요 → 소절. **완료**, secy 벽 대체.

### E5 — 하드웨어 (집중 소절) · FR3 + 모바일 [C8 / W3일부,D13]
FR3 이동표적 추종(장애물 없음, 실시간 배치) + **모바일 유니사이클 ㄷ자**(HW non-convex,
실시간 이중계상) + **resolved-rate baseline**(D13).

## 6. 리뷰 지적 → 조치
- **W1** E1: 자작 CPU 오라클 대신 **named DIAL-MPC baseline**(`xue2025full`) —
  diffusion-annealing + full-order MJX 롤아웃을 우리 과제에서 구현, **GPU vs CPU**
  측정; 우리것은 CPU로 동일 rate. rate는 **가정 아닌 측정**(N-sweep; ≥20 Hz = 실시간
  하한, 데이터 지지 시 50 Hz). abstract는 raw 배수가 아니라 *GPU-free 실시간/deadline
  충족*으로 재프레이밍.
- **W2** 4C: tube를 abstract/기여에서 강등; 차용 페널티로만 유지.
- **W3** 새 실험 없이 **범위 축소** — sim은 참-모델 알려진(rate에 비싼) 과제로 한정,
  model-free 필요성은 E5 HW 재식별로만 시연, 미지-동역학 플랜트는 한계로 명시.
- **W4** E4: 강화 QP 베이스라인(multi-homotopy 초기화: branch별로 풀고 최선 선택; 및/또는
  tangential/CBF `zhao2025mppitan`) 추가 또는 "표준 *실시간* convex Koopman-MPC"로 범위
  명시; McNemar 유지.
- **W5** E1/E3: 50 ms-budget 오라클(N≈10) + 재계획빈도 정합 어닐링(제어주기 낮춰
  wall-clock 동일화).
- **W6** 4A/4D: 기준은 인용으로 요약; reach/floor는 remark로 강등.
- **W7** E1/E2: latency를 optimizer-only / plant-inclusive 두 열로 분리; abstract "64 ms"
  vs "50 ms 데드라인" 정리.
- **W8** §1/§4: 신규성 재중심화(차용 vs 신규 표); 전반 귀속.
- **W9** E2/E4: **1 seed**이므로 seed 클러스터링 없음 — 통계력은 **타깃 N개**에서, 성공률은
  타깃 단위 **paired McNemar**, 정착시간은 paired Wilcoxon 전면.
- **D2** 드론이 FR3 벽 대체로 해소(점질량, 비용-내-FK 비대칭 없음).
- **D3/D7/D8/D11/D12/D13**: 보고 수정 — 0.023 m 바닥 설명; ablation CI(1 seed·N타깃의
  타깃 분산); free-energy MC를 부록으로; 긴 문장 분할; 하드웨어 시간축 오차 플롯;
  하드웨어 추종에 resolved-rate 베이스라인.

## 7. 유지 / 재집계 / 신규 / 제거
- **재집계+부분 재실행 → E2 한 표로 통합**(version2 자산 활용, **seed0만·MuJoCo·N타깃**):
  기존 parity + baselines(Table III) + 모델클래스 사다리 + mechanism을 **E2 단일 표**로
  병합. 타깃을 7→N(예 35)로 확장 → 이 부분은 재실행. (kinematic은 물리-격리 1행만.)
- **유지-재프레이밍**: tube(강등, 4C), 기준(귀속, E2에서만).
- **기존 데이터로 재분석**: latency 표 분리(W7), paired McNemar(W9).
- **신규 코드/실험**: (a) **DIAL-MPC baseline**(`xue2025full`식) FR3/드론에서 GPU vs CPU +
  N-sweep + 50 ms 오라클 → E1(W1,W5); (b) 재계획빈도 정합 어닐링 + **오차 2요인(편향주입
  BK + reverse-anneal)** → E3(W5,W6); (c) 강화 QP 드론 변형 → E4(W4); (d) 하드웨어 FR3
  추종 + 모바일 ㄷ자 → E5.
- **제거**: `secy/` FR3 전신-팔 벽(드론 window로 대체).
- **완료**: 드론 window E4 — `drone/drone_window.py`, `drone_window_stats.py`,
  `drone_window_fig.py`.

## 핵심 파일
- `paper/main.tex` — abstract + contributions(tube 강등, 신규성 재중심화),
  Preliminaries(MBD만), Method 4A–4D(차용부 귀속), Experiments(위 매트릭스 + 프로토콜
  문단), FR3-벽 Section C 제거, 드론/모바일 절 추가.
- `paper/ours.bib` — `kim2026bilinear` 중립 인용; `korda2018mpc`, `bruder2021bilinear`,
  `zhao2025mppitan`, `softmanisim2024` 사용 확인.
- `drone/drone_window*.py` — Section C(존재). 강화 QP 변형 추가.
- `version 2/experiments/*` — FR3 실험 자산(**seed0만·MuJoCo 위주로 E2 재집계**);
  DIAL-MPC baseline + 재계획정합 어닐링 스크립트 추가.
- 신규: 편향주입 BK 체크포인트 + reverse-anneal 스케줄(E3); 모바일 유니사이클 ㄷ자 스크립트.
- 논문에서 제거: `secy/`(FR3 벽).

## 검증
- `pdflatex main.tex` 컴파일; orphan 주장 없음(모든 abstract/contribution 주장이 E1–E5에
  매핑; tube는 contribution에 없음).
- `grep -i "our previous\|we previously\|our prior"` in main.tex = 0; 모든
  tube/multistep/기준/드론/유니사이클 언급이 `kim2026bilinear` 인용.
- 재현: 드론 `python drone/drone_window.py && …_stats.py && …_fig.py`
  (35/35 vs 2/35, viol 0); FR3 5-seed는 version2 스크립트; 신규 실험 각각 실행 가능
  스크립트 + 기록된 수치.
