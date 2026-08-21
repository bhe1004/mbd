# Annealing under Surrogate Error (§3.3 설계 문서)

> **이 소절이 하는 일은 하나다: 왜 고정 σ 가 아니라 스케줄을 쓰는지 설명한다.**
>
> Proposition 이 "넓게 시작해야 한다" 를, Remark 가 "좁게 끝나야 한다" 를 말한다.
> 둘 다 **지형과 무관하게** 성립하므로, 대리 모델이 지형을 오염시켜도 논증이 살아남는다.
> 여기에 학습 모델 특유의 사정이 하나 얹힌다 — 좁은 끝은 *느린 게 아니라 틀린 것*이 된다.
>
> 관련: `annealing_robustness.md`, `framing.md`

---

## 1. 설정

모든 후보는 참 비용이 아니라 오염된 비용으로 채점된다.

```
Ĵ(U) = J(U) + ΔJ(U)
```

ΔJ 는 모델오차가 평가 비용에 유도하는 오차장이다.

**방침 (1문장):** 우리는 이 지형이 어떻게 생겼는지 모르므로, 스케줄의 근거를 **지형의
성질에 두지 않고 갱신식 자체의 성질에 둔다.**

> 지형에 대해 아무 주장도 하지 않으므로, **지형 이야기는 아예 꺼내지 않는다.**
> free energy 식도, 평활화가 지형을 단순화하는지도, 기존 coarse-to-fine 논변에 대한
> 반박도 이 절에 넣지 않는다. 주장하지 않는 것을 반박할 이유가 없다.

---

## 2. Proposition — 도달 범위

U_k = U + σ ε_k, ε_k ~ iid N(0, I), k = 1..N 이고 U⁺ = Σ w_k U_k (w_k ≥ 0, Σ w_k = 1)
일 때, 임의의 단위방향 v 와 ρ ∈ (0,1) 에 대해 확률 1 − ρ 이상으로

```
⟨ U⁺ − U , v ⟩  ≤  max ⟨ σ ε_k , v ⟩  ≤  σ √( 2 log(N/ρ) )
                     k
```

**증명 (3줄).** 볼록결합은 최대 원소로 상계된다. ⟨ε_k, v⟩ 가 iid 표준정규이므로 가우시안
꼬리 경계 + union bound 로 Pr[ max_k ⟨ε_k, v⟩ > √(2 log(N/ρ)) ] ≤ N·e^{−log(N/ρ)} = ρ.

**무엇을 보이나.** 한 번의 갱신으로 갈 수 있는 거리에 상한이 있고 σ 에 비례한다. 새 nominal
은 후보들의 가중평균이고, 평균은 평균한 것들의 범위 밖으로 나갈 수 없기 때문이다.

**왜 필요한가.** **스케줄이 넓게 시작해야 하는 이유가 이것뿐이다.** 좁게만 뿌리면 멀리 있는
더 나은 해에 원리적으로 도달할 수 없다. 이 명제가 없으면 *"σ 작게 놓고 여러 스텝 가면
되지 않나"* 에 답할 수 없다.

**왜 굳이 형식적으로 진술하나.** 경계가 **가중치와 무관하게** 성립하기 때문이다. 가중치는
대리 모델 비용에서 나오고 그것은 틀렸다. 만약 진술이 "가중치가 합리적이라면" 에 의존했다면
모델오차가 진술을 무너뜨렸을 것이다. 의존하지 않으므로 **대리 모델 아래서도 그대로 쓸 수
있고**, 그래서 명제로 세울 값어치가 있다.

---

## 3. Remark — 정밀도 바닥

```
U⁺ − U  =  σ Σ w_k ε_k                    ← 최소점에서도 0이 아님
              k

축당 분산  ~  σ / √N_eff ,   N_eff = 1 / Σ w_k²
                                          k
```

**무엇을 보이나.** **이미 정답 위에 서 있어도** 갱신이 σ/√N_eff 만큼 밀어낸다. 후보가
무작위라 가중평균이 정확히 제자리가 아니기 때문이다. 그리고 receding horizon 이므로 그
흔들림이 **매 스텝 실제로 실행된다.**

**왜 필요한가.** **스케줄이 좁게 끝나야 하는 이유가 이것뿐이다.** σ 가 계속 크면 실행 입력이
영영 떨고, 정착 자체가 불가능하다. 이는 수렴 속도가 아니라 정상상태의 성질이다.

**왜 Proposition 이 아니라 Remark 인가.** 정확하지 않기 때문이다. 가중치가 후보와 **같은
난수**에서 계산되므로 엄밀한 분산이 아니라 **스케일링 어림**이다. 정리로 포장하지 않고
Remark 로 두며, 원고에 그 사실을 명시한다.

---

## 4. 귀결

```
Proposition  →  "넓게 시작해야 한다"
Remark       →  "좁게 끝나야 한다"
             ─────────────────────────────────────
             어느 하나도 단독으로는 스케줄을 정당화하지 못한다.
             둘을 합치면 고정 σ 는 반드시 하나를 포기하게 되고,
             스케줄은 둘을 시간으로 분리한다.
```

**지형이 어떻게 생겼는지는 한 마디도 필요 없다** — 두 진술 모두 지형과 무관하게 성립한다.

---

## 5. ★ 학습 모델 하에서 좁은 끝의 성격이 바뀐다 (신규 주장)

**참 모델 위:** 좁은 stage 는 **느릴 뿐 틀리지 않는다.** 그 고정점은 과제 목적함수의
극소이고, 더 넓은 스케일이었다면 더 빨리 찾았을 뿐이다.

**학습된 대리 모델 위:**

- 자세 의존 입력 게인을 상수로 근사할 때 남는 잔차는 **구조적 편향**이다. 주어진 자세에서
  항상 같은 방향을 가리키고, 변화의 길이 스케일은 게인이 정하지 과제 비용의 basin 구조가
  정하지 않는다 → **비용 basin 보다 훨씬 짧은 스케일**.
- 좁은 stage 는 이를 볼 **분해능**과 착취할 **가중치**를 둘 다 갖는다. 따라서 **대리 모델
  지형의 극소**로 정확하게 수렴하는데, 그것이 과제 지형의 극소일 이유는 없다.
  최적화는 성공했으나 대상이 틀린 것이다.
- 편향이 체계적으로 부호를 가지므로 재계획이 매 스텝 **재확약**한다 → 과도현상이 아니라
  **폐루프의 고정점**.

**넓은 stage 가 빠지지 않는 두 기제 (서로 다름):**

1. **평균화** — 제안이 편향의 길이 스케일보다 넓은 영역을 덮으므로 가중평균이 그 구조를
   지운다. 갱신에 들어오지 않는다.
2. **분산의 dither 작용** — §3 의 분산(정밀도에는 부채인 그것)이 실행 상태를 편향
   평형점에서 떼어낸다.

**결론:**

> **같은 성질(분산)이 스케줄 끝에서는 부채이고 중간에서는 자산이다.**
> 어떤 고정 σ 로도 대체할 수 없는 이유이며, 넓은 stage 가 **탐색 이상의 일**을 한다는 뜻이다.

---

## 6. 필수 / 선택 / 삭제

**필수 (없으면 설계 선택을 설명 못 함)**

1. Ĵ = J + ΔJ (수식 1 + 1문장)
2. 방침 1문장 — 지형이 아니라 갱신식에 근거
3. Proposition + 증명 3줄
4. 해석 1문장 — 임의 가중치에 성립하므로 ΔJ 가 가중치를 망쳐도 유효
5. Remark + 스케일링 어림임을 명시
6. 귀결 1문장 — 고정 σ 는 맞바꿔야 하고 스케줄은 시간으로 분리
7. §5 문단 — 학습 모델 하의 성격 전환

**선택 (없어도 논증 성립)**

- N 이 √(log N) 으로만 들어감 → 샘플 수는 σ 의 약한 대체재 *(§4 연산 대조군의 예고)*
- 경계는 사영 전 후보 구름에 대한 것이라는 단서
- 재계획이 도달 범위를 대체하지 못함 (중간 nominal 이 벌점을 받으므로)
- **오프셋 불변성** — softmax 가중은 후보 공통 오프셋에 불변이므로 갱신을 왜곡하는 것은
  ΔJ 의 크기가 아니라 후보 간 변동뿐이고, 이는 σ 와 함께 축소된다. 고정 스케일 샘플러가
  표본 비용에 명시적 모델오차 항을 두게 만든 이유. *튜브 부재의 원리적 근거이지만, 튜브를
  아예 언급하지 않는 논문에서 "왜 없는지" 를 설명하는 것은 불필요할 수 있다. 넣는다면 §3.3
  본문보다 관련연구나 한계 문단이 낫다.*

**삭제**

| | 이유 |
|---|---|
| free energy 식 F_σ | 아무 주장도 하지 않는데 도입할 이유가 없다. §2 가 이미 p_σ 를 다뤘고, §3.3 은 "이제 그 표적이 Ĵ 로 만들어진다" 한 문장이면 된다 |
| coarse-to-fine 전제 논의 | 기존 논변 반박은 이 절의 일이 아니다 |
| 지형 위상·평활화 관련 서술 전부 | 주장하지 않으므로 불필요 |

---

## 7. §3 / §4 경계 — 제거한 수치

**규칙:** *"실험을 하나도 안 돌린 상태에서도 참인가?"* 참이면 §3, 아니면 §4.

| 제거한 것 | 이동처 |
|---|---|
| `N=800, ρ=10⁻³ → 봉투 5.2σ` | §4.5 |
| `800→4000 이면 reach +6%` | §4.5 (연산 대조군 각주) |
| `σ=1.2 에서 좌표 21% clip` | §4.5 |
| `σ≡1.2 / σ≡0.3 / 1.2→0.3` | §4.5 (실험 조건) |
| `분산 0.060 → 0.024 rad/s` | §4.5 |
| `4.1 s → 64 ms` | §4.2 / §4.6 |

남은 수치는 `σ√(2 log(N/ρ))`, `σ/√N_eff`, `Σ w_k²` — 전부 기호값.
선행 논문 (BK-MPPI §3, DIAL-MPC III, MBD §4) 과 같은 수준이다.

**용어:** `lever(레버)` 는 쓰지 않는다. 비유가 뒤집혀 있다 (σ 가 레버이고 도달 범위·정밀도는
그 결과). 원고 L102/L296/L557 의 `lever` 를 제거하고 **reach** 와 **precision floor** 로
직접 부른다. L269 의 `leverage` → `advantage`.

---

## 8. 대응 실험 (§4.5)

**3 × 3 (모델오차 수준 × 스케줄)**, kinematic FR3, 1 cm 허용오차, 셀당 5 시드 × 7 타깃 = 35 시행.

| rollout 모델 | 넓게 고정 | 좁게 고정 | 어닐링 |
|---|---|---|---|
| **오라클** (오차 0) | 정체 예상 | **성공 예상** | 성공 |
| **BK 전체 데이터** | 30/35 | **19/35** | 34/35 |
| **BK 1/10 데이터** | ↓ | **↓↓** | 완만 |

- **표 안의 상호작용이 곧 주장이다.** 모델오차를 키울 때 *좁게* 열만 가파르게 무너지고
  *어닐링* 열은 평탄.
- **오라클 행 = 반대사실.** 좁게가 오라클에서 성공하면 *"좁은 게 그냥 나쁜 최적화기"* 라는
  대안 설명이 죽는다. 인위적 섭동 없이 진짜 모델오차만으로 얻는다.
- **1/10 행 = 용량–반응.**
- 오라클 행의 넓게-고정이 정체하면 **정밀도 바닥이 모델과 무관한 성질**임이 같은 표에서
  공짜로 드러난다.
- 각주: `1×4000 넓게` 연산 대조군 — √log N 예측대로 개선 없음.

**보조 문장 2개**

- 좁게-고정의 수렴 제어열을 그대로 굴리면 **모델 예측 0.019 m vs 실제 0.060 m**.
  참 지역최소였다면 대리 모델도 큰 잔여 오차를 예측했어야 한다 → §5 의 기제 확정.
- 정착 상태에서 상태 고정 후 계획만 반복: 첫 행동 분산 **σ=1.2 → 0.060, σ=0.3 → 0.024
  rad/s**. 정밀도 바닥의 스케일링을 폐루프 밖에서 확인.

**폐기한 실험:** free-energy 단면 60개 / cold-start 분산 (도달 범위 증거인데 리칭에서는
물리지 않음 — 벽 과제 소관) / `annealing_robustness.py` Exp 1·2 /
`explore_commit_grid.py` 5×5 + 합성 δ.

---

## 9. LaTeX 초안

```latex
\subsection{Annealing under Surrogate Error}\label{subsec:anneal_theory}

With a learned rollout the sampler never sees the true objective, and every
candidate is instead scored by
\begin{equation}
\hat J(U) \;=\; J(U) + \Delta J(U),
\label{eq:jhat}
\end{equation}
where $\Delta J$ collects the effect of model error on the evaluated cost. We do
not know this landscape, so we do not base the schedule on its properties.
Instead we use two properties of the update rule itself, which hold whatever the
landscape is.

\begin{proposition}[Directional reach of a stage]\label{prop:reach}
Let $U_k = U + \sigma \varepsilon_k$ with
$\varepsilon_k \overset{\mathrm{iid}}{\sim} \mathcal{N}(0, I_{mT})$,
$k = 1, \dots, N$, and let $U^{+} = \sum_k w_k U_k$ with any weights
$w_k \ge 0$, $\sum_k w_k = 1$.
For every unit direction $v$ and $\rho \in (0, 1)$, with probability at least
$1 - \rho$,
\begin{equation}
\big\langle U^{+} - U,\, v \big\rangle
\;\le\; \max_k\, \big\langle \sigma \varepsilon_k,\, v \big\rangle
\;\le\; \sigma \sqrt{2 \log (N/\rho)} .
\label{eq:reach}
\end{equation}
\end{proposition}
\begin{proof}
A convex combination is bounded by its largest element. Because
$\langle \varepsilon_k,v\rangle$ are i.i.d.\ standard normal, a Gaussian tail
bound followed by a union bound gives
$\Pr[\max_k \langle \varepsilon_k,v\rangle > \sqrt{2\log(N/\rho)}]
\le N e^{-\log(N/\rho)}=\rho$.
\end{proof}

A single update therefore cannot move further than the scale at which it
samples, so a basin beyond that scale is unreachable however the candidates are
weighted. The bound holds for \emph{any} admissible weights and is thus a
property of the proposal rather than of the evaluated cost: however badly
$\Delta J$ distorts the weights, it survives, which is what makes it usable
under a surrogate. The sample count enters only through $\sqrt{\log N}$, so
enlarging the population is a weak substitute for enlarging $\sigma$.

\begin{remark}[Precision floor]\label{rem:floor}
The weighted-mean update at noise level $\sigma$ moves the nominal by
$\sigma \sum_k w_k \varepsilon_k$, which does not vanish at a minimizer. Under a
locally frozen-weight approximation (the weights are computed from the same
$\varepsilon_k$, so this is a scaling heuristic rather than an exact variance),
it disperses per axis on the order of $\sigma / \sqrt{N_{\mathrm{eff}}}$,
$N_{\mathrm{eff}} = 1 / \sum_k w_k^{2}$.
In receding horizon the executed control inherits this dispersion at every step,
so a sampler at fixed $\sigma$ resists settling below it; this is a property of
its steady state rather than of its convergence rate.
\end{remark}

The two statements pull in opposite directions, and neither alone justifies a
schedule: a fixed noise level must give up one of them, whereas a schedule
separates them in time, opening at the reach and closing at the floor.

Under an exact rollout that is the whole story, and the narrow end is merely
slow: its fixed point is a minimizer of the task objective, and a wider scale
would only have found it sooner. A learned rollout changes the character of that
end. The residual left by the best constant approximation of a
configuration-dependent input gain is a \emph{structural} bias
(Sec.~\ref{subsec:prelim_bk}): for a given configuration it points the same way
every time, and it varies over the length scale of the gain rather than that of
the task cost's basins. A narrow stage has both the resolution to resolve this
bias and the weighting to exploit it, so it converges accurately to a minimizer
of the surrogate's landscape, which need not be one of the task's; because the
bias is systematically signed, receding-horizon replanning re-commits to it at
every step, and the resulting equilibrium is a fixed point of the closed loop
rather than a transient. Wide stages avoid this in two distinct ways: their
proposals span a region larger than the length scale of the bias, so the
weighted mean averages over it, and the dispersion of Remark~\ref{rem:floor}, a
liability for terminal precision, dislodges the executed state from the biased
equilibrium. The same property is thus a cost at the end of the schedule and an
asset in its interior, and the wide stages do more than explore.
```

### 9.1 확인 필요

1. `Sec.~\ref{subsec:prelim_bk}` — 구조적 편향의 근거를 Preliminaries 의 bilinear Koopman
   review 에 걸었다. 실제 라벨명에 맞출 것.
2. `\cite{kim2026bilinear}` 제목 변경 — `Linear or Bilinear: A Criterion for Koopman
   Rollouts in Sampling-Based Predictive Control`, IJCAS 투고. `ours.bib` 갱신.
3. 기존 §3.5 말미의 요약 문단 (`In summary, one control step of BK-MBD ...`) 은
   **§3.1 로 이동**.

### 9.2 분량

약 0.7 컬럼 (기존 1.0). free energy 식과 지형 논의를 들어내고 성격 전환 문단이 들어간
순감이다.
