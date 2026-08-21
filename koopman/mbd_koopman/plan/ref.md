# 기반별 참고문헌 리스트 (References by Foundation)

각 기반(B1–B4)에 인용할 수 있는 선행연구를 **anchor(반드시 인용)**와 **supporting(보강용)**으로 구분.
서로 다른 기반에 겹치는 논문은 `(공유)`로 표시.

---

## B1 — Model-Based Diffusion 프레임워크 (그대로 채택)

**Anchor**

- Chaoyi Pan, Zeji Yi, Guanya Shi, Guannan Qu, "Model-Based Diffusion for Trajectory Optimization," *Advances in Neural Information Processing Systems (NeurIPS)* 37, pp. 57914–57943, 2024. (arXiv:2407.01573)

**Supporting** — MBD가 MPPI/sampling 계열과 연결된다는 근거 (C3의 "convexity 불요" 논거 토대)

- Grady Williams, Andrew Aldrich, Evangelos A. Theodorou, "Model Predictive Path Integral Control: From Theory to Parallel Computation," *Journal of Guidance, Control, and Dynamics* 40(2):344–357, 2017.
- Grady Williams, Nolan Wagener, Brian Goldfain, Paul Drews, James M. Rehg, Byron Boots, Evangelos A. Theodorou, "Information Theoretic MPC for Model-Based Reinforcement Learning," *IEEE ICRA*, pp. 1714–1721, 2017.

---

## B2 — control-affine에서 bilinear > linear (존재성 + 근사 우월성)

**Anchor**

- Daniel Bruder, Xun Fu, Ram Vasudevan, "Advantages of Bilinear Koopman Realizations for the Modeling and Control of Systems with Unknown Dynamics," *IEEE Robotics and Automation Letters (RA-L)* 6(3):4369–4376, 2021. — 존재성 정리의 원출처. **(B4와 공유)**
- Debdipta Goswami, Derek A. Paley, "Global Bilinearization and Reachability Analysis of Control-Affine Nonlinear Systems," in *The Koopman Operator in Systems and Control*, Springer, pp. 81–98, 2020. — Koopman Canonical Transform(KCT).
- Carl Folkestad, Joel W. Burdick, "Koopman NMPC: Koopman-based Learning and Nonlinear Model Predictive Control of Control-affine Systems," *IEEE ICRA*, pp. 7350–7356, 2021. — "linear는 nonlinear actuation을 못 잡는다 → KCT로 lifted bilinear"라는 동일 논리.

**Supporting**

- Cody Bakker, Sean Rosenthal, Kathleen E. Nowak, "Koopman Representations of Dynamic Systems with Control," arXiv:1908.02233, 2019. — "유효한 linear realization이 존재 보장 안 됨"의 근거.
- Sebastian Peitz, Samuel E. Otto, Clarence W. Rowley, "Data-Driven Model Predictive Control using Interpolated Koopman Generators," *SIAM Journal on Applied Dynamical Systems* 19(3), 2020. — control-affine 보존 + bilinear surrogate. **(B4와 공유)**
- Milan Korda, Igor Mezić, "Linear Predictors for Nonlinear Dynamical Systems: Koopman Operator Meets Model Predictive Control," *Automatica* 93:149–160, 2018. — bilinear가 개선하는 "linear-input 예측기"의 표준 baseline.
- Yunbei Li, Zhaobing Liu, Yaping Cen, Kaiwei Zheng, "Data-driven Bilinear Model Predictive Tracking Control of Soft Manipulators with Unmodeled Dynamics and Unknown Disturbances Compensation," *Robotics and Autonomous Systems* 200:105397, 2026. — soft manipulator에 bilinear Koopman 적용. **(B4와 공유)**

---

## B3 — soft robot Koopman (관측 기반 모델링/제어)

**Anchor** — Bruder 3부작 (soft robot Koopman의 원조)

- Daniel Bruder, Brent Gillespie, C. David Remy, Ram Vasudevan, "Modeling and Control of Soft Robots Using the Koopman Operator and Model Predictive Control," *Robotics: Science and Systems (RSS)*, 2019.
- Daniel Bruder, Xun Fu, R. Brent Gillespie, C. David Remy, Ram Vasudevan, "Data-Driven Control of Soft Robots Using Koopman Operator Theory," *IEEE Transactions on Robotics (T-RO)* 37(3):948–961, 2021.
- Daniel Bruder, Xun Fu, R. Brent Gillespie, C. David Remy, Ram Vasudevan, "Koopman-Based Control of a Soft Continuum Manipulator Under Variable Loading Conditions," *IEEE RA-L* 6(4):6852–6859, 2021.

**Anchor** — 동적/관성 영역으로의 확장 (당신의 "동적 영역" 정당화의 핵심 레퍼런스)

- David A. Haggerty, Michael J. Banks, Ervin Kamenar, Alan B. Cao, Patrick C. Curtis, Igor Mezić, Elliot W. Hawkes, "Control of Soft Robots with Inertial Dynamics," *Science Robotics* 8(81):eadd6864, 2023. — 속도 10배·가속도 40배, 준정적→관성 영역. static/dynamic Koopman.

**Supporting**

- Daniel Bruder, David Bombara, Robert J. Wood, "A Koopman-based Residual Modeling Approach for the Control of a Soft Robot Arm," *International Journal of Robotics Research (IJRR)* 44(3):388–406, 2025.
- Lu Shi, Zhichao Liu, Konstantinos Karydis, "Koopman Operators for Modeling and Control of Soft Robotics," *Current Robotics Reports*, 2023. — 서베이(진입점으로 유용).
- "Multi-segment Soft Robot Control via Deep Koopman-based Model Predictive Control," arXiv:2505.00354, 2025. — DK-MPC. deep Koopman을 soft robot MPC에 통합(당신 방법에 가장 근접). *저자·게재정보 최종본에서 확인 권장.*
- Lei Han, Kairui Peng, Wenfeng Chen, Zhaobing Liu, "A Data-driven Koopman Modeling Framework with Application to Soft Robots," *International Journal of Control, Automation and Systems* 23(1):249–261, 2025.

---

## B4 — bilinear의 실시간 최적화 트릭 (z[0] 고정 국소선형화 = 당신이 넘어서는 대상)

**Anchor** — 식으로 직접 뒷받침 ("MPC는 이렇게 회피한다"의 물증)

- Bruder, Fu, Vasudevan (RA-L 2021) — **B2 anchor와 동일**. 식 45–46이 z[0] 고정 국소선형화.
- Li, Liu, Cen, Zheng (RAS 2026) — **B2 supporting과 동일**. 식 12가 동일 트릭.

**Supporting** — deep bilinear Koopman을 MPC에 쓴 최신 계열 (당신 방법과 대비)

- Sebastian Peitz, Samuel E. Otto, Clarence W. Rowley (SIAM JADS 2020) — **B2와 공유**. interpolated Koopman generators.
- Minghao Wang, Xuyang Lou, Baotong Cui, "Deep Bilinear Koopman Realization for Dynamics Modeling and Predictive Control," *International Journal of Machine Learning and Cybernetics*, 2024.
- Dongdong Zhao, Boyu Li, Fuxiang Lu, et al., "Deep Bilinear Koopman Model Predictive Control for Nonlinear Dynamical Systems," *IEEE Transactions on Industrial Electronics* 71(12):16077–16086, 2024.

---

## 참고: 기반은 아니지만 반드시 함께 인용 (C2 대비용)

"병렬 시뮬 없이 실시간" novelty(C2)를 방어하려면 MBD의 실시간 후속작을 반드시 대비 인용해야 함.

- Haoru Xue, Chaoyi Pan, Zeji Yi, Guannan Qu, Guanya Shi, "Full-Order Sampling-Based MPC for Torque-Level Locomotion Control via Diffusion-Style Annealing" (DIAL-MPC), *IEEE ICRA*, pp. 4974–4981, 2025.

---

## 투고 전 확인 사항

- **DK-MPC** (arXiv:2505.00354): 저자·최종 게재정보 미확정 → 최종본 대조.
- **Zhao et al.**, **Han et al.**: 공저자 일부를 `et al.`로 표기 → 투고 전 전체 저자 채우기.
- 페이지·권호 표기는 각 저널/학회의 공식 게재본 기준으로 최종 대조 권장.