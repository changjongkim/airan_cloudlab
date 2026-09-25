> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall substrate + feasibility envelope 논문 전환안 — 2026-09-23

## 최종 연구축

새로운 NRx/AI 선택 optimizer의 우월성 추적을 종료한다. C78/79, C96, C102의
세 독립 단계에서 exact 또는 guarded joint가 강한 greedy/max-radio 기준선과 같은
결정을 냈고, C102의 사전 고정 outcome gate도 실패했다. 이후 정책은
`max-radio + exact recovery recourse`를 단순하고 강한 기본 controller로 고정한다.

논문의 연구 질문은 다음으로 바꾼다.

> MIG 없이 MPS만 사용하는 AI-and-RAN GPU에서, optional NeuralRx의 모든 실패·지연
> 분기에 대해 RAN 복구 가능성을 유지하면서 조건부 slack을 AI 요청에 줄 수 있는가?
> 가능 영역과 불가능 영역을 어떤 실행 계약으로 예측할 수 있으며, 그 경계에서 필요한
> 최소 runtime 메커니즘은 무엇인가?

## C102가 실제로 말하는 것

C102의 100개 중 61개 `infeasible`을 scheduling infeasibility로 해석하면 안 된다.
[사후 구조 분해](../../results/softwall_same_gpu/confirm102_feasibility_posthoc.json)는
61개 전부가 최대 두 NRx의 예측 radio gain으로도 사전 `radio_floor`에 도달하지 못한
경우였고, all-fail scheduling이 안전한 subset을 찾지 못한 경우는 0개였음을 보였다.

실행 가능한 39개에서도 조건부 복구 효과는 사라지지 않았다.

- 37/39에서 NRx subset에 따라 예상 AI 용량이 달랐다.
- 20/39에는 max-radio의 0.01 radio guard 안에 복수 subset이 있었다.
- 그 guard 안에서 max-radio보다 예상 AI가 높은 사례는 **0/39**였다.
- AI가 더 높은 선택은 8/39에 있었지만 모두 radio guard 밖의 tradeoff였다.

따라서 정확한 음성 결론은 “feasible 상태에서 제약이 전혀 걸리지 않는다”가 아니다.
**조건부 용량과 선택지는 존재하지만, 안전하고 radio-noninferior한 AI 개선 방향이
없어 max-radio가 그대로 최적**이라는 것이다. 이것이 optimizer 중단의 근거다.

## 논문 기여 구조

### C1. Conditional-recovery contract

각 optional NRx에 conventional 복구 credit을 부여하고, 모든 미해결 NRx가 동시에
실패하거나 늦는 분기의 충돌 없는 복구 일정을 executable certificate로 유지한다.
평균 성공률이나 MPS cap을 안전 증명으로 사용하지 않는다.

### C2. Atomic recovery-credit substrate

여러 셀의 recovery credit retiming과 bounded AI lease 발급을 하나의 transaction으로
처리한다. CUDA 완료 fence, IPC buffer, endpoint credit, timeout 뒤 미완료 kernel,
single radio commit까지 동일한 물리 lifecycle에 묶는다.

### C3. Feasibility envelope

다음 축에서 `qualified-safe-useful`, `qualified-safe-no-AI-slack`,
`mandatory-infeasible`, `unqualified-mode`의 경계를 예측하고 물리 실행으로 검증한다.

- 셀 수와 endpoint 수
- period `P`, MAC expiry `D`, commit guard
- mode별 `B_NRx`, `B_conv_path`, `B_AI`
- MPS cap과 client priority
- NRx 실패 상관도와 관측 시각
- warm/cold, GC, worker restart/teardown lifecycle
- resident receiver/LLM 메모리

기본 필요조건은 single conventional lane에서
`n_cells × B_conv_path <= P`이고, 한 release의 wait-observe 경로에는
`B_NRx_pair + n_cells × B_conv_path + guard <= D`가 필요하다. 실제 runtime은
이 폐형식 조건보다 구체적인 per-request interval certificate를 사용한다. 이미 제출한
kernel, host 관측 지연, 메모리 residency를 포함하지 않는 계산은 envelope 자격이 없다.

### C4. Negative policy result

조건부 용량이 있어도 현재 자격화된 실행 영역에서는 복잡한 joint optimizer가
max-radio보다 안전하고 radio-noninferior한 AI 이득을 만들지 못했다. C78/79, C96,
C102의 독립 음성 결과와 C103–105의 불안정한 scalar interference 결과를 함께 제시한다.
이는 “optimizer가 필요 없다”는 보편 정리가 아니라, 측정한 mode와 계약에서
substrate가 핵심이고 optimizer 복잡도는 근거가 없다는 경계 결과다.

## 스킴

1. TB release에서 모든 mandatory conventional credit을 먼저 예약한다.
2. 자격화된 NRx bound와 request cutoff를 만족하는 optional NRx만 허가한다.
3. NRx 성공, 실패, late, AI 도착, AI 완료 사건마다 all-fail certificate를 다시 만든다.
4. 새 복구 일정과 AI lease가 함께 안전할 때만 원자적으로 commit한다.
5. 첫 비선점 행동만 실행하고 다음 사건에서 재계획한다.
6. GPU completion fence 전에는 어떤 물리 credit도 반환하지 않는다.

Radio subset 선택은 C102 이후 `max-radio`로 고정한다. Exact recovery recourse는 작은
상태의 실행 증명서 생성에 사용하며 새로운 optimizer 기여로 세지 않는다.

## 비교군

| 비교군 | 제공하는 기능 | 질문 |
|---|---|---|
| Uncontrolled MPS | 동일 cap과 작업, recovery certificate 없음 | MPS만으로 deadline을 지킬 수 있는가 |
| Static safe calendar | channel gate, max-radio, 고정 worst-case 복구 예약, bounded AI | 고정 예약의 안전 비용은 얼마인가 |
| Safe work-conserving baseline | 같은 all-fail 검사와 조기 복구, 빈 구간 AI; 복수 credit+lease 원자 교환 없음 | 단순 안전한 회수만으로 충분한가 |
| SoftWall substrate | 복수 credit retiming, atomic AI lease, physical fence lifecycle | 원자 조건부 slack 가상화가 안전한 유효 AI를 늘리는가 |
| Offline future oracle | 미래 CRC와 도착을 아는 상한 | 남은 개선 여지는 얼마인가 |

모든 안전 비교군에 같은 PHY feature, max-radio 선택, endpoint 정보, MPS cap,
서비스 bound, fault 처리와 single-commit 규칙을 준다. 정책 선택 차이를 만들기 위해
약한 staged-min을 주 baseline으로 사용하지 않는다.

## 최종 workload와 지표

AI-and-RAN workload는 KDD'25 BurstGPT의 실제 Azure OpenAI request timestamp와
input token 수를 사용한 Qwen2.5-1.5B prefill replay로 고정한다. 원본과 CC-BY-4.0
license를 `data/public/burstgpt`에 보존했다. 원본의 가장 조밀한 유효 60초 구간을
결정적 규칙으로 선택하며, token 수를 허용 context bucket으로 올림하고 512보다 큰
입력만 512로 cap한다. SLO는 원 trace에 없으므로 별도의 synthetic sensitivity로
명확히 표시한다.

Primary safety 지표:

- RAN deadline miss, conventional/NRx/AI bound 위반은 모두 0
- duplicate/stale commit, residual recovery credit, fence 전 lease 반환은 모두 0
- all-fail fault injection coverage와 lifecycle별 결과

Primary utility 지표:

- offered request와 offered token value
- deadline 내 완료 request와 token value
- late, expired, rejected request
- 동일 trace·PHY의 paired 차이와 bootstrap 95% CI

Envelope 지표:

- 분석 모델의 feasible/infeasible 분류 정확도
- 예측 경계와 물리 경계의 거리
- false-safe 수는 반드시 0
- mode별 bound와 메모리/lifecycle 자격 여부

## 사전 효과 크기와 중단 기준

과거 약 1% 처리량 효과는 C63/64, C69, C71/72, C103/104에서 방향이 유지되지 않았다.
향후 처리량 우위는 각 독립 ABBA pair에서 다음을 동시에 만족할 때만 주장한다.

- 적시 token value 상대 개선이 **각 pair에서 2% 이상**
- paired bootstrap 95% CI 하한이 0보다 큼
- 모든 safety gate 통과
- 동일 원인 ablation이 개선 방향을 설명

2%는 보편적 통계 검출한계가 아니라 이 실험에서 사전 고정할 최소 공학적 효과다.
표본 수는 pilot 분산으로 power calculation한 뒤 protocol에 고정한다. 이 기준보다 작은
차이는 성공으로 세지 않는다.

추가 joint optimizer 탐색은 종료한다. 실제 request workload에서도 SoftWall과
safe work-conserving baseline이 같으면 처리량 우위 주장을 종료하고, envelope 예측과
atomic/lifecycle safety를 주 결과로 쓴다. Envelope 모델이 false-safe를 내거나 독립
물리 반복에서 경계를 예측하지 못하면 substrate+envelope 논문 주장도 축소한다.

## 남은 완료 gate

1. BurstGPT→bounded Qwen request mapping과 synthetic SLO를 사전 protocol로 봉인
2. context bucket별 isolated/co-run 전체 host bound 자격화
3. static safe, safe work-conserving, SoftWall의 동일 trace ABBA 구현
4. 셀 수·P/D·fault/lifecycle 축의 envelope grid 실행
5. 독립 node/seed 반복과 false-safe 감사
6. 결과표, 재현 스크립트, frozen source hash를 ledger에 연결

이 여섯 gate 전에는 탑 컨퍼런스 제출 준비 완료로 판정하지 않는다.
