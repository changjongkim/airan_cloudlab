# SoftWall 공동 스킴 후보: 복구 의무를 가진 NeuralRx와 허용 AI의 온라인 배치

> **역사 문서:** C102 outcome gate 실패로 joint-optimizer 우위 추적은 종료했다.
> 현재 설계와 논문 구성은 [논문 구조 검토본](SOFTWALL_PAPER_STRUCTURE_KO.md)과
> [substrate/envelope 전환안](SOFTWALL_SUBSTRATE_ENVELOPE_PAPER_PLAN_KO.md)을 따른다.

**상태 (2026-09-21): 연구 가설·설계안. 구현·성능·신규성 입증 전.**
[더 구체적인 노벨리티·스킴 판정 (2026-09-22)](SOFTWALL_NOVELTY_SCHEME_DECISION_KO.md)은
NeuralRx의 conventional 대비 조건부 추가 성공 확률과 복구 의무·AI 허가를 함께 정의한다.
[최신 판정](SOFTWALL_NOVELTY_DECISION_KO.md)의 “현재 노벨리티 미확보”를 변경하지 않는다.
[2026-09-22 검증 기록](SOFTWALL_VALIDATION_PROGRESS_KO.md)에서 단순 복구 자리 이동은
강한 greedy도 재현함을 확인했다. 원자적 재배치의 CPU primitive는 구현됐지만 공동 정책의
GPU 우위는 아직 없다.

## 논문에서 입증할 한 문장

같은 GPU·MIG OFF·MPS에서 optional NeuralRx가 만드는 **조건부 conventional 복구 의무**를
여러 셀 사이에서 사건별로 재배치하고, 그때 실제로 비는 구간에 **허용된 AI work unit**을
배치한다. 모든 허용 NRx 실패·지연 분기에 대해 각 TB의 MAC expiry 전 복구 일정을 계속
보유하면서, 같은 정보·자원·안전장치를 갖춘 강한 결합 baseline보다 무선 요구를 유지하고
더 많은 유효 AI를 완료할 수 있는지가 논문의 가설이다. MPS는 이 정책의 GPU 실행 수단이다.

## 핵심 메커니즘: 조건부 복구 credit의 원자적 교환

각 optional NRx는 성공이 확정되기 전까지 같은 TB의 conventional 전체 경로를 끝낼 수 있는
**복구 credit**을 빚진다. credit은 요청에 귀속되지만, calendar에서 차지하는 *시각과 순서*는
고정되지 않는다. 스케줄러는 release, CRC 성공/실패, late completion, AI 도착 때마다
남은 복구 의무의 all-fail 일정을 다시 구성한다. 새로운 일정이 모든 expiry를 만족하고
GPU/IPC 물리 credit이 유효한 것을 확인한 뒤에만 기존 예약을 원자적으로 바꾸고 AI unit을
허가한다. NRx 성공이면 해당 의무를 소멸시키고, 실패하면 새 예약에서 conventional을 실행한다.
정책은 이 교환 가능성까지 고려해 NRx 수락, endpoint, 복구 순서, AI 허가를 결정한다.

**결정이 달라지는 작은 예시(측정 결과 아님):** 동시에 온 A/B가 각각 expiry 130 ms,
commit guard 2 ms, 자격 검증된 conventional 전체 경로 상한 25 ms를 가진다고 하자.
고정 예약은 A=[78,103], B=[103,128] ms다. B가 20 ms에 성공하고 A가 60 ms에 실패했을 때,
78 ms에 도착한 AI unit은 20 ms가 필요하고 100 ms까지 끝나야 한다. 기존 A 예약을 그대로
두면 AI의 78–98 ms 실행을 허가할 수 없다. A 복구를 실패 직후 60–85 ms에 앞당겨도 AI는
85–105 ms가 되어 늦는다. 반면 B의 성공으로 비워진 뒤쪽 자리에 A 복구를 [103,128] ms로
**이동한 뒤** AI를 [78,98] ms에 넣으면 두 마감시각을 지킨다. 이는 이동·commit overhead,
host-to-commit 상한, 동시 실행 상한이 검증됐을 때의 설명용 일정이다. 특히 **AI 수요를 보고
뒤로 재배치하는 강한 baseline도 이 예시를 재현할 수 있으므로, 이 한 예시가 노벨리티의
증거는 아니다.** 실제 기여는 여러 요청·상관 실패·AI 마감시각에서 안전한 교환을 선택하는
정책과 그 이득을 강한 재배치 baseline에 대해 입증해야 성립한다.

`FallbackCalendar`는 고정 `fallback_latest_start_ns` 예약과 `retime_earlier`에서 출발했다.
후속 CPU runtime에는 **NeuralRx 물리 완료 후 복수 credit의 원자적 재배치**를 넣고 단위
시험을 통과시켰다. 아직 AI lease를 같은 원자적 결정에 묶거나 GPU controller로 실행하지
않았고, 미완료 NeuralRx의 분기별 물리 co-run 검사도 없다. 단순 MPS cap, 조기 복구,
성공 뒤 예약 해제, 빈 구간 AI 채우기는 구현 요소로 취급한다.

## 실행 계약

| 입력 | 정책에서 쓰는 값 | 선행 gate |
|---|---|---|
| 무선 요청 `i` | release `r_i`, 대상 DU에서 유도한 MAC/CRC expiry `d_i`, 관측 가능한 채널 feature, conventional/NRx의 추정 무선 효용 | [실제 타이밍 계약](SOFTWALL_PUSCH_TIMING_CONTRACT_KO.md), 비중복 held-out 채널과 online feature 비용 |
| GPU 작업 | endpoint별 queue·물리 미완료 credit, MPS client/phase, NRx pair·NRx/복구·AI/필수 경로의 **전체 host-to-commit** 서비스 상한 | 동시 실행 모드별 검증. 미검증 모드는 동시 실행 금지 |
| conventional 복구 | 모든 허용 NRx 실패가 동시에 일어날 때도 충돌하지 않는 lane interval과 commit guard | `B_conv_path`를 GPU event가 아닌 host 결정→commit 반환으로 검증 |
| 허용 AI | 고정된 모델·입력 shape·kernel/메모리 접근·최대 work-unit·RPC timeout, 살아 있는 MPS client | 포화 HBM, 동적 할당/종료, 미완료 unit은 검증 전 허용 집합에서 제외 |

**우선 구현할 물리 경계:** 가능한 경우 NeuralRx 실행을 상한이 있는 graph/engine 단계로
나누고 단계 사이의 완료 fence에서 다음 단계 진행 여부를 결정한다. 복구 cutoff 전에
마지막 허용 단계의 **GPU 물리 완료**를 확인하고, 완료 가능성이 없으면 뒤 단계의 제출을
중지한 다음 conventional을 실행한다. 이미 제출한 CUDA kernel을 논리적으로 취소했다고
간주하지 않는다. TensorRT graph 분할 가능성, stage별 상한, 모델 정확도/처리량 비용을
측정해야 하며 현 엔진에서 아직 구현되지 않았다. 분할이 불가능하면 미완료 NRx와
conventional이 겹칠 때에도 mandatory 전체 경로 상한이 유지된다는 별도 co-run 계약이
필요하다. 그 계약도 확보하지 못하면 같은 요청의 late 복구 보장 가설은 기각한다.

모든 요청은 먼저 conventional-only로도 수용 가능한지 검사한다. 수용할 수 없는 offered load를
NRx의 평균 성공률로 숨기지 않는다. 현 25 ms 예약 계약·단일 conventional lane에서
`n_cells × 25 ms ≤ P`는 모든 실패를 허용한 지속 부하의 조건부 필요조건이다.
현재 P150/D130 실험은 대상 DU의 production expiry를 입증하지 않는다.

## 온라인 정책: 복구 증명서를 붙여 한 단계씩 실행

1. 매 release와 NRx 완료/실패, AI 완료, endpoint 상태 변화에서 현재 물리 미완료 작업을
   먼저 반영한다. 요청 소유 PHY 입력과 endpoint 소유 IPC slot을 분리하고, 완료 fence 전에는
   buffer·endpoint credit을 반환하지 않는다.
2. 모든 미완료 무선 요청의 **all-fail 복구 일정**을 만든다. 각 conventional 구간은 같은 lane에서
   겹치지 않고, 해당 요청의 expiry−guard 전에 끝나야 한다. 이 일정이 없으면 optional NRx와
   AI를 허가하지 않는다. NRx 없이도 필수 경로가 불가능하면 그 부하는 연구 계약 밖으로 표시한다.
3. 작은 동시 요청 집합에서 `(NRx 수락/거절, endpoint, 복구 순서와 cutoff, 다음 AI unit)`의
   후보를 열거한다. 각 후보를 *성공·실패·late completion의 허용 분기 모두*와 검증된
   co-run 상한으로 검사한다. 기존 고정 예약보다 이득을 낼 수 있는 행동을 찾되, 실시간
   결정 비용도 expiry 예산에 포함한다. 상태 수가 크면 동일 검사를 보존하는 bounded heuristic을
   사용하고 작은 경우의 정확 최적해와 gap을 보고한다.
4. 안전한 후보 중에서 무선 효용 하한을 지키는 계획만 남긴 뒤, **완료되는 AI unit 가치**를
   최대화한다. 채널 feature가 아직 무효인 현재 상태에서는 이 최적화의 효용 항을 임의의
   true SNR·미래 CRC로 채우지 않는다. 지금은 구조·불변식과 결정 차이만 검토한다.
5. 첫 non-preemptive 행동만 commit하고 다음 사건에서 다시 계산한다. 새 all-fail 일정이
   확정되기 전에는 기존 credit을 해제하거나 AI를 시작하지 않는다. 성공으로 사라진 복구
   의무의 자리에 다른 요청의 예약을 앞이나 뒤로 옮길 수 있다. NRx의 다음 단계나
   AI unit은 남은 복구 cutoff와 검증한 최대 blocking 시간을 넘지 않을 때만 제출한다. NRx 성공이면 CRC가
   물리적으로 완료되고 단일 commit이 확인된 뒤 그 요청의 복구 credit을 해제한다. 알려진
   CRC 실패면 새 일정에서 복구한다. late/stale 결과는 폐기하되 그 GPU
   작업의 credit은 물리 종료까지 점유한다. AI timeout도 동일하게 처리한다.

**안전 불변식:** 어떤 시각에도 허가된 모든 무선 요청에 대해, 모든 허용 실패 분기에서
`commit_i ≤ d_i`가 되는 충돌 없는 conventional 복구 계획이 남아 있어야 한다. 또한 아직
끝나지 않은 GPU/IPC 작업의 자원을 재사용하지 않는다. 이 불변식은 검증된 서비스 상한과
고정한 허용 AI 집합 안에서만 의미가 있다. 표본 최대·p99는 WCET 증명이 아니다.

## 기존 조합과 구별할 정확한 지점

같은 GPU의 RAN+AI 공유·MPS cap은 [YinYangRAN](https://jaayala.github.io/papers/24_infocom_2.pdf),
queue/무선 자원 공동 제어는 [CloudRIC](https://dspace.networks.imdea.org/bitstream/handle/20.500.12761/1795/mobicom24-final428_authors_v.pdf?sequence=1),
AI/conventional PHY 선택은 [ARCHES](https://arxiv.org/abs/2604.23397),
MPS+시간 분리형 허가는 [DARIS](https://arxiv.org/abs/2504.08795)에 이미 있다.
[OCUDU dApp 플랫폼](https://arxiv.org/html/2609.07843v1)도 conventional 경로를 보존한다.
특히 OCUDU Class A는 enqueue 실패 때 같은 호출에서 conventional을 쓰지만, 이미 enqueue된
작업의 늦은 완료는 결과를 사용하고 incident/breaker로 대응한다고 설명한다.
따라서 우리 차별 가설은 **여러 동시 PUSCH의 아직 풀리지 않은 복구 의무, 물리 미완료 NRx와
AI의 간섭, 유효 무선 효용, AI 허가량을 한 결정으로 묶고 같은 요청의 late 복구를 expiry
전에 성립시키는 것**이다. 논문에서 이 조합의 최초성을 단정하지 않는다. 일반 primary–backup
스케줄링과도 문제·가정·이득을 대조해야 한다.

강한 baseline에는 같은 채널 feature, queue 정보, MPS 배치, 조기 복구, bounded AI,
buffer/fault 안전장치를 모두 준다. 비교 대상은 (a) 튜닝된 고정 채널 gate+queue-aware 배치+
고정 recovery calendar+bounded AI, (b) AI 수요를 보는 greedy NRx gate와 빈 구간 AI 채우기,
(c) AI 수요를 보고 앞뒤로 복구 예약을 옮기는 안전한 greedy 재배치,
(d) 작은 문제의 정확 최적해다. 단순 무제어 MPS·조기 복구 OFF만을 주 baseline으로 삼지 않는다.
기록할 정책 차이는 요청별 `(NRx 수락, endpoint, 복구 시작, AI 허가)`와 그 시각에 실제로
알 수 있었던 정보다. greedy 재배치까지 같은 행동과 성과를 재현하면 알고리즘 노벨리티
후보를 폐기한다.

## 유한한 go/no-go 순서

1. **오프라인 결정 gate:** 실제 도착·채널·AI 수요의 서로 겹치지 않는 trace를 고정하고,
   고정 예약, AI 수요 인지 greedy 재배치, 작은 상태의 정확 최적해와 비교한다. 제안 정책이
   greedy와 다른 안전한 결정을 내리고 AI 완료량/무선 효용에서 이기는 상태가 있는지 찾는다.
   없으면 GPU campaign을 하지 않는다. 현재 Confirm55는 AI 허가 29,910개가 전부 무선 처리
   후여서 이 gate를 통과할 사례가 없다.
2. **물리 gate:** 대상 DU expiry, AI 없는 필수 경로 용량, 외부 PHY 호환성, online 채널 gate,
   그리고 실제 필요한 NRx pair·NRx/복구·AI/필수 경로 동시 상한을 검증한다. 실패하면 더 긴
   임의 deadline으로 보증 주장을 옮기지 않는다.
3. **한 번의 사전 고정 paired 비교:** 같은 seed/trace·offered load·GPU budget에서 deadline,
   paired 무선 정답/효용, 완료 AI unit, 동시 실패·late 결과·buffer credit·controller overhead를
   함께 보고한다. 같은 안전·무선 요구에서 강한 baseline 대비 반복 가능한 유효 AI 순증이
   없으면 공동 정책 우월성 주장을 중단한다. 정확 최적해와의 차이도 보여 알고리즘 가치와
   단순 구성요소 효과를 분리한다.

**현재 추천 순서:** 새 GPU 반복 실험이 아니라 `실제 expiry·PHY/feature 계약 확정 → 오프라인
결정 차이와 안전 증명서 → 필요한 co-run 상한의 제한된 측정 → paired 비교`다. 앞의 두 단계에서
실패하면 이 방향을 접거나, 별도 인과 측정 문제로 피벗한다.
