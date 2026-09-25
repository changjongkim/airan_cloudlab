> **보관 사유 (2026-09-25):** 이 문서는 실험 이력으로 보존한다. 최신 스킴과 claim boundary는 `docs/current`의 원고·모델·production gate 문서가 권위 기준이며, 과거 GPU0 전용, MIG, CloudLab 또는 4-GPU pool 가정은 현재 논문의 근거가 아니다.

# SoftWall incremental exchange 가설 감사와 중단 판정

**상태:** 2026-09-24, envelope v12 및 C135/C136 뒤 가설 기각  
**판정:** `HYPOTHESIS_REJECTED_NO_NEW_CANDIDATE`  
**목적:** observe-all controller를 completion-triggered controller로 바꿔야 조건부 service가
생기는지 exact branch enumeration으로 확인

## 1. 가설이 생긴 이유

V10은 endpoint마다 여러 completion slot을 허용해 한 home의 네 NRx를 모두 수락한다고
모델링했다. 이 상태에서는 endpoint completion이 `45,45,90,90 ms`가 되고, 모든 응답을
기다리는 현재 controller는 90 ms까지 결정을 못 한다. 그 결과 `W+Delta` 기하 후보는
42개였지만 decision-time 후보는 0개였다.

이 결과만 보면 첫 45 ms cohort 직후 certificate를 고치는 incremental controller가
필요해 보였다. V10 기반 exact branch analyzer도 C132--134 full-control mode에서 두 home의
`AI40 + control15 = 55 ms` 후보를 새로 찾았다. 후속 V12 감사는 별도 AI completion
guard 2 ms를 더해 완전한 transaction을 57 ms로 교정했다.

## 2. Runtime 대조에서 발견한 누락

실제 CUDA-IPC endpoint는 forward/backward buffer가 하나이고 `EndpointState`도
`ring_depth=1`이다. 모든 NRx를 동기 admission한 뒤 물리 완료 전에는 그 credit을 반환하지
않으므로 endpoint당 한 요청만 수락한다. C132 raw record에서도 한 home·release의 NRx 수락
수는 0--2개였다.

또 executor phase가 다르다.

- C113/C121/C122 계열은 NRx 미수락 요청의 conventional recovery를 먼저 실행한 뒤
  exchange를 시도한다.
- C125--134 global certificate executor는 그 recovery도 live credit으로 남겨 둔 채
  exchange를 먼저 시도한다.

V10은 첫 차이를 모델링하지 않았고 두 번째 순서를 모든 mode에 동일하게 적용했다.

## 3. V11 ring/phase 교정과 V12 completion-guard 교정

Checker v7과 grid v11은 각 endpoint의 physical ring depth와
`rejected_recovery_before_exchange`를 mode 축으로 추가했다.

Full-control mode 한 home의 정확한 상태는 다음과 같다.

```text
requests n                   = 4
physical NRx admission       = 2  (two endpoints × ring depth 1)
rejected mandatory recovery  = 2  (exchange 시점에도 live)
all admitted NRx observed    = 45 ms
both admitted NRx succeed    => remaining recovery = 2
first recovery start         = 155 - 2 - 2×25 = 103 ms
transaction window           = 103 - 45 = 58 ms
AI40 + 3×RPC5                = 55 ms
AI completion guard          = 2 ms
complete effective bound     = 57 ms
modeled margin               = 1 ms
```

따라서 현재 observe-all controller 자체가 이 조건부 class를 모델상 수용한다. Incremental
controller도 같은 45 ms에 두 admitted NRx를 모두 관측하므로 창은 똑같다.

| 판정 | home×AI-class 쌍 |
|---|---:|
| V12 현재 observe-all exchange 후보 | 2 |
| completion-triggered incremental 후보 | 2 |
| incremental이 새로 만드는 후보 | **0** |

V12 전체 상태 count는 `QSU5/QSN0/MI3/UQ10`으로 유지된다. 달라진 것은 QSU의 이유를
실제 physical credit과 executor order에 맞춰 설명할 수 있게 된 점이다.

## 4. Exact branch 식

시각 `t`까지 성공이 확인된 NRx가 `s`개면, 아직 성공하지 않은 요청은 모두 recovery
의무를 유지한다. 그 시점에 남은 recovery 수를 `m(t,s)`라 하면 tail compaction 뒤 가장
이른 recovery 시작은

```text
H(t,s) = D - guard - m(t,s) × B_conv
```

이고, control plane과 AI completion guard를 포함한 AI transaction은

```text
B_eff <= H(t,s) - t
```

일 때만 허가할 수 있다. `m`은 topology만으로 정해지지 않는다. NRx 미수락 recovery가
exchange 전에 이미 실행됐는지, live credit으로 남았는지를 executor mode가 결정한다.

## 5. 구현 결정

Completion-triggered controller는 현재 모델에서 새 service class를 만들지 않으므로
구현하지 않는다. 새 concurrency, thread safety, in-flight NRx/AI co-run mode를 추가하면서
얻는 모델상 이득이 0이기 때문이다. 이 결정은 새로운 optimizer 탐색을 중단한다는 기존
연구 원칙과도 일치한다.

후속 gate는 controller 교체가 아니라 **현재 full-control mode의 57 ms 조건부 class가
물리 trace에서 실제로 발생하는지 확인하는 C135**로 수행했다. 판정 항목은 다음이었다.

1. release별 NRx 수락 2개, 성공 2개, rejected recovery 2개가 동시에 있었는가
2. 그 branch에서 raw AI40 request가 선택됐는가
3. prepare+commit+AI+complete 전체가 58 ms window 안에 끝났는가
4. recovery start, deadline, bound, horizon과 credit 위반이 0인가
5. 이 증거가 없으면 QSU는 static AI35 때문인 것으로만 유지하고 exchange-only service는
   물리 미검증으로 표시하는가

C135는 두 독립 arm에서 목표 분기 288회, AI40 exchange 10회, 2,560 TB safety 위반 0으로
통과했다. C136은 기존 node를 제외한 새 A100 node의 6개 arm에서 목표 분기 908회,
AI40 exchange 28회, 7,680 TB safety 위반 0으로 재자격했다. 결합된 38/38 사건은 모두
static margin −4 ms, conditional margin +1 ms였다.

## 6. 주장 경계

허용되는 문장은 다음이다.

> After modeling the one-slot physical endpoint credit and executor-specific
> recovery phase, exact branch enumeration found that the existing observe-all
> controller already exposes the same two conditional service candidates as a
> prospective incremental controller; the incremental design adds none.

57 ms class의 physical occurrence는 C135/C136의 두 A100 node에서 확인했다. 이는
finite-sample same-family requalification이며 WCET, production L1 deadline, hardware
population 보장과 처리량 우위는 주장하지 않는다.

## Artifact

- [Envelope v11 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v11.json)
- [Envelope v12 prediction](../../results/softwall_multigpu/softwall_sharded_home_envelope_prediction_v12.json)
- [V11→V12 guard regression](../../results/softwall_multigpu/softwall_envelope_v11_v12_ai_guard_regression_v1.json)
- [C135 static counterfactual](../../results/softwall_multigpu/confirm135_static_counterfactual_audit_v1.json)
- [C136 독립 node 결과](SOFTWALL_CONFIRM136_V12_REQUALIFICATION_RESULT_KO.md)
- [C135/C136 결합 감사](../../results/softwall_multigpu/confirm135_136_combined_v12_qualification.json)
- [Incremental audit v2](../../results/softwall_multigpu/softwall_incremental_exchange_prospective_v2.json)
- [Analyzer](../../scripts_for_node/softwall_same_gpu/analyze_softwall_incremental_exchange.py)
- [Unit tests](../../scripts_for_node/softwall_same_gpu/test_analyze_softwall_incremental_exchange.py)
