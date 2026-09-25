# C161 2단계 실제 GPU fault 결과

**결합 판정:** `C161_PHASE2_TWO_NODE_PASS`  
**범위:** A2 stale/duplicate NeuralRx, A3 post-fence reply delay, A5 pre-fence channel loss,
A6 stale/duplicate recovery response  
**결합 결과:** `results/softwall_multigpu/c161_phase2_two_node.json`

## 검증한 의미론

- **A2:** actual NeuralRx outcome batch를 정상 적용한 뒤 같은 transaction과 prior-epoch event를
  다시 전달한다. Duplicate/stale event는 recovery debt와 generation을 바꾸지 않아야 한다.
- **A3:** Qwen CUDA completion 뒤 matching marker를 먼저 공개하고 RPC response만 100 ms
  지연한다. Controller는 marker의 lease ID와 물리 완료시각을 검사하고 현재 detection time에서
  recovery certificate를 다시 계산한 뒤 lease를 retire한다. 이후 AI는 quarantine한다.
- **A5:** Qwen CUDA launch 뒤 controller channel을 끊고 completion marker는 공개하지 않는다.
  Controller는 `physical_fence_required`로 retire를 거절하고 ambiguous lease token을 유지하며
  이후 AI admission을 닫는다. Mandatory RAN은 계속 실행한다.
- **A6:** 실제 conventional response를 한 번 처리한 뒤 duplicate와 prior-epoch response를
  다시 전달한다. 추가 radio commit과 추가 physical recovery는 없어야 한다.

A3와 A5는 terminal ambiguity이므로 각각 fresh Qwen/NRx/shared-cuPHY process set에서 실행했다.
Fault 이후 최소 50 radio epoch를 요구했다. A2와 A6는 node당 각각 10회의 duplicate/stale pair를
주입했다.

## 권위 campaign

| Campaign | Job / node | Seed | arm 순서 | 판정 |
|---|---|---:|---|---|
| Development | 58859872 / nid001145 | 41,000,000 | A2 → A3 → A5 → A6 | 4 arm PASS |
| Holdout | 58859986 / nid001069 | 42,000,000 | A6 → A5 → A3 → A2 | 4 arm PASS |

Holdout은 development node를 실행 전에 제외했다. 두 campaign은 같은 32-source hash를 사용했고
서로 다른 allocation, node, seed와 반대 arm 순서를 사용했다.

## 결합 결과

| 항목 | Development | Holdout | 합계/최대 |
|---|---:|---:|---:|
| Radio epoch | 260 | 260 | 520 |
| Actual NeuralRx request | 1,040 | 1,040 | 2,080 |
| Actual NeuralRx success | 751 | 759 | 1,510 |
| Physical conventional recovery | 289 | 281 | 570 |
| Qwen physical unit | 121 | 118 | 239 |
| A2 duplicate/stale pair | 10 | 10 | 20 |
| A6 duplicate/stale pair | 10 | 10 | 20 |
| A3/A5 terminal fault | 2 | 2 | 4 |
| Terminal fault 뒤 radio epoch | 130 | 130 | 260 |
| Radio single commit | 1,040 | 1,040 | 2,080 |
| Deadline miss | 0 | 0 | 0 |
| NRx host path 최대 | 26.361 ms | 24.997 ms | 26.361 ms |
| Recovery path 최대 | 10.795 ms | 9.659 ms | 10.795 ms |
| Qwen GPU 최대 | 52.933 ms | 52.052 ms | 52.933 ms |
| Radio commit 최대 | 107.139 ms | 110.580 ms | 110.580 ms |

각 arm은 source, node/MIG inventory, sample, actual NRx transport, input prestaging, full recovery
preflight, common transition, fault semantics, Qwen identity/bound/start, recovery contract, single timely
commit, IPC lifecycle와 process error를 검사한 12개 gate를 모두 통과했다.

## 실패 사슬과 교정

세 실패를 삭제하지 않고 별도 artifact로 보존했다.

1. **Quarantine sentinel 오류:** AI quarantine을 10초짜리 가짜 lease로 표현했지만 debt가 0인
   상태에서는 그 lease가 허용됐다. Quarantine은 workload 크기가 아니라 admission state여야
   한다. Recovery-only certificate를 직접 만드는 명시적 branch로 교정했다.
2. **Marker lookup 오류:** Qwen completion marker는 lease finish 전에 존재했지만 RPC timeout을
   감지한 시각이 polling deadline 뒤라 analyzer가 파일을 한 번도 읽지 않았다. Deadline 뒤에도
   즉시 한 번 읽고 marker의 `completed_ns`를 lease finish와 대조하며, recovery는 더 늦은
   detection time에서 다시 계산하도록 교정했다.
3. **독립 node NRx bound 실패:** 같은 최종 source의 `nid001044`에서 sequence 1 두 owner가
   40.068/47.470 ms를 관측해 45 ms bound를 한 건 초과했다. Worker 자체 최대는 10.194 ms여서
   first-round host/response observation tail이었다. Bound를 늘리지 않고 해당 node/lifecycle을
   UQ로 남겼다. 이후 다른 두 node만 qualified PASS다.

이 세 실패는 각각
`c161p2a_quarantine_sentinel_failure.json`,
`c161p2b_marker_deadline_audit_failure.json`,
`c161p2c_first_round_nrx_bound_failure.json`에 있다.

## 논문 주장과 한계

C160 상태 모델과 C161 1·2단계를 합치면 A0--A6의 계획된 fault 의미론을 실제
NeuralRx/shared-cuPHY/Qwen 경로에 연결했다. Matching fence가 있는 response loss와 fence가
없는 channel loss를 서로 다르게 처리하고, 두 경우 모두 AI를 fail-closed로 격리하면서 radio를
계속 처리한 것이 핵심 결과다.

이는 qualified warm P180 node 두 곳의 finite-sample 결과다. `nid001044` 반례 때문에 같은 A100
family 전체의 45 ms bound를 주장하지 않는다. GPU/driver hang, process restart 뒤 durable
reconciliation, cold/long-idle lifecycle, production `d_MAC`, 다른 GPU family는 UQ다.
