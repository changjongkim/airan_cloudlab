# DART-Rx v2 · 실행 결정 요약

> **Novelty source of truth:** `DART_RX_CANONICAL_NOVELTY_THESIS_KO.md`  
> DART-Rx의 핵심은 transport/routing 조합이 아니라, remote execution과 local
> recovery를 함께 예약하고 유효한 결과 하나만 공개하는 DART transaction이다.

## 한 문장 결론

**MIG/MPS/MIG+MPS/P2P/GDR를 모두 공정하게 비교한다.** 그 비교에서 드러난
contention과 fragmentation을, deadline과 결과 유효기간을 이해하는 remote
accelerator command/commit architecture로 해결한다. 비교는 필수 evidence이고
DART-Rx가 최종 contribution이다.

## 문제

> **고정 격리로 보호된 real-time PHY가, 재구성 없이 격리 밖의 optional NRx
> capacity를 빌리면서도 deadline과 output validity를 어떻게 보존할 것인가?**

AI-RAN의 L1은 deadline이 있고 NRx utility는 channel에 따라 달라진다. MIG는 L1
interference를 막지만 NRx capacity를 고정한다. burst 때 한 NRx queue는 넘치고 다른
eligible endpoint는 놀 수 있으며, dynamic MIG는 drain/reload outage 때문에 slot
timescale에서 쓸 수 없다.

remote NRx로 넘기는 것만으로도 충분하지 않다. 현재 CUDA/RDMA command에는 다음 의미가 없다.

- 이 request가 언제까지 끝나야 하는가
- 늦을 request를 시작하지 말아야 하는가
- late result를 버려야 하는가
- conventional fallback을 언제 시작해야 하는가

즉 isolation-elasticity는 관찰되는 현상이고, command semantic gap은 safe borrowing을
막는 원인이며, dual reservation과 epoch-safe commit은 여기서 도출되는 correctness
requirement다. 세 개를 서로 다른 problem으로 주장하지 않는다.

## 제안: DART-Rx

```text
fixed 4g L1
   |
   | deadline-tagged tensor descriptor
   v
isolated NRx fabric (3g/full GPU, resident replicas)
   |
   | feasible result + slot/epoch
   v
device commit guard ---- late/failed ---> JIT conventional fallback

unused expert slack ---- bounded lease ---> video/speech/Qwen/training
```

구성:

1. request에 deadline, latest fallback start, tensor class, epoch를 붙인다.
2. endpoint별 queue tail과 p99 service/transport로 finish time을 예측한다.
3. remote endpoint credit, conventional fallback window, commit entry를 **함께
   예약할 수 있을 때만** P2P/GDR endpoint에 admit한다.
4. NRx가 늦으면 예약된 conventional graph를 마지막 안전 시점에 실행한다.
5. GPU에서 현재 `(slot, epoch)`에 유효한 NRx/conventional 결과 하나만 atomic
   commit하고 late/stale completion을 폐기한다.
6. background work는 모든 remote 및 fallback reservation 이후의 slack에만 launch한다.

따라서 scheduling unit은 일반 inference job이 아니라 **만료되는 optional result와
mandatory recovery를 함께 가진 transaction**이다. Profile, feasibility, credit,
fallback, commit, lease는 독립 scheme 이름이 아니라 이 transaction의 내부 단계다.

## ISCA용 architecture primitive

**DART-Q**를 GPU command processor/copy-completion path에 추가한다.

- 64/128-entry deadline descriptor queue
- graph latency profile table
- endpoint credit/tail table
- remote endpoint와 local fallback의 dual-reservation unit
- latest-start selector
- background lease gate
- slot/epoch commit state
- P2P/RDMA completion visibility validation과 doorbell-to-graph/commit

현재 A100에서는 persistent CUDA scheduler + CUDA Graph + atomics로 emulate한다. 이후 trace simulator와 small RTL을 만들어 software polling 대비 p99, reserved SM, area/power를 비교한다.

## 지금 결과에서 바뀌는 점

| 기존 | v2 |
|---|---|
| MIG/MPS 성능 비교 | isolation-elasticity contradiction 규명 |
| P2P/GDR latency 감소 | remote endpoint를 위한 data-plane substrate |
| Qwen threshold gating | heterogeneous AI graph-unit lease |
| host shortest queue | deadline feasibility + epoch commit |
| NRx 완료까지 기다림 | JIT conventional fallback |
| 평균 latency | p99.9, miss, consecutive miss, radio utility |
| software router | device prototype + DART-Q hardware model |

## 필수 five-way comparison

| approach | placement | 무엇을 검증하는가 |
|---|---|---|
| MPS | L1+NRx+AI, full GPU 공유 | 최고 work conservation과 interference |
| MIG | L1+NRx 4g, AI 3g | AI 격리 후 local NRx 경합 |
| MIG+MPS | L1+NRx 4g 내부 quota, AI 3g | 계층적 sharing의 tail 제어 |
| P2P | L1 4g, NRx+AI 3g | cross-partition isolation 대 peer-copy 비용 |
| GDR | L1 4g, isolated NRx endpoint | NIC 우회 비용과 remote scale-out |

같은 NRx engine, current payload, open-loop arrival, background offered load,
7g-equivalent GPU budget으로 비교한다. 절대 latency뿐 아니라 각 topology의
L1-only 대비 slowdown도 함께 보고한다.

## 가장 먼저 필요한 killer figure

같은 realistic multi-cell trace의 시간축에 다음 네 개를 겹친다.

- NRx arrivals와 queue depth
- deadline misses
- 각 isolated endpoint utilization
- idle eligible GPU capacity

**queue가 deadline을 놓치는 동시에 다른 eligible accelerator가 idle**이어야 problem이 성립한다. 이 현상이 없으면 논문 방향을 중단한다.

## 그 다음 필요한 cause figure

background kernel의 남은 실행시간과 NRx dispatch delay를 injection별 scatter로 그린다. CUDA high priority를 써도 running kernel은 preempt되지 않으므로, residual time이 tail을 지배하는지 확인한다.

동시에 NRx critical path를 다음으로 분리한다.

```text
pack -> forward transport -> queue wait -> TensorRT kernels
     -> return transport -> commit -> LDPC/CRC
```

현재 예상은 transport보다 NRx compute/queue/control이 크다는 것이지만, 결과로 확인한 뒤 claim한다.

## realistic workload

Qwen만 쓰지 않는다.

- real prompt Qwen prefill/decode: TTFT/TPOT
- actual video decode + pretrained TensorRT detector: frame p99/FPS
- true streaming speech: chunk p99/RTF
- real-data online/FL training: samples/s
- optional RAN-native channel/beam predictor

random tensor, random weight, 30초 강제 padding, closed-loop-only workload는 제외한다.

## baseline

반드시 포함:

- cuPHY only / isolated NRx only
- MPS / MPS priority / MIG / MIG+MPS
- host shared memory / P2P / GDR
- dedicated peak replicas
- shortest queue / predicted-finish host scheduler
- block-on-NRx / immediate dual execution / JIT fallback
- ARCHES-like selected-only/concurrent
- 가능한 REEF/Tally baseline
- oracle

## 1일차 결과물

1. CUDA device graph/conditional/doorbell capability report
2. raw TensorRT/binding/CUDA Graph NRx critical path
3. NRx replica 1/2/4/8 capacity curve
4. Qwen+training residual-kernel vs NRx delay plot
5. C++ host vs device polling control CDF
6. multi-cell same-trace idle+miss killer figure
7. current payload P2P/GDR full-boundary transport breakdown
8. G0/H2 pass-fail decision memo

## 논문 진행 조건

다섯 개가 모두 필요하다.

- realistic trace에서 idle+miss 동시 존재
- stale commit 0, failure 시 bounded fallback
- DART가 strongest software scheduler보다 Pareto 개선
- 최소 세 background class에서 재현
- DART-Q가 polling보다 p99 또는 SM cost를 개선하고 합성 overhead가 작음

radio trace에서 NRx의 BLER/CRC/goodput 이점까지 확인되어야 최종 AI-RAN claim을 강하게 할 수 있다.

## 가장 정직한 paper claim

> DART-Rx adds deadline and validity semantics to remote accelerator execution, allowing a protected PHY partition to borrow resident neural-receiver capacity across isolated GPUs without reconfiguration or unsafe late commits.

## 문서

- novelty/realism pivot와 현재 gate: `DART_RX_NOVELTY_REALISM_PIVOT_KO.md`
- architecture 전체: `ISCA_ARCHITECTURE_V2_KO.md`
- 실험 프로토콜: `ISCA_EXPERIMENT_PLAN_V2_KO.md`
- 77개 run matrix: `ISCA_EXPERIMENT_MATRIX_V2.csv`
- 실행 가능한 scheme algorithm/state: `DART_RX_SCHEME_DESIGN_V0_KO.md`
- Day-1 scheme+experiment co-design: `ISCA_TODAY_ALL_IN_ONE_PLAN_KO.md`
