# DART-Rx canonical novelty thesis

작성일: 2026-08-13  
상태: **논문 설계와 novelty 판단의 canonical 문서**  
대상: ISCA급 architecture paper  
시스템 이름: **DART-Rx**  
하드웨어 primitive 이름: **DART-Q**

> 이 문서는 `MIG/MPS/P2P/GDR 비교`를 논문 novelty로 오해하지 않도록 연구의
> 문제, 새로운 architectural contract, 설계 이유, 기존 연구와의 경계, 논문 claim,
> 필수 증거를 한곳에 고정한다. 다른 계획 문서와 표현이 충돌하면 이 문서를 우선한다.

## 1. 최종 판정

현재 연구는 다음처럼 쓰면 novelty가 약하다.

> MIG로 L1과 NRx를 격리하고 P2P 또는 GPUDirect RDMA로 데이터를 전송한다.

MIG isolation, 고정 MIG의 fragmentation, remote GPU execution, RDMA pooling,
deadline-aware inference scheduling, AI/conventional expert selection은 각각 기존 연구가
이미 다룬 문제다. 이들을 한 시스템에 모은 것만으로는 ISCA novelty가 되기 어렵다.

논문의 새로운 대상은 **GPU 배치 방식이 아니라 accelerator command의 의미**다.

> **DART-Rx는 결과가 PHY deadline에서 만료되는 optional accelerator request를
> 정의하고, remote AI 실행 자원과 local recovery 자원을 함께 예약하며, remote 또는
> fallback 결과 중 현재 slot/epoch에 유효한 하나만 commit하는 validity-scoped
> accelerator transaction을 제공한다.**

이를 실행하는 **DART-Q**는 GPU command processor와 P2P/NIC completion path 사이에서
deadline timer, remote endpoint credit, fallback reservation, epoch validation,
single-winner commit을 처리한다.

따라서 논문의 novelty를 다음과 같이 분리한다.

| 항목 | 논문에서의 역할 | novelty 여부 |
|---|---|---|
| MIG/MPS/MIG+MPS | isolation과 interference baseline | 아님 |
| P2P/GDR/host copy | isolated endpoint transport | 아님 |
| predicted-finish routing | feasibility policy | 단독으로는 아님 |
| radio-aware NRx 선택 | utility input | 단독으로는 아님 |
| DART transaction | expiring result와 recovery를 갖는 accelerator contract | **핵심 novelty** |
| DART-Q | dual reservation, timer-driven fallback, validity commit의 hardware realization | **ISCA architecture novelty** |
| Aerial/A100 구현 | 실제 PHY에서 문제와 효과 검증 | empirical contribution |

## 2. 근본 문제

### 2.0 논문 전체가 답할 하나의 질문

> **고정 격리로 보호된 real-time PHY가, 재구성 없이 격리 밖의 optional NRx
> capacity를 빌리면서도 deadline과 output validity를 어떻게 보존할 것인가?**

이 한 문장이 논문의 유일한 problem statement다. 문서에서 보이는 세 층은 서로 다른
문제가 아니라 다음 인과사슬이다.

```text
[observable systems symptom]
static isolation + dynamic multi-cell demand
    -> overloaded NRx queue와 eligible idle capacity가 공존

[why ordinary remote routing is insufficient]
NRx는 optional이고 결과가 만료되며 conventional output은 반드시 필요
    -> remote credit만 잡으면 late result와 correlated fallback storm이 발생

[root architectural cause]
GPU/NIC command는 completion은 표현하지만 expiry, recovery reservation,
single-winner validity는 표현하지 않음

[proposed architectural response]
DART transaction + DART-Q
    -> remote execution과 recovery를 함께 예약하고 timely valid result 하나만 commit
```

따라서 `isolation-elasticity contradiction`은 관찰되는 현상이고, `command semantic
gap`은 그 현상을 안전한 remote borrowing으로 해결하지 못하게 하는 근본 원인이며,
`expiry/recovery/commit`은 그 원인으로부터 도출되는 correctness requirement다.
Layer 1을 Layer 2/3과 경쟁하는 별도 problem으로 주장하지 않는다.

### 2.0.1 왜 지금 가능한 동시에 필요한가

근거는 막연한 6G 상용화 예측이 아니라 실제 software/hardware convergence에 둔다.

- NVIDIA Aerial 25.3은 commercial-grade AI-native 5G/6G gNB SDK와 general-purpose
  multi-tenancy를 명시하고, multi-cell configuration을 제공한다.
- pyAerial은 TensorRT 기반 PUSCH neural receiver를 conventional receiver와 비교하는
  예제를 제공한다. 즉 optional AI PHY stage는 더 이상 순수한 가상 workload가 아니다.
- IMT-2030은 2024--2027년에 requirement/evaluation 기준을 만드는 단계이며 상용 시점을
  2027년으로 주장하지 않는다. 6G 상용화 예측은 본 논문의 urgency 근거로 사용하지 않는다.
- GPU spatial isolation, resident AI graph, P2P/GDR가 동시에 가능해졌지만 이를 하나의
  deadline/validity transaction으로 묶는 command abstraction은 제공되지 않는다.

공식 근거:

- Aerial 25.3: <https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-3/index.html>
- pyAerial NRx example: <https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-3/pyaerial/examples.html>
- Aerial multi-cell test vectors:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-3/release_notes/supported_test_vector_configurations.html>
- ITU IMT-2030 timeline:
  <https://www.itu.int/en/itu-r/study-groups/rsg5/rwp5d/imt-2030/pages/default.aspx>

### 2.0.2 누가 겪고 무엇이 나빠지는가

대상은 AI-RAN runtime 설계자, GPU accelerator/command-processor 설계자, 그리고
deadline-critical PHY와 opportunistic AI를 통합하는 edge-system 운영자다.

문제를 해결하지 않으면 다음 중 하나를 선택해야 한다.

- MPS로 함께 실행해 protected L1 tail interference를 허용한다.
- MIG endpoint를 peak demand 기준으로 overprovision하고 idle capacity를 남긴다.
- remote NRx를 무조건 보내고 late result, fallback burst, stale output 위험을 감수한다.
- conventional과 NRx를 항상 중복 실행해 isolation 이득을 duplicate work로 소모한다.
- burst 때 MIG를 재구성하여 seconds-scale drain/reload를 감수한다.

### 2.1 시스템 상황

AI-RAN GPU에는 다음 자원이 있다.

- deadline-critical cuPHY/L1 partition
- 하나 이상의 resident NRx endpoint
- conventional receiver graph
- NRx pool의 남는 자원을 사용하려는 LLM, vision, speech, training workload
- P2P 또는 GPUDirect RDMA로 연결되는 pre-registered tensor slots

여러 cell의 slot이 들어오고 NRx는 channel condition에 따라 선택적으로 호출된다.
따라서 request arrival은 평균적으로 낮더라도 cell aggregation, 동일 timing boundary,
hard-channel episode 때문에 burst할 수 있다.

### 2.2 기존 선택지의 구조적 한계

#### MPS: elastic하지만 L1을 보호하지 못한다

같은 GPU에서 L1, NRx, background AI를 work-conserving하게 실행할 수 있지만, 긴 kernel,
memory bandwidth, launch queue interference가 L1 tail에 들어온다.

#### MIG: L1을 보호하지만 NRx capacity가 고정된다

MIG는 spatial isolation을 제공하지만 request demand는 partition boundary를 따르지 않는다.
한 NRx queue가 deadline을 놓치는 동안 다른 isolated endpoint가 idle일 수 있다.

#### Dynamic MIG: demand timescale과 맞지 않는다

MIG reconfiguration은 resident model, CUDA context, graph, registered buffer를 다시 만들게
하며 slot/burst timescale에 사용할 수 없다. DART-Rx는 fast path에서 MIG topology를
절대 변경하지 않는다.

#### 단순 remote routing: PHY correctness를 보장하지 못한다

Shortest queue나 predicted finish로 remote NRx를 선택해도 다음 질문은 남는다.

- remote NRx가 늦으면 conventional receiver를 언제 시작하는가?
- 여러 remote request가 동시에 지연될 때 fallback storm을 어디서 실행하는가?
- fallback이 먼저 끝난 뒤 도착한 NRx result를 누가 폐기하는가?
- 재사용된 output slot에 이전 request가 쓰지 못하게 어떻게 막는가?
- NIC/P2P completion과 tensor visibility, graph launch, commit의 ordering은 누가 보장하는가?

기존 GPU/NIC command는 work completion을 알리지만, 결과의 **유효기간과 대체 경로의
correctness**를 표현하지 않는다.

### 2.3 정확한 research problem

각 request는 다음과 같다.

```text
r = (release, deadline, latest_fallback_start, slot, epoch,
     tensor_class, utility_class, candidate_endpoints)
```

각 endpoint `e`에는 queue tail, tensor credit, health epoch, transport 및 NRx service
bound가 있다. 시스템은 request마다 conventional, local NRx, remote NRx 중 하나를
선택하고, 선택한 remote NRx가 늦을 경우 recovery를 실행해야 한다.

목표는 다음과 같다.

```text
maximize  timely delivered radio utility + lambda * background utility

subject to
  Pr(PHY completion > deadline) <= epsilon
  at-most-one committed result per (slot, epoch)
  zero stale/wrong-epoch commits
  reserved recovery capacity
  no fast-path MIG reconfiguration or protected-L1 restart
```

외부 PHY deadline은 hard하지만 현재 service estimate는 p99 측정값이다. 따라서 실제
보장은 **firm/probabilistic real-time**으로 표현한다. hard real-time을 주장하려면
모든 path의 WCET와 완전한 temporal reservation을 추가로 입증해야 한다.

## 3. 새로운 architectural contract: DART transaction

### 3.1 기존 inference request와 무엇이 다른가

일반 inference request는 완료되면 결과를 반환하고 deadline을 넘기면 SLO violation으로
기록한다. DART request는 결과의 의미가 다르다.

1. NRx result는 deadline 이후에는 **사용하면 안 되는 expired result**다.
2. NRx가 실패해도 PHY pipeline은 멈추지 않고 conventional path로 복구해야 한다.
3. remote NRx와 conventional fallback이 모두 실행될 수 있지만 결과는 하나만 공개한다.
4. output buffer가 다음 slot에 재사용되므로 completion generation을 검증해야 한다.
5. remote path를 선택할 때 recovery capacity도 함께 보존해야 한다.

즉 DART는 단순 job scheduling이 아니라 **speculative remote execution과 mandatory
recovery를 하나의 accelerator transaction으로 만든다.**

### 3.2 핵심 동작: dual reservation, single commit

Remote NRx를 admit할 때 DART-Q는 원자적으로 다음 세 자원을 확보한다.

1. remote endpoint queue/tensor credit
2. protected side의 conventional fallback window 또는 fallback credit
3. `(slot, epoch)` commit-table entry

세 자원 중 하나라도 확보할 수 없으면 remote NRx를 실행하지 않고 conventional path로
간다. 이 **remote service와 recovery service의 동시 예약**이 correlated NRx delay 때
발생하는 fallback storm을 막는다.

실행 중에는 다음 규칙을 적용한다.

```text
remote completion before expiry
    -> validate DMA visibility and (slot, epoch)
    -> CAS commit NRX
    -> release fallback reservation

latest_fallback_start reached while transaction is open
    -> launch reserved conventional graph

conventional completion
    -> CAS commit CONVENTIONAL

late, stale, duplicate, or losing completion
    -> discard without exposing its private result buffer
```

### 3.3 상태 기계

```text
                    remote accepted
  OPEN --------------------------------------> NRX_RUNNING
    |                                               |
    | latest-safe-start                             | timely + valid
    v                                               v
  FALLBACK_RUNNING ----------------------------> NRX_COMMITTED
    |                         first valid winner     ^
    | timely + valid                                |
    v                                               |
  CONV_COMMITTED <----------------------------------+

  any non-winning late/stale completion -> EXPIRED/DROPPED
```

Remote NRx와 fallback이 겹친 경우 먼저 **완료한 것**이 아니라, deadline 안에 끝나고
현재 epoch이며 memory visibility를 만족한 결과 중 commit CAS를 먼저 성공한 것이
winner다.

### 3.4 correctness invariant

- **Feasibility:** conservative finish가 deadline 안인 remote request만 admit한다.
- **Recovery reservation:** admitted remote request는 fallback capacity를 가진다.
- **At-most-one commit:** output은 NRx 또는 conventional 중 한 번만 공개된다.
- **Epoch safety:** `(slot, epoch)`가 다르면 payload가 있어도 commit하지 않는다.
- **Visibility ordering:** payload DMA visibility가 completion/graph launch보다 선행한다.
- **Expiry:** deadline 이후 completion은 결과 buffer를 오염시키지 못한다.
- **Bounded background blocking:** background unit은 다음 reserved work 이전에 끝날 수 있을
  때만 dispatch한다.

## 4. DART-Rx overall architecture

```text
                         protected PHY partition
 +--------------------------------------------------------------------+
 | CE -> DART transaction front-end -> commit guard -> LDPC/CRC/output |
 |             |                         ^                             |
 |             | latest-safe-start       | valid winner                |
 |             +-> reserved conventional fallback                     |
 +-------------|-------------------------|-----------------------------+
               | descriptor + tensor     | result + slot/epoch
               v                         |
        P2P / GPUDirect RDMA transport adapters
               |                         ^
 +-------------v-------------------------|-----------------------------+
 |             isolated resident NRx service fabric                    |
 | endpoint credits | graph handles | tensor rings | bounded BG lease  |
 +--------------------------------------------------------------------+
```

논문의 architecture는 세 subsection으로 설명한다.

### 4.1 Transaction front-end

- radio utility 입력 수신
- endpoint feasibility 계산
- remote/fallback/commit 동시 예약
- deadline 및 latest-fallback timer 설정

### 4.2 Resident expert fabric

- NRx engine과 graph를 endpoint에 상주시킴
- pre-registered input/output ring 사용
- local/P2P/GDR를 동일 transaction API 아래 제공
- 여러 cell의 요청을 resident replica로 수평 확장

### 4.3 Recovery, commit, and slack reclamation

- JIT conventional fallback
- epoch-safe single-winner commit
- 다음 grant와 reservation 이후의 slack만 background에 lease
- arbitrary background kernel preemption을 가정하지 않음

## 5. DART-Q microarchitecture

DART-Q는 SM/Tensor Core datapath를 바꾸는 accelerator가 아니다. GPU command processor와
copy/network completion path 사이의 작은 queue extension이다.

| block | 역할 |
|---|---|
| Slot Request Table | open DART transaction과 deadline 저장 |
| Endpoint Credit Table | queue tail, free tensor slot, health epoch 저장 |
| Service Profile Table | tensor/transport/graph별 conservative bound 저장 |
| Dual Reservation Unit | remote endpoint와 local fallback window 동시 확보 |
| Latest-Start Timer | fallback graph launch 시점 감시 |
| Commit Status Table | slot/epoch/winner 상태와 CAS commit 제공 |
| Completion Doorbell | P2P/RDMA completion을 CPU 없이 transaction event로 변환 |
| Lease Gate | bounded background graph unit만 dispatch |

최소 command abstraction은 다음과 같다.

```text
DART_SUBMIT(desc)
DART_RESERVE(remote_endpoint, fallback_window)
DART_COMPLETE(slot, epoch, result_slot, visibility_token)
DART_FALLBACK(slot, epoch)
DART_COMMIT(slot, epoch, winner)
DART_LEASE(graph, bounded_budget)
```

Clockwork-style scheduler는 예정된 inference action을 deadline에 맞춰 실행한다. DART-Q는
그보다 강한 다음 동작을 한다.

- remote와 recovery 두 실행 경로를 동시에 reserve
- time-triggered recovery graph launch
- DMA completion과 GPU visibility를 검증한 commit
- expired/stale remote output의 architectural suppression
- background issue를 future recovery reservation과 연동

이 다섯 동작이 DART-Q의 novelty다.

## 6. Pain point에서 mechanism이 도출되는 과정

| pain point | 필요한 property | DART mechanism | 선택하지 않은 대안 |
|---|---|---|---|
| L1 interference와 static capacity mismatch | isolation을 유지한 endpoint borrowing | fixed protected partition + resident NRx fabric + P2P/GDR adapter | MPS-only 또는 burst마다 MIG reconfiguration |
| NRx utility가 channel마다 다름 | 불필요한 optional work 억제 | utility-gated invocation | 모든 slot에 무조건 NRx |
| 평균은 작아도 queue/transport tail이 존재 | 늦을 work를 launch 전에 거절 | conservative predicted-finish admission | mean-latency shortest queue |
| remote NRx가 늦어도 PHY output은 필요 | deadline 전 mandatory recovery | JIT conventional fallback | NRx가 끝날 때까지 대기 |
| 여러 remote request가 함께 지연될 수 있음 | correlated fallback도 실행 가능한 capacity | remote endpoint + fallback window dual reservation | remote credit만 예약 |
| NRx와 fallback이 동시에 끝날 수 있음 | output을 한 번만 공개 | private result slots + atomic single-winner commit | shared output에 직접 write |
| asynchronous completion과 ring reuse | 이전 slot 결과의 격리 | `(slot, epoch)` validation | sequence 없는 ready flag |
| NIC/P2P completion과 GPU consumer 사이 | DMA visibility와 launch/commit ordering | visibility token/fence + completion doorbell | host가 여러 API를 polling |
| completion-only command interface | expiry/recovery/commit을 한 명령 contract로 표현 | DART transaction + DART-Q | 독립 CUDA/RDMA call의 host orchestration |
| background kernel은 즉시 preempt되지 않음 | future reserved work의 blocking bound | bounded background lease | 무제한 background enqueue |

이 표에서 특히 다음 오해를 피한다.

- Dual reservation은 단순히 `optional + isolation`을 푸는 것이 아니라 **correlated
  remote delay가 local fallback overload로 전파되는 문제**를 푼다.
- Epoch commit은 deadline 자체를 푸는 것이 아니라 **비동기 completion과 buffer reuse의
  correctness**를 푼다.
- DART-Q는 interface gap이 있다는 이유만으로 자동으로 필요하지 않다. Host/device
  software의 control tail 또는 polling resource가 실질적 병목임을 G8에서 보여야 한다.

## 7. 기존 연구와 novelty 경계

### 7.1 가장 가까운 연구

| 연구 | 이미 해결한 것 | DART-Rx가 추가하는 architectural semantic |
|---|---|---|
| Flex-MIG | fixed MIG one-to-many execution, host shared-memory collective, fragmentation 완화 | expiring PHY result, remote+recovery reservation, epoch commit 없음 |
| MIGRator | accuracy/SLO 기반 dynamic MIG reconfiguration | DART는 fast path reconfiguration 없이 resident service를 사용 |
| HAF | AI-RAN의 deadline-aware placement와 resource allocation | microsecond command/data plane과 result-validity transaction 없음 |
| ARCHES | slot-boundary AI/conventional expert switching과 safe output selection | isolated remote queue/transport admission, reserved JIT recovery, asynchronous epoch completion 없음 |
| Clockwork | predictable inference와 deadline action scheduling | optional result expiry, mandatory alternative, dual reservation, single-winner recovery 없음 |
| PREMA | predictive scheduling과 preemptible NPU | remote completion/expiry/fallback transaction 없음 |
| REEF/Tally | shared GPU에서 RT와 best-effort 간섭 제어 | protected partition을 유지한 remote capacity borrowing과 result validity 없음 |
| Planaria | accelerator의 dynamic spatial fission | fixed commodity partition 위 transaction primitive가 아님 |
| Prism | RDMA 기반 GPU disaggregation과 SLO-aware communication | slot-scale optional result, local recovery reservation, epoch-safe commit 없음 |
| INFaaS/SuperServe | accuracy-latency model 선택과 burst 대응 | radio-validity 및 PHY recovery semantics 없음 |

### 7.2 reviewer가 제기할 가장 강한 반론

> DART-Rx는 Clockwork admission, ARCHES expert switch, Prism RDMA pooling,
> Flex-MIG fixed-partition execution을 조합한 시스템이다.

이에 대한 답은 “AI-RAN에 적용했다”가 아니다.

> 기존 시스템의 scheduling unit은 완료해야 할 job이다. DART의 scheduling unit은
> **만료되는 speculative result와 반드시 보존해야 하는 recovery path를 함께 가진
> transaction**이다. DART-Q는 두 resource domain을 동시에 reserve하고, remote DMA와
> local graph completion을 하나의 epoch-safe commit protocol로 종결한다.

이 차이를 구현과 invariant로 보여주지 못하면 novelty claim은 성립하지 않는다.

### 7.3 논문에서 주장하지 않을 것

- MIG가 MPS보다 빠르다는 사실
- GPUDirect가 host copy보다 빠르다는 사실
- remote GPU에서 NRx를 실행하는 것 자체
- deadline-aware shortest queue 자체
- radio condition에 따라 expert를 고르는 것 자체
- 고정 MIG에서 여러 workload를 실행하는 것 자체

## 8. 왜 AI-RAN이 이 architecture를 필요로 하는가

DART transaction은 generic inference에 임의로 추가한 기능이 아니다. PHY가 다음 조건을
동시에 갖기 때문에 필요하다.

- 반복되는 slot deadline
- multi-cell synchronization에 의한 correlated arrivals
- channel condition에 따른 optional AI utility
- 항상 존재하는 conventional recovery path
- late result가 useless한 것을 넘어 다음 slot correctness를 해칠 수 있음
- UL grant/slot calendar를 통한 near-future demand lookahead
- L1은 중단할 수 없지만 NRx replica는 isolated endpoint에 분산 가능

Aerial은 multi-cell PUSCH 실행을 지원하므로 2/4/8-cell aggregated stream은 단순 cloud
inference trace보다 이 문제를 직접 드러내는 workload다.

## 9. 현재 측정 결과가 말하는 것

현재 결과는 “GDR가 single-slot latency를 극적으로 줄인다”는 가설을 지지하지 않는다.

- CPU-buffer RDMA 1 MiB steady WRITE: 약 107 us
- GDR forward 1,415,232 B: 약 758 us
- GDR backward 1,257,984 B: 약 665 us
- GDR full pipeline 평균: 약 109.6 ms
- SHM 대비 GDR pipeline 개선: 약 1.2 ms

따라서 GDR의 논문상 역할은 **latency trick이 아니라 elasticity enabler**다. protected
L1 밖의 resident NRx replica를 CPU staging 없이 호출할 수 있게 한다.

현재 direct NRx가 약 1.34 ms이고 과거 100 ms대 결과가 wrapper artifact라는 측정은
중요하다. NRx가 최적화될수록 transport와 control tail의 비중은 다시 커진다. 논문은
raw enqueue, wrapper, copy, synchronization, graph launch, transport, queueing을 각각
분해해야 한다.

## 10. ISCA claim을 성립시키는 필수 증거

### 10.1 Problem existence

Problem에는 구조적 존재와 실질적 중요성 두 단계가 있다.

구조적으로는 fixed endpoint의 service capacity보다 특정 window의 admitted arrival가 크고
다른 eligible endpoint에 capacity가 남으면 queueing과 fragmentation이 동시에 생긴다.
이는 service curve와 arrival curve로 보일 수 있다. 그러나 논문 significance를 위해서는
그 조건이 **plausible multi-cell/selective NRx workload에서 실제로 발생**해야 한다.

G4 및 연계 실험에서는 같은 시간축에 다음이 나타나야 한다.

- static endpoint의 NRx deadline miss
- 다른 eligible endpoint의 idle capacity
- dynamic MIG outage가 burst duration보다 훨씬 큼

여기서 `다른 eligible endpoint`는 동일 queue가 순간적으로 idle했다는 뜻이 아니다.
해당 request의 tensor/graph를 실행할 수 있는 별도 sibling MIG, 다른 GPU 또는 remote
resident endpoint를 뜻한다. 하나의 NRx endpoint만 둔 실험은 elasticity opportunity를
증명할 수 없다.

한 synthetic trace의 G4 결과만으로 problem을 확정하지 않는다. standards-derived
periodic/staggered cell schedule, selective invocation rate, burst correlation을 sweep하고
측정된 NRx service curve와 함께 재현한다. 이 realistic envelope 전체에서 borrowing
opportunity가 없다면 실질적 problem significance가 부족한 것으로 판정한다.

### 10.2 Mechanism benefit

DART-Rx는 다음 baseline보다 좋은 deadline/utility/utilization Pareto front를 보여야 한다.

- same-partition MPS
- static MIG
- MIG+MPS
- static P2P/GDR routing
- shortest queue
- Clockwork-style predicted-finish admission
- immediate conventional+NRx dual execution
- block-on-NRx 후 fallback

### 10.3 Correctness

- late completion injection
- duplicate completion
- wrong epoch
- endpoint failure/restart
- delayed DMA visibility
- correlated multi-endpoint slowdown
- fallback storm

위 조건에서 wrong/stale commit은 0이어야 한다.

### 10.4 Hardware necessity

Host DART, persistent CUDA DART, DART-Q를 비교한다.

- descriptor-observe-to-launch p50/p95/p99/p99.9
- completion-to-commit tail
- fallback timer error
- reserved SM 및 memory traffic
- CPU utilization와 polling cost
- deadline miss와 consecutive miss
- DART-Q area, power, frequency, queue depth sensitivity

Host와 device software가 이미 충분하다면 DART-Q는 ISCA contribution이 아니다. 이 경우
정직하게 systems paper로 전환한다.

### 10.5 End-to-end radio value

- 같은 TB/channel realization을 conventional과 NRx 양쪽에 입력
- MCS/channel/EsNo별 paired CRC/BLER
- delivered radio utility는 deadline 전에 commit된 결과만 집계
- NRx가 이기지 않는 bin은 conventional로 route
- 1/2/4/8-cell, selective 10/25/50/75/100%, IID와 burst 모두 실행

## 11. 논문의 canonical claim

### 한 문장 problem statement

> Fixed accelerator isolation protects a real-time PHY but strands neural-receiver
> capacity under bursty multi-cell demand, while existing remote-execution interfaces
> cannot safely use optional results that expire and require a conventional recovery path.

### 한 문장 solution statement

> DART-Rx turns optional neural reception into a validity-scoped accelerator
> transaction that co-reserves remote execution and local recovery, then commits
> exactly one timely result across isolated accelerators without reconfiguration.

### 한 문장 architecture statement

> DART-Q extends the accelerator command/completion path with dual reservation,
> deadline-triggered fallback, DMA-visibility validation, and epoch-safe single-winner
> commit for expiring remote results.

### 예상 contribution

1. 실제 Aerial/NRx에서 **isolation-elasticity-validity gap**을 kernel, transport,
   queue, radio utility 수준으로 규명한다.
2. expiring optional result와 mandatory recovery를 묶는 **DART transaction**을 정의하고
   correctness invariant를 제공한다.
3. dual reservation과 timer/doorbell/commit을 구현하는 **DART-Q microarchitecture** 및
   software emulation을 제안한다.
4. A100 MIG/Aerial에서 MPS, MIG, MIG+MPS, P2P, CPU-RDMA, GDR와 host/device DART를
   realistic multi-cell 및 heterogeneous background workload로 평가한다.

## 12. Novelty kill conditions

다음 중 하나라도 성립하면 현재 claim을 수정하거나 venue를 변경한다.

1. plausible multi-cell/selective workload envelope 어디에서도 deadline miss와 별도
   eligible endpoint의 idle capacity가 동시에 나타나지 않는다.
2. optimized NRx가 realistic deadline/replica budget 안에 들어오지 않는다.
3. paired radio sweep에서 양의 NRx utility 영역이 없다.
4. recovery reservation 없이도 모든 overload/failure 조건에서 동일한 결과가 나온다.
5. DART가 Clockwork-style admission + shortest queue + fallback baseline을 이기지 못한다.
6. device/DART-Q가 host implementation의 tail 또는 resource cost를 유의미하게 줄이지 못한다.
7. integrated P2P/GDR path에서 epoch/visibility/commit correctness를 입증하지 못한다.

Kill condition을 미리 등록하는 이유는 novelty를 약하게 만드는 것이 아니다. 어떤 결과가
나와도 novel이라고 주장하는 것을 막고, DART-Q가 실제 필요한 architecture임을 강하게
증명하기 위해서다.

## 13. 논문 구성

```text
1. Introduction
2. Motivation and Characterization
   2.1 Isolation vs. elasticity
   2.2 Optional NRx utility and deadline
   2.3 Why scheduling/transport alone is insufficient
3. DART Transaction Semantics
   3.1 Dual reservation
   3.2 Expiry, fallback, and single commit
   3.3 Correctness invariants
4. DART-Q Architecture
   4.1 Queue/table organization
   4.2 Completion and timer pipeline
   4.3 Background lease and resource cost
5. DART-Rx System Integration
   5.1 Aerial and resident NRx fabric
   5.2 P2P/GDR adapters
   5.3 Software/device prototypes
6. Evaluation
7. Related Work
8. Conclusion
```

## 14. 관련 연구

- Flex-MIG: <https://arxiv.org/abs/2511.09143>
- MIGRator: <https://arxiv.org/abs/2407.13126>
- HAF AI-RAN: <https://arxiv.org/abs/2605.07547>
- ARCHES: <https://arxiv.org/abs/2604.23397>
- Clockwork, OSDI 2020: <https://www.usenix.org/conference/osdi20/presentation/gujarati>
- REEF, OSDI 2022: <https://www.usenix.org/conference/osdi22/presentation/han>
- Tally, ASPLOS 2025: <https://arxiv.org/abs/2410.07381>
- PREMA, HPCA 2020: <https://doi.org/10.1109/HPCA47549.2020.00027>
- Planaria, MICRO 2020: <https://research.nvidia.com/publication/2020-07_planaria-dynamic-architecture-fission-spatial-multi-tenant-acceleration-deep>
- Prism, NSDI 2025: <https://www.usenix.org/conference/nsdi25/presentation/yang>
- INFaaS, ATC 2021: <https://www.usenix.org/conference/atc21/presentation/romero>
- SuperServe, NSDI 2025: <https://www.usenix.org/conference/nsdi25/presentation/khare>
- NVIDIA Aerial multi-cell PUSCH:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/content/notebooks/datalake_pusch_multicell.html>
- NVIDIA Aerial 25.3 overview:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-3/index.html>
- NVIDIA pyAerial neural receiver example:
  <https://docs.nvidia.com/aerial/cuda-accelerated-ran/25-3/pyaerial/examples.html>
- ITU IMT-2030 timeline:
  <https://www.itu.int/en/itu-r/study-groups/rsg5/rwp5d/imt-2030/pages/default.aspx>

## 15. 현재 구현 산출물과 정확한 한계

장시간 GPU campaign과 독립적으로 다음 control-plane 골격을 구현했다.

| 파일 | 현재 제공하는 것 | 아직 제공하지 않는 것 |
|---|---|---|
| `cloudlab_aerial/task1/isca_v2/dart_runtime.py` | endpoint/tensor credit, fallback interval calendar, dual reservation, JIT fallback, visibility/health-epoch validation, single-winner commit, bounded lease | 실제 CUDA/P2P/GDR/Aerial graph dispatch |
| `test_dart_runtime.py` | deterministic unit test, 10,000 fault injection, correlated 8-way fallback storm | hardware timing/correctness |
| `dart_transaction_trace.py` | G7/G8 공통 `dart-transaction-input-v1` schema | 자동 Nsys/CUPTI ingest |
| `build_dart_q_trace.py` | actual NRx open-loop artifact를 공통 trace로 변환 | transport 및 paired radio utility 결합 |
| `dart_q_simulator.py` | 동일 trace의 Host/Device/DART-Q control-path what-if replay | calibrated DART-Q performance claim |
| `run_dart_offline_tests.sh` | 전체 offline correctness suite | integrated G7/G8 hardware gate |

현재 offline 검증 결과:

- deterministic unit tests: 12/12 pass
- randomized fault injection: 10,000건 pass
- stale, duplicate, late, invisible payload, endpoint restart accepted: 모두 0
- fallback storm: 100 groups × 8 simultaneous requests, fallback capacity 2에서
  over-admission 0, reservation leak 0
- trace converter tests: 3/3 pass
- DART-Q simulator tests: 6/6 pass

재현 명령:

```bash
cloudlab_aerial/task1/isca_v2/run_dart_offline_tests.sh /tmp/dart_rx_offline
```

이 결과는 **transaction semantics의 software correctness**만 검증한다. G7 완료라고
부르려면 실제 L1→P2P/GDR→NRx→reserved conventional fallback→commit pipeline에
연결해야 한다. G8 완료라고 부르려면 simulator의 `illustrative-placeholder` profile을
Nsys/CUPTI 측정으로 교체하고 persistent-device baseline, RTL/cycle model, area/power를
평가해야 한다.
