# DART-Rx: ISCA 방향의 AI-RAN accelerator architecture

> **동결된 exploratory design:** 현재 연구의 authoritative checkpoint는
> `MIG_NRX_RESEARCH_CHECKPOINT_KO.md`다. 아래 DART-Q/ISA 설계는 causal experiment가
> architecture gap을 입증할 때만 재검토하며, 현재 구현 목표로 간주하지 않는다.

작성일: 2026-08-13  
상태: v2 architecture specification  
대상 플랫폼: NVIDIA A100 MIG + Aerial cuPHY + TensorRT NRx + ConnectX-6 Dx

> **Canonical novelty 문서:** `DART_RX_CANONICAL_NOVELTY_THESIS_KO.md`  
> 본 문서는 구현 상세 specification으로 유지한다. 논문의 문제 정의, novelty claim,
> dual reservation과 real-time 보장 용어가 충돌하면 canonical 문서를 우선한다.

## 0. 결정

`MIG 대 MPS 대 MIG+MPS 대 P2P 대 GDR` 비교는 논문의 **필수 첫 번째
evaluation 축**으로 둔다. 다만 비교표 자체를 novelty로 끝내지 않고, 각
approach가 만드는 contention, isolation, fragmentation, transport 비용을
분해하여 다음 architecture problem으로 연결한다.

> **고정 격리로 보호된 real-time PHY가, 재구성 없이 격리 밖의 optional NRx
> capacity를 빌리면서도 deadline과 output validity를 어떻게 보존할 것인가?**

Isolation-elasticity contradiction은 관찰되는 systems 현상이다. Completion-oriented
GPU/NIC interface에 expiry, recovery reservation, single-winner validity 의미가 없다는
것은 단순 remote routing으로 그 현상을 안전하게 해결하지 못하는 architectural root
cause다. Expiring result, dual reservation, epoch commit은 별도 문제가 아니라 여기서
도출되는 correctness requirement다.

제안하는 시스템은 **DART-Rx: Deadline-Aware Remote Tensor execution for Neural Receivers**다.

DART-Rx의 핵심은 새로운 배치 정책 하나가 아니다. 다음 네 가지를 하나의 accelerator command abstraction으로 결합한다.

1. **Deadline-tagged remote tensor request**
2. **Predicted-finish admission과 endpoint credit**
3. **Device-resident result commit 및 just-in-time conventional fallback**
4. **Deadline-safe background lease**

MIG, MPS, MIG+MPS, P2P, GPUDirect RDMA는 모두 본문에서 동등하게 비교한다.
이들은 DART-Rx의 필요성을 증명하는 substrate와 baseline이며, 그 자체만을
novelty로 주장하지 않는다.

## 1. 왜 지금 설계가 부족한가

현재 prototype은 이미 중요한 사실을 확인했다.

- fixed MIG는 L1을 background Qwen으로부터 거의 완전히 보호한다.
- current direct NRx는 약 1.34 ms이고 wrapper 100 ms대 수치는 Python/public-wrapper 복사와 동기화 artifact다.
- 같은 GPU의 MIG 간 P2P round trip은 current QPSK payload에서 약 76.8 us다.
- full chain은 P2P 약 5.89 ms, GDR staging 약 6.25--6.40 ms다.
- fixed 4g NRx capacity는 약 745 slots/s이고, 750--800 slots/s 부근에서 queue cliff가 나타난다.
- warmed MIG reconfiguration조차 평균 약 3.64 s outage를 만든다.

따라서 transport 하나만 줄이는 것으로는 문제를 해결할 수 없다. transport는 이미 NRx compute보다 작다. 실제 남은 문제는 다음 세 가지다.

### 1.0 다섯 approach 비교가 먼저 필요한 이유

다음 비교는 생략하지 않는다.

| approach | L1 | NRx | background AI | data path | 드러내는 질문 |
|---|---|---|---|---|---|
| MPS | full GPU shared | 같은 full GPU | 같은 full GPU | local pointer | work conservation이 isolation 없이 가능한가? |
| MIG | 4g의 L1+NRx | L1과 같은 4g | 별도 3g | local pointer | background 격리 후에도 L1-NRx 내부 경합이 남는가? |
| MIG+MPS | 4g의 L1+NRx | 같은 4g, MPS quota | 별도 3g | local pointer | partition 내부 quota가 tail을 제어하는가? |
| P2P | 전용 4g | sibling 3g | NRx pool과 3g 공유 | CUDA P2P | L1을 완전히 분리했을 때 peer copy 비용보다 isolation 이득이 큰가? |
| GDR | 전용 4g | sibling/remote endpoint | NRx pool과 공유 | NIC loopback GDR | P2P가 없거나 remote GPU일 때 isolation을 유지할 수 있는가? |

추가로 host shared memory와 CPU-buffer RDMA를 negative transport baseline으로,
DART-P2P/DART-GDR를 제안 방식으로 둔다.

공정성은 두 층으로 확보한다.

1. **Deployment-level:** 한 A100의 총 7g-equivalent budget, 동일 NRx model,
   동일 open-loop arrival, 동일 background offered load를 사용한다.
2. **Mechanism-level:** 가능한 경우 동일 `4g+3g` endpoint와 동일 payload에서
   P2P/GDR/host transport만 바꾸고, topology가 다르면 각 topology의 L1-only
   baseline 대비 slowdown도 함께 보고한다.

따라서 절대 latency와 normalized slowdown을 둘 다 보고해야 한다. P2P가 더
큰 NRx slice를 얻어서 이긴 결과나, MPS가 더 큰 full GPU를 써서 이긴 결과를
단순 transport 효과라고 해석하지 않는다.

### 1.1 Spatial isolation과 work conservation의 충돌

MIG는 interference를 막지만 capacity를 고정한다. 한 endpoint queue가 burst로 넘칠 때 다른 GPU/MIG에 idle compute가 있어도, 현재 PHY pipeline은 이를 deadline-safe하게 빌려 쓰지 못한다.

### 1.2 Completion-oriented GPU API와 deadline-oriented PHY의 충돌

CUDA stream priority는 scheduling hint일 뿐이며, 이미 실행 중인 긴 low-priority kernel을 preempt하지 않는다. `cudaGraphLaunch`와 RDMA WRITE도 결국 “언젠가 완료될 work”를 제출한다. 하지만 slot pipeline에 필요한 의미는 다음과 같다.

- 이 요청은 `D` 전까지만 가치가 있다.
- 예측 완료가 늦으면 시작하지 말아야 한다.
- late NRx result는 절대 다음 slot buffer에 commit되면 안 된다.
- NRx가 늦으면 마지막 안전 시점에 conventional receiver를 실행해야 한다.

현재 API에는 이 네 의미가 없다.

### 1.3 Host control loop의 jitter와 stale-result 위험

host polling/router는 feasibility 판단, graph submit, completion polling, fallback submit, output selection을 여러 API boundary로 나눈다. 평균이 작아도 p99 control delay와 epoch 오류가 hard-real-time path에 들어간다.

## 2. DART-Rx가 푸는 정확한 문제

고정 topology에서 다음 입력이 주어진다.

- 보호된 L1 partition `P_L1`
- 하나 이상의 resident NRx endpoint `E={e0...en}`
- endpoint와 함께 실행할 best-effort AI workload
- slot request `r_i`, arrival `a_i`, hard deadline `d_i`
- NRx를 쓸 때의 radio utility `u_i`
- conventional receiver latency bound `C_conv^99`

목표는 MIG를 바꾸는 것이 아니다.

1. L1 partition을 재시작하지 않는다.
2. deadline을 만족할 수 있는 NRx request만 remote endpoint에 admit한다.
3. NRx가 늦거나 실패해도 conventional path가 deadline 전에 commit되게 한다.
4. 남은 endpoint slack만 background AI에 빌려준다.
5. NRx utility와 background throughput을 최대화하면서 deadline miss를 제한한다.

이는 generic GPU utilization 문제가 아니라 **optional accelerator result를 포함한 hard-real-time transaction** 문제다.

## 3. 시스템 구조

```text
                protected partition                         elastic expert fabric
        +--------------------------------+        +--------------------------------------+
slot --> cuPHY CE --> DART descriptor ----P2P/GDR--> DART-Q --> resident NRx replicas   |
        |                  |             |        |     ^          |                    |
        |                  |             |        |     | credit   | result + epoch     |
        |        latest-safe-start       |        | background lease controller         |
        |                  v             |        +--------------------------------------+
        |          conventional graph    |                         |
        |                  \              |<-------- P2P/GDR -------+
        |                   DART commit guard --> LDPC/CRC
        +--------------------------------+
```

물리 배치는 세 형태를 지원한다.

- sibling MIG: 4g L1 + 3g expert pool, same-GPU P2P
- remote GPU: L1 MIG + full-GPU expert pool, P2P가 가능하면 P2P
- isolated fabric: L1 MIG + remote GPU/MIG, GPUDirect RDMA

NRx model과 CUDA Graph는 endpoint에 상주한다. request마다 model load나 MIG reconfiguration은 없다.

## 4. DART descriptor

각 request는 최소 다음 필드를 가진 64-byte aligned descriptor다.

```c
struct dart_desc {
    uint64_t slot_id;
    uint32_t epoch;
    uint16_t graph_id;
    uint16_t tensor_class;
    uint64_t release_gpu_clock;
    uint64_t deadline_gpu_clock;
    uint64_t latest_fallback_start;
    uint32_t input_slot;
    uint32_t output_slot;
    uint16_t candidate_bitmap;
    uint8_t  utility_class;
    uint8_t  flags;
    uint32_t checksum;
};
```

핵심은 pointer를 매번 넘기는 것이 아니라 사전에 등록된 tensor slot과 graph handle을 참조하는 것이다. payload memory는 startup 때 P2P/GDR 등록하고, control descriptor만 ring으로 이동한다.

`tensor_class`는 current QPSK payload뿐 아니라 MCS/PRB/layer 조합별 크기와 calibrated latency profile을 식별한다.

## 5. 실행 의미와 invariant

DART-Rx는 다음을 correctness invariant로 둔다.

### I1. Deadline feasibility

endpoint `e`에 대한 예측 완료 시점은 다음과 같다.

```text
F_e(r) = max(now, tail_e)
       + T_forward^99(tensor_class, e)
       + T_nrx^99(graph_id, tensor_class, e)
       + T_return^99(tensor_class, e)
       + G_e
```

`F_e(r) <= deadline - commit_guard`일 때만 admit한다. 평균 latency는 admission에 사용하지 않는다.

### I2. At-most-one commit

각 output slot은 `EMPTY -> NRX_COMMITTED` 또는 `EMPTY -> CONV_COMMITTED`로 한 번만 전이한다. atomic compare-and-swap으로 winner를 정한다.

### I3. Epoch safety

completion의 `(slot_id, epoch)`가 현재 output slot과 일치하지 않으면 즉시 drop한다. 늦게 온 이전 request가 재사용된 buffer를 오염시키지 못한다.

### I4. Bounded fallback

NRx result가 `latest_fallback_start`까지 valid 상태가 아니면 conventional graph를 실행한다. fallback을 request arrival 때부터 중복 실행하지 않고, 필요한 마지막 안전 시점까지 미룬다.

### I5. Bounded best-effort blocking

background graph unit `b`는 다음 조건에서만 dispatch한다.

```text
T_b^99 + drain_guard < earliest_admitted_nrx_latest_start - now
```

이미 실행 중인 arbitrary kernel을 preempt한다고 가정하지 않는다. background model은 측정된 non-preemptible quantum을 가진 graph unit으로 나누거나, 안전 window가 없으면 dispatch하지 않는다.

## 6. DART-Q: 제안하는 architecture primitive

### 6.1 왜 software scheduler만으로 끝내지 않는가

software-only persistent polling은 구현 가능하지만 다음 비용이 남는다.

- 한 CTA/SM 또는 polling traffic의 상시 비용
- NIC/P2P completion에서 graph launch까지의 jitter
- host가 graph handle pool과 queue state를 갱신하는 경로
- late completion과 fallback의 여러 stream 간 ordering 복잡도

ISCA-level claim은 이 gap을 정량화한 뒤에만 성립한다.

### 6.2 DART-Q microarchitecture

DART-Q는 GPU command processor와 copy/network completion path 사이의 작은 queue extension이다.

구성 요소:

- **Slot Request Table (SRT)**: 64 또는 128 entry descriptor SRAM
- **Graph Profile Table (GPT)**: `(graph_id, tensor_class)`별 conservative execution bound
- **Endpoint Credit Table (ECT)**: queue tail, free tensor slots, health epoch
- **Latest-Start Selector (LSS)**: 가장 이른 `latest_start` request 선택
- **Lease Gate (LG)**: best-effort graph unit의 dispatch 가능 여부 검사
- **Commit Status Table (CST)**: slot/epoch/winner 상태
- **Completion Doorbell**: P2P/RDMA immediate가 CPU 없이 queue/commit event를 생성

DART-Q는 SM datapath나 tensor core를 바꾸지 않는다. 핵심은 **work를 빨리 preempt하는 것**보다 **늦을 work를 launch하지 않고, remote completion을 deadline-aware graph/commit으로 연결하는 것**이다.

### 6.3 명령

최소 ISA/API abstraction은 다음 네 개다.

```text
DART_SUBMIT(desc)             // deadline-bearing expert request
DART_COMPLETE(slot,epoch)     // remote payload completion
DART_LEASE(graph,budget)      // bounded best-effort graph unit
DART_COMMIT(slot,epoch,kind)  // NRx or conventional atomic winner
```

실제 prototype에서는 GPU-visible rings와 atomics로 emulate한다. hardware study에서는 doorbell-to-launch와 commit guard를 command processor에 둔다.

### 6.4 기존 기능을 활용하는 부분

- CUDA Graph는 resident NRx/fallback/background unit의 launch overhead를 줄인다.
- device graph launch와 conditional graph node는 host round trip 없이 control flow를 표현할 수 있는지 capability gate를 수행한다.
- DOCA GPUNetIO/RDMA GPU API가 설치 가능한 플랫폼에서는 GPU-initiated control/data path baseline으로 사용한다.
- 현재 CloudLab에서는 pyverbs P2P/GDR data path + persistent device polling을 먼저 사용한다.

CUDA 기능이 DART semantics를 완전히 제공한다고 주장하지 않는다. 특히 stream priority는 이미 실행 중인 work를 preempt하지 않고, graph handle은 동시 재사용 제약이 있어 request window만큼 exec-handle pool이 필요하다.

## 7. Software prototype

### 7.1 단계 S0: host DART

현재 router를 고쳐 descriptor, p99 profile, endpoint credit, epoch commit, JIT fallback을 구현한다. 이는 기능 기준선이지 최종 contribution이 아니다.

### 7.2 단계 S1: device-resident DART

각 partition에 작은 persistent scheduler kernel을 둔다.

- producer가 GPU-visible descriptor ring에 publish
- endpoint kernel이 credit/admission을 확인
- 사전 upload한 NRx graph exec handle을 launch
- completion record에 epoch와 finish timestamp 기록
- protected partition의 commit kernel이 NRx/fallback winner를 선택

CUDA 13 device graph launch가 A100/현재 driver에서 실제 동작하는지는 가장 먼저 probe한다. 불가능하면 CUDA Graph host launch + GPU-side commit을 사용하고, host launch 시간을 별도 upper-bound로 남긴다.

### 7.3 단계 S2: DART-Q model

실제 Nsys/CUPTI trace를 입력으로 하는 cycle/event simulator를 만든다.

- observed kernel and DMA durations replay
- SRT depth, profile error, doorbell latency, graph launch latency sweep
- software persistent polling 결과로 simulator calibration
- 64/128-entry RTL queue와 selector를 SystemVerilog로 구현
- synthesis로 frequency, area, power를 보고

hardware 결과가 software 대비 의미 있는 tail/SM 절감이 없으면 architecture claim을 접고 systems venue로 pivot한다.

## 8. Routing과 fallback policy

policy는 architecture 위에 올라가는 단순하고 설명 가능한 형태로 제한한다.

### 8.1 Utility-feasible routing

1. radio utility가 threshold 아래면 conventional만 실행한다.
2. 각 endpoint의 conservative `F_e(r)`를 계산한다.
3. feasible endpoint 중 `utility / incremental_gpu_time`이 가장 큰 곳을 고른다.
4. feasible endpoint가 없으면 conventional로 간다.

학습 기반 policy를 첫 contribution으로 두지 않는다. oracle과의 gap이 큰 경우에만 후속 predictor를 넣는다.

### 8.2 JIT fallback

```text
fallback_latest_start = deadline - C_conv^99 - commit_guard
```

NRx가 그 전에 끝나면 NRx를 commit한다. 끝나지 않으면 conventional graph를 launch한다. 늦은 NRx는 drop한다.

즉 ARCHES처럼 “어떤 expert가 radio condition에 좋은가”를 결정하는 부분과 경쟁하지 않는다. ARCHES-style utility signal을 DART descriptor가 입력으로 받을 수 있고, DART는 **선택된 optional expert가 isolated accelerator를 건너 deadline-safe하게 실행되는가**를 책임진다.

## 9. Background AI는 무엇을 쓸 것인가

Qwen 하나로는 workload-independent claim을 할 수 없다. 다음 네 class를 필수로 둔다.

| class | 실제 입력/arrival | unit of lease | application metric |
|---|---|---|---|
| Generative AI | real prompt trace, prefill+decode 분리 | layer group 또는 decode step | TTFT, TPOT, token/s |
| Video analytics | 실제 압축 video decode + TensorRT detector | frame 또는 engine segment | frame deadline, FPS, drop |
| Streaming speech | 실제 audio chunk arrival, padding 없는 streaming model | audio chunk/encoder block | chunk p99, real-time factor |
| Online/FL training | real dataset micro-batch + optimizer | micro-batch | samples/s, round time |

선택적 fifth workload로 RAN-native beam/channel predictor를 추가한다.

모든 workload는 다음 조건을 만족해야 한다.

- random input만 사용하지 않는다.
- closed-loop 최대속도만 보고하지 않는다.
- 실제 arrival trace 또는 명시된 server load를 쓴다.
- model weights, dataset subset, preprocessing, batch, precision을 manifest에 고정한다.
- background throughput뿐 아니라 자체 SLO degradation도 보고한다.

## 10. 기존 연구와의 경계

### ARCHES

ARCHES는 Aerial/OAI PHY 안에서 radio condition에 맞는 AI/conventional expert를 slot boundary에 switch한다. DART-Rx는 expert 선택 정책 자체를 claim하지 않는다. 격리된 accelerator service의 deadline admission, remote tensor movement, JIT fallback, epoch-safe commit을 claim한다.

### Flex-MIG

Flex-MIG는 fixed MIG의 one-to-many 일반 job 실행과 host shared-memory collective로 makespan/fragmentation을 개선한다. DART-Rx는 hard PHY deadline, GPU-direct data path, resident NRx service, stale-result suppression을 다룬다.

### REEF/Tally

REEF와 Tally는 shared GPU에서 preemption 또는 thread-block scheduling으로 foreground isolation을 높인다. DART-Rx는 protected partition을 유지한 채 remote endpoint를 수평 확장하고 optional result를 deadline-safe하게 commit한다. 이들을 가능한 범위에서 same-partition baseline으로 둔다.

### HAF/일반 AI-RAN scheduler

HAF 같은 작업은 placement와 resource allocation policy가 중심이다. DART-Q는 그 policy가 사용할 수 있는 microsecond data/command-plane primitive다.

## 11. 논문 hypothesis

### H1. Existence

realistic multi-cell arrival에서 fixed isolation은 동시에 `idle GPU-seconds`와 `deadline miss`를 만든다. 이 둘이 같은 trace에 나타나야 한다.

### H2. Architectural gap

host scheduling과 stream priority의 p99 blocking/jitter는 transport보다 크거나 deadline guard를 불가능하게 한다.

### H3. Mechanism

DART device commit/admission은 static MIG와 host router보다 deadline miss와 consecutive miss를 줄이면서 background service를 유지한다.

### H4. Generality

효과가 Qwen에만 국한되지 않고 vision, speech, training에서도 유지된다.

### H5. Hardware value

DART-Q는 persistent polling SM cost와 doorbell-to-launch tail을 줄이며, 작은 queue state로 software prototype보다 나은 deadline/utilization frontier를 만든다.

## 12. 논문의 예상 contribution

1. 실제 Aerial/NRx에서 spatial isolation과 dynamic optional-AI demand 사이의 **isolation-elasticity gap**을 CUDA/API/kernel/queue 수준으로 규명
2. deadline, feasibility, epoch, fallback을 묶은 **DART request/commit abstraction**
3. P2P/GDR completion과 CUDA Graph를 잇는 **DART-Q command-processor extension** 및 software emulation
4. real multi-cell trace와 heterogeneous AI-on-RAN workload에서 MPS, MIG, MIG+MPS, P2P, GDR, host/device DART를 비교한 hardware evaluation

## 13. 반드시 통과해야 하는 kill test

다음 중 하나라도 실패하면 현재 ISCA framing을 중단한다.

1. realistic trace에서 stranded capacity와 deadline miss가 동시에 관측되지 않는다.
2. device-resident control이 host DART보다 p99 또는 resource cost에서 유의미하게 낫지 않다.
3. DART가 shortest-queue + conservative admission보다 이기지 못한다.
4. radio-aware NRx selection이 BLER/CRC/goodput에서 conventional 대비 실질적 utility가 없다.
5. heterogeneous workload 두 개 이상에서 같은 conclusion이 재현되지 않는다.

## 14. Claim 문장

좋은 claim:

> DART-Rx adds deadline and validity semantics to remote accelerator execution, allowing a protected PHY partition to borrow resident neural-receiver capacity across isolated GPUs without reconfiguration or unsafe late commits.

피해야 할 claim:

- MIG가 MPS보다 빠르다.
- GPUDirect가 host copy보다 빠르다.
- remote NRx routing은 처음이다.
- expert switching은 처음이다.
- dynamic MIG 없이 fixed MIG를 활용하는 것은 처음이다.

## 15. 근거 문헌과 공식 기능

- NVIDIA Aerial은 RAN과 AI application의 general-purpose multi-tenancy를 명시한다: https://docs.nvidia.com/aerial/cuda-accelerated-ran/latest/index.html
- CUDA stream priority는 hint이며 running work를 preempt하지 않는다: https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/asynchronous-execution.html
- CUDA device graph launch와 conditional graph는 device-side dynamic control flow의 구현 후보지만 제약이 있다: https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html
- DOCA GPUNetIO는 GPU-managed queue와 GPU semaphore/RDMA primitive를 제공한다: https://docs.nvidia.com/doca/archive/doca-v2-5-1/DOCA%2BGPUNetIO/index.html
- ARCHES는 real-time expert switching의 가장 가까운 AI-RAN work다: https://arxiv.org/abs/2604.23397
- Flex-MIG는 fixed MIG 위 distributed execution의 가장 가까운 work다: https://arxiv.org/abs/2511.09143
- REEF는 microsecond GPU preemption baseline이다: https://www.usenix.org/conference/osdi22/presentation/han
- Tally는 thread-block-level non-intrusive sharing baseline이다: https://arxiv.org/abs/2410.07381
- Clockwork는 predictable DNN execution/SLO scheduling의 기준점이다: https://arxiv.org/abs/2006.02464
- AI-RAN HAF는 high-level deadline-aware placement/resource sharing의 인접 work다: https://arxiv.org/abs/2605.07547
