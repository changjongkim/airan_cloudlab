# MIG–NRx AI-RAN 연구 체크포인트

**작성 시각:** 2026-08-13 KST  
**상태:** 문제 원인판별 실험 진행 중 · 최종 architecture 미확정  
**Canonical 역할:** 현재 연구 문제, 실험 범위, 설계 분기 및 주장 경계를 정의하는 기준 문서

> 이 문서는 DART-Rx 또는 DART-Q를 최종안으로 선언하지 않는다. 지금 단계의 목적은
> MIG 중심 문제의 존재와 원인을 공정한 hardware measurement로 판별하고, 결과가
> 지지하는 mechanism만 다음 설계에 포함하는 것이다.

---

## 0. 지금 무엇을 하고 있는가

연구가 한동안 `deadline transaction`, `utility routing`, `DART-Q` 같은 일반적인
accelerator mechanism으로 확장되면서 원래 질문에서 멀어졌다. 이를 중단하고 다음
순서로 되돌렸다.

1. **MIG가 L1을 실제로 얼마나 보호하는지** 측정한다.
2. **L1과 NRx를 분리했을 때 생기는 통신 비용과 NRx slice 축소 비용**을 분리한다.
3. **NRx replica와 open-loop queue의 안정 경계**를 측정한다.
4. **MPS, MIG, MIG+MPS, P2P, NIC GDR**를 같은 workload contract로 비교한다.
5. multi-cell/selective burst에서 **overloaded NRx와 idle isolated capacity가 실제로
   동시에 존재하는지** 확인한다.
6. 위 결과를 보고서야 scheduler, routing, fallback, background-control mechanism을
   선택한다.

따라서 지금 진행 중인 캠페인은 proposed scheme 평가가 아니라 **problem-discovery 및
causal characterization campaign**이다.

---

## 1. 한 문장 problem statement

> **AI-RAN에서 cuPHY L1과 dependency-coupled Neural Receiver를 같은 GPU에 함께
> 배치하면 MPS의 work conservation은 L1 tail interference를 만들고, MIG의 spatial
> isolation은 L1을 보호하는 대신 NRx capacity와 통신 경로를 partition boundary에
> 고정한다. 실행 중 MIG를 재구성하거나 L1을 중단하지 않고, 어떻게 L1 isolation과
> bursty NRx service의 elasticity 및 background-AI utilization을 동시에 달성할 것인가?**

핵심 키워드는 `MIG 자체`, `RDMA 자체`, `routing 자체`가 아니다. 다음 세 속성이 동시에
존재하는 dependency-coupled real-time pipeline이 문제다.

- **protected stage:** cuPHY L1은 co-tenant 때문에 tail이 흔들리면 안 된다.
- **elastic dependent stage:** NRx는 L1 output을 받아 실행되며 arrival가 cell/channel에
  따라 burst할 수 있다.
- **shared residual capacity:** 통신사업자는 남는 accelerator capacity에 inference,
  training 또는 RAN-native AI를 실행하려 한다.

---

## 2. 시스템 배경과 현실적인 사용 시나리오

### 2.1 하나의 PHY job

하나의 job은 `(cell, slot, scheduled PUSCH allocation)`이다.

```text
slot arrival
  -> common cuPHY front / channel estimation
  -> receiver stage {conventional or NRx}
  -> LDPC / CRC
  -> downstream deadline
```

NRx는 독립적인 batch inference가 아니다. L1 front 결과를 소비하고 LLR를 L1 back에
돌려주는 dependency stage다. 따라서 NRx를 다른 partition/GPU에 놓으면 compute
격리뿐 아니라 request/result data path가 필요하다.

### 2.2 현실적인 arrival

단일 cell은 periodic slot stream을 만든다. 여러 cell을 한 accelerator pool에 수용하면
다음 형태가 생긴다.

- cell별 0.5 ms 또는 1 ms periodic arrival
- synchronized cell의 동시 arrival
- staggered cell의 offset arrival
- channel 상태에 따른 selective NRx invocation
- hard-channel 구간의 연속 burst
- cell 수 증가에 따른 aggregate offered load 증가

NRx가 모든 slot에 반드시 유용하다는 가정은 사용하지 않는다. invocation 확률은
`10/25/50/75/100%`를 sweep하되, performance-only trace와 radio-ground-truth trace를
명확히 분리한다.

### 2.3 background AI

AI-RAN의 배경 가정은 L1만 돌리고 나머지 GPU를 비워두는 것이 아니다. 남는 capacity에는
다음과 같은 workload가 존재할 수 있다.

- LLM prefill/decode
- edge vision inference 또는 training
- streaming speech
- channel/CSI/beam prediction 등 RAN-native AI

NRx와 background AI를 같은 AI partition에서 실행하는 이유는 work conservation이다.
그러나 arbitrary long kernel은 이미 실행되면 stream priority만으로 즉시 preempt되지 않을
수 있으므로, background workload의 kernel/graph quantum과 NRx tail의 관계를 측정해야 한다.

---

## 3. 근본적인 모순

### 3.1 MPS: 유연하지만 L1 isolation이 약함

MPS는 L1, NRx, background AI가 같은 physical GPU 자원을 공유하므로 local pointer와
work conservation을 제공한다. 반면 SM, memory system, launch queue 및 non-preemptible
kernel 간섭이 L1 p99/p99.9에 들어올 수 있다.

MPS percentage는 평균적인 active-thread allocation이지 hard real-time isolation
contract가 아니다. 따라서 낮은 quota만으로 L1 deadline을 보장한다고 가정하지 않는다.

### 3.2 MIG: 격리하지만 capacity를 고정함

MIG는 SM, L2 slice, memory controller/bandwidth domain 등을 공간 분리해 L1 보호에
유리하다. 그러나 topology가 정해지면 각 partition의 compute/memory capacity도 고정된다.

한 NRx partition의 queue가 넘쳐도 다른 partition의 idle capacity를 자동으로 사용할 수
없다. 실행 중 geometry를 바꾸면 resident model/context가 사라지고 L1 drain 또는 service
outage가 발생할 수 있으므로 slot-scale elasticity 수단으로 사용할 수 없다.

### 3.3 same-partition dependency의 대가

`L1 + NRx`를 같은 MIG에 두면 통신은 가장 단순하지만 두 stage가 같은 제한된 SM/HBM
resource를 사용한다. 이때 sibling MIG에 background AI를 격리해도 L1과 NRx 사이의
내부 경합은 남는다.

따라서 다음 두 latency를 분리해야 한다.

```text
L1 active time: L1 자체 kernel이 실제로 소비한 시간
E2E sojourn: arrival부터 NRx, transfer, LDPC/CRC까지 포함한 전체 시간
```

MIG 분리는 NRx 알고리즘 시간을 없애지 않는다. 올바른 기대는 `cuPHY subpath가
L1-only baseline에 가까워지는가`이지 `NRx를 포함한 E2E가 cuPHY-only와 같아지는가`가
아니다.

### 3.4 cross-partition communication의 대가

L1과 NRx를 다른 partition에 두면 request tensor와 result LLR를 전달해야 한다.

- host staging: GPU→CPU DRAM→GPU
- same-process/supported topology P2P
- NIC-loopback GPUDirect RDMA

현재 R580 driver/A100 환경에서는 same-GPU MIG P2P가 실제 benchmark에서 동작했다.
따라서 “MIG 사이 P2P는 항상 불가능하다”를 전제로 삼지 않는다. 다만 cross-process
CUDA IPC/P2P 지원, isolation boundary 및 remote-GPU path는 별도 문제다. NIC GDR의
역할은 P2P보다 무조건 빠른 transport가 아니라 **CPU bounce 없이 process-isolated 또는
remote resident NRx endpoint를 연결하는 fabric**이다.

---

## 4. 문제를 formal하게 정의하기

### 4.1 입력

- protected L1 partition `P_L1`
- 고정된 NRx endpoint 집합 `E={e0,...,en}`
- job `r_i`의 arrival `a_i`
- downstream absolute deadline `d_i`
- endpoint/path별 service distribution `S_e`, `T_path`
- NRx invocation 또는 expected radio utility `u_i`
- background workload의 kernel unit과 utility `b_j`

### 4.2 결정 변수

- L1과 NRx의 placement
- job별 NRx endpoint 선택
- NRx admission 또는 conventional path 선택
- background work의 admission 시점과 quantum

MIG geometry는 measured run 중 결정 변수가 아니다. deployment 전에 고정한다.

### 4.3 목적

우선순위 순으로 다음을 만족해야 한다.

1. protected L1의 tail isolation 유지
2. PHY deadline miss 제한
3. 유효한 NRx service capacity 및 radio benefit 확보
4. background AI utility 최대화
5. idle isolated capacity와 CPU data movement 최소화

### 4.4 안정 조건

단순히 평균 service time이 작다는 것으로 충분하지 않다.

```text
lambda_NRx < aggregate effective service capacity
```

여기에는 다음이 함께 포함돼야 한다.

- common L1 front/back의 직렬 capacity
- endpoint별 NRx service distribution
- transport와 visibility cost
- burst length와 cell synchronization
- background kernel의 blocking interval
- routing imbalance 및 queue state

capacity 근처에서는 평균 latency가 작아 보여도 짧은 overload로 queue p99가 수십~수백
ms로 급증할 수 있다. 따라서 closed-loop throughput만으로 system capacity를 주장하지
않고 open-loop arrival를 사용한다.

---

## 5. 현재까지 확인된 사실과 미확정 사항

### 5.1 확인된 component-level 사실

다음은 기존 측정의 출발점이며 새 fair matrix에서 재검증한다.

- public pyAerial wrapper의 약 100 ms대 수치는 wrapper copy/synchronization을 포함한다.
- caller-owned direct TensorRT/CUDA Graph NRx는 MIG 4g에서 약 1.3 ms 수준이다.
- P2P round trip은 current payload에서 약 76.8 us였다.
- 동일-depth NIC GDR는 P2P 대비 약 0.438 ms 추가 비용이 관찰됐다.
- same 4g에서 L1+NRx를 실행하면 L1 active time slowdown이 관찰됐다.
- cross-partition P2P에서는 L1 slowdown이 훨씬 작았다.
- fixed MIG reconfiguration과 warmed service recovery는 slot timescale보다 매우 길다.

과거 wrapper 기반 39/108 ms 결과와 optimized direct-path 수치를 같은 bar에서 직접
비교하지 않는다.

### 5.2 현재 캠페인의 preliminary signal

4g MIG, one NRx replica, 20초 open-loop preliminary trial에서 measured capacity의
105% 부하가 되자 p99가 약 `0.8 s`, 5 ms miss가 약 `99.6%`까지 증가했다. 같은 4g에서
독립 context 두 개는 해당 offered load에서 p99 약 1.5 ms를 유지했다.

이 값은 campaign 종료 전 수치이므로 최종 result가 아니다. 다만 다음을 보여주는
smoke signal이다.

- queue stability를 측정할 필요가 있다.
- transport microsecond 차이만으로 전체 behavior를 설명할 수 없다.
- replica가 실제 capacity를 늘리는지 topology별로 검증해야 한다.

### 5.3 아직 확인하지 못한 핵심

- 공정한 resource budget에서 어느 placement가 가장 좋은 Pareto frontier를 만드는가?
- 같은 GI의 context 증가가 일반적으로 capacity를 늘리는가, 특정 tactic/concurrency
  artifact인가?
- homogeneous fixed MIG endpoint를 추가하면 capacity가 선형에 가깝게 늘어나는가?
- realistic multi-cell burst에서 `miss + eligible idle MIG`가 동시에 발생하는가?
- NRx와 background AI가 같은 AI MIG를 공유할 때 어떤 kernel이 tail을 만드는가?
- routing만으로 충분한가, admission/background gating 또는 hardware support가 필요한가?
- primary PHY deadline은 실제 gNB/Aerial configuration에서 얼마인가?

---

## 6. 진행 중인 공정한 비교

### 6.1 canonical five-way

| 구성 | L1/NRx placement | isolation | data path | 답하는 질문 |
|---|---|---|---|---|
| MPS | full A100 공유 | 약함 | local/IPC | work conservation의 tail 대가 |
| MIG local | 같은 4g MIG | background와 격리 | local | L1–NRx 내부 경합 |
| MIG+MPS | 같은 4g MIG의 별도 client | background와 격리 | CUDA IPC | quota로 내부 경합 제어 가능성 |
| P2P | 4g L1 / 3g NRx | 강함 | same-GPU P2P | isolation 이득 대 peer transport/작은 NRx slice |
| NIC GDR | 4g L1 / isolated NRx | 강함 | NIC loopback GDR | process/remote endpoint fabric 비용 |

### 6.2 공정성 원칙

- 동일 direct TensorRT engine contract
- 동일 tensor shape, dtype, payload 및 correctness check
- 동일 absolute arrival timestamps와 seed
- open-loop load: reference capacity의 `50/85/95/105%`
- 동일 warm-up과 trial 수
- profiler 없는 measurement run과 profiler attribution run 분리
- absolute latency와 각 topology의 L1-only 대비 slowdown을 함께 보고
- P2P/GDR에서 endpoint 크기가 다르면 transport 효과와 compute-size 효과를 분리

### 6.3 필수 timing breakdown

```text
T_front       cuPHY front / CE
T_pack        registered tensor 준비
T_forward     request transport + visibility
T_queue       NRx endpoint wait
T_nrx         TensorRT enqueue/GPU service
T_backward    result transport + visibility
T_back        LDPC/CRC
T_e2e         전체 sojourn
```

최종 그림은 `T_e2e` 하나만 보여주지 않고 L1 active time, NRx service, queue wait,
transport를 분리한다.

---

## 7. 하루치 campaign 구성

### 7.1 Core causal campaign

결과 root:

```text
/mydata/results/isca_v2/mig_causal_20260813T1138Z
```

실행 순서:

1. NRx capacity: `3g/4g/full × replicas 1/2/4/8`
2. open-loop load: `50/85/95/105%`
3. canonical five-way placement comparison

### 7.2 Real-data workload qualification

- Alpaca 52,002 prompt 기반 Qwen prefill/decode
- CIFAR-10 50,000 samples와 pretrained ResNet-50 asset
- full GPU와 3g MIG에서 duty cycle `0.5/0.9`

이 단계는 workload service/kernels를 qualification하는 것이며 application quality를 새로
평가하는 단계는 아니다.

### 7.3 Background contention

- ResNet-50
- BERT-base
- Whisper-base
- Qwen decode
- naive sharing 대 admission-gated reclaim

ResNet/BERT/Whisper background run은 model/framework kernel characterization이다. 일부는
synthetic input 또는 random weights를 사용하므로 accuracy claim에 쓰지 않는다.

### 7.4 Fixed-MIG multi-cell

측정 시 GPU0/1/2에 고정 `4g+3g` topology를 만들고 3개의 homogeneous 3g NRx endpoint를
사용한다. measured run 중 topology를 바꾸지 않는다. 완료 후 GPU1/2는 full-GPU mode로
복구한다.

- cells: `1/2/4/8`
- slot interval: `0.5/1.0 ms`
- synchronized/staggered arrival
- selective IID/bursty NRx: `10/25/50/75/100%`
- baseline routing: static-one/static-cell/RR/shortest-queue/predicted-finish

이 단계는 actual resident TensorRT NRx compute와 queue를 사용하지만, 모든 endpoint에
cuPHY payload를 GDR로 보내는 최종 integrated service fabric은 아니다. 따라서 compute/
queue problem-existence 결과로만 해석한다.

---

## 8. 가설과 kill criteria

### H1 · L1 isolation

Cross-partition P2P/GDR에서 protected L1 active-time p99가 own-partition L1-only
baseline의 1.05배 이내를 유지하는가?

**Kill/redirect:** L1 slowdown이 transport와 무관하게 크게 남으면 MIG isolation leak,
power/thermal/shared-fabric contention 또는 measurement boundary가 주 문제다.

### H2 · NRx capacity cliff

offered load가 endpoint capacity에 접근하거나 넘을 때 queue p99/deadline miss가 급격히
증가하는가?

**Kill/redirect:** realistic arrival가 단일 endpoint capacity보다 항상 충분히 작다면
elastic NRx pool은 주 contribution이 될 수 없다.

### H3 · fixed-capacity fragmentation

static placement에서 한 endpoint가 deadline을 놓치는 동안 다른 fixed MIG endpoint가
해당 request를 deadline 전에 처리할 수 있을 만큼 idle한가?

**Kill:** `miss + eligible idle capacity`가 realistic trace에서 공존하지 않으면
queue-aware pooling problem은 존재한다고 주장하지 않는다.

### H4 · independent endpoint scale-out

별도 MIG/GPU endpoint를 추가할 때 aggregate NRx service capacity가 증가하는가?

**Kill/redirect:** common L1 front/back 또는 NIC가 먼저 포화되면 NRx routing보다 그
직렬 stage가 주 bottleneck이다.

### H5 · background work conservation

NRx headroom을 침범하지 않는 background admission으로 static reservation 대비 AI utility를
회수하면서 L1/NRx tail을 지킬 수 있는가?

**Kill/redirect:** background kernel blocking이 너무 길어 gating으로도 tail을 지킬 수
없다면 model/kernel decomposition 또는 hard spatial dedication이 필요하다.

### H6 · GDR의 역할

GDR data path 비용이 NRx compute/queue보다 작고 CPU staging을 제거하면서 remote fixed
endpoint 활용을 가능하게 하는가?

**Kill/redirect:** GDR visibility/transport tail이 deadline budget을 지배하면 remote
pooling 대신 sibling P2P 또는 local dedicated NRx만 현실적이다.

---

## 9. 결과에 따른 architecture 분기

현재는 어떤 branch도 최종 선택하지 않는다.

### Branch A · queue/fragmentation이 지배

조건:

- transport는 작음
- cross-partition L1은 보호됨
- static miss와 eligible idle endpoint가 공존
- endpoint 추가로 capacity 증가

후속 설계:

```text
fixed protected L1 MIG
  + resident fixed NRx MIG pool
  + P2P/GDR service fabric
  + capacity-aware routing/admission
  + NRx-priority background gating
```

이 경우 novelty 후보는 **dependency-coupled real-time stage를 fixed isolated endpoints
위에서 drain-free하게 elastic하게 만드는 architecture**다.

### Branch B · same-partition interference가 지배하나 remote pooling은 불필요

조건:

- split placement로 L1 tail은 개선
- realistic NRx load는 단일 endpoint로 충분
- idle-fragmentation 문제가 없음

후속 설계는 단순 dedicated split-MIG + P2P/GDR pipeline이다. 이는 유용한 engineering
result지만 ISCA-level architecture novelty는 약하므로 kernel/data-path optimization 또는
더 근본적인 interference-control mechanism이 필요하다.

### Branch C · background kernel blocking이 지배

조건:

- NRx capacity는 충분
- co-tenant가 AI-MIG queue/service tail을 크게 악화

후속 설계는 bounded GPU work unit, grant-ahead admission, graph segmentation 또는 command-
processor scheduling support 중심이 된다. 이때만 hardware primitive의 필요성을 논증한다.

### Branch D · common PHY stage가 지배

조건:

- NRx replica를 늘려도 end-to-end throughput이 증가하지 않음
- cuPHY front/back이 먼저 포화

후속 설계는 NRx pool이 아니라 cell/slot-level PHY pipeline parallelism, common-stage
replication 또는 operator fusion 문제로 전환한다.

---

## 10. 현재 단계에서 주장하지 않을 것

- MIG가 MPS보다 좋다는 단순 결론
- P2P 또는 GDR 자체가 novelty라는 주장
- NIC GDR가 P2P보다 항상 빠르다는 주장
- MIG geometry를 slot 단위로 동적 변경한다는 주장
- 현재 5 ms experimental threshold가 실제 production PHY deadline이라는 주장
- synthetic selective trace가 실제 radio utility를 증명한다는 주장
- 동일 MIG 안의 TensorRT context 수를 독립 hardware replica로 일반화
- compute-only multi-cell 결과를 full cuPHY/GDR integrated 결과로 표현
- DART-Q 또는 새로운 ISA가 이미 필요하다고 가정

---

## 11. ISCA 수준으로 가기 위한 증거 사슬

ISCA-level paper는 다음 순서가 모두 연결되어야 한다.

1. **Problem existence:** realistic multi-cell trace에서 isolation-induced capacity
   fragmentation을 실제 hardware로 보여준다.
2. **Root-cause attribution:** CUDA API, kernel, copy, queue, synchronization, background
   blocking을 Nsight Systems/Compute로 분해한다.
3. **Architecture gap:** 기존 CUDA priority/MPS/MIG/P2P/GDR 조합만으로 해결되지 않는
   구체적인 dispatch/visibility/preemption/commit gap을 보인다.
4. **Minimal mechanism:** 측정된 gap에 필요한 최소 mechanism만 설계한다.
5. **Prototype:** 실제 Aerial/cuPHY/TensorRT/GDR chain에서 구현한다.
6. **End-to-end evaluation:** L1 tail, deadline miss, NRx/radio utility, background utility,
   utilization 및 overhead를 함께 평가한다.

1–2번 결과 없이 3–4번을 먼저 설계하면 mechanism이 문제보다 앞서게 된다. 현재 연구는
1–2번을 완료하는 단계다.

---

## 12. 체크포인트

### C0 · 환경 및 component path

- [x] A100 MIG 4g+3g topology
- [x] ConnectX-6 Dx loopback RC QP
- [x] CPU RDMA 및 GPUDirect MR
- [x] direct TensorRT/caller-owned binding/CUDA Graph
- [x] P2P/GDR component measurement

### C1 · causal characterization · 완료

- [x] 4g capacity smoke test
- [x] 3g/4g/full × replicas 1/2/4/8, 3 trials
- [x] MPS/MIG/MIG+MPS/P2P/GDR fair matrix, 5 placements × 8 rates × 3 trials
- [x] L1-front/L1-back/L1-active/E2E/sojourn timing boundary 검증

### C2 · realistic workload qualification · 완료

- [x] Alpaca/CIFAR-10/pretrained ResNet asset 준비
- [x] Qwen prefill/decode full/3g qualification, 24/24 pass
- [x] CIFAR-10 ResNet50 training full/3g qualification, 24/24 pass
- [x] ResNet50/BERT/Whisper/Qwen-decode cooperative-unit blocking 측정
- [ ] Nsight Systems/Compute 기반 개별 kernel·copy·sync root-cause attribution

### C3 · problem existence · 완료

- [x] 물리적으로 독립된 homogeneous 3g MIG endpoint 3개
- [x] 1/2/4/8-cell periodic arrival
- [x] synchronized/staggered arrival
- [x] selective IID/bursty 10–100%
- [x] miss + eligible idle capacity 동시성: analyzer `problem_gate=True`

### C4 · 결과 기반 설계 선택 · 일부 완료

- [x] Branch 판정: 주 원인은 A(fixed-capacity fragmentation), background co-run 시 C(blocking)
- [x] 최소 mechanism 방향: fixed NRx pool + tail-aware admission/routing/fallback + cooperative background reclaim
- [ ] full cuPHY↔multi-endpoint GDR integrated implementation
- [ ] outlier-aware final scheduler와 실제 conventional fallback 실행
- [ ] production deadline 및 radio-derived selective trace/BLER/CRC validation

---

## 13. 현재 실행 상태와 재현 위치

실행 root:

```text
/mydata/results/isca_v2/mig_causal_20260813T1138Z
```

runner:

```text
/mydata/aerial-cuda-accelerated-ran/pyaerial/isca_v2/run_nrx_capacity_full.sh
/mydata/aerial-cuda-accelerated-ran/pyaerial/isca_v2/run_fiveway_compute_full.sh
/mydata/aerial-cuda-accelerated-ran/pyaerial/isca_v2/run_mig_causal_all_day_followup.sh
```

2026-08-14 KST 최종 상태:

- capacity: 3g/4g/full × replicas 1/2/4/8 × 3 trials 완료
- fair five-way: 5 placements × 8 absolute rates × 3 trials = 120/120 완료
- workload qualification: Qwen 24 + training 24 = 48/48 완료
- background contention: 4 workloads × 2 policies = 8/8 완료
- multi-cell/selective: 87 traces × 5 policies = 435 policy runs 완료
- multi-cell analyzer: `PASS files=87 problem_gate=True`
- 원격 결과 크기: 약 850 MB, 1,608 files
- 실패 흔적 없이 모든 stage `COMPLETE`; 측정 후 환경 복구 완료
- GPU0: 원래 4g+3g MIG topology, GPU1/2/3: MIG disabled/full GPU
- `mlx5_0`: `PORT_ACTIVE`, RoCE v2 GID 3 및 loopback IPv4 복구
- generic DART policy와 DART-Q 실험은 계속 동결

---

## 14. 완료 결과와 판정

### 14.1 단일 NRx endpoint의 queue cliff

`replicas=1`의 saturation rate를 기준으로 같은 topology에서 95%와 105% offered load를
재측정했다.

| topology | measured saturation | 95% load p99 | 105% load p99 | 105% miss ratio |
|---|---:|---:|---:|---:|
| 3g | 621.7 req/s | 1.79 ms | 834.25 ms | 0.9961 |
| 4g | 739.7 req/s | 1.53 ms | 803.00 ms | 0.9955 |
| full A100 | 1164.1 req/s | 1.01 ms | 643.00 ms | 약 0.996 |

동일 MIG GI 안에 TensorRT context를 2/4/8개로 늘려도 capacity가 scale-out하지 않았다.
오히려 105% load에서 tail이 더 악화됐다. 즉 같은 GI의 여러 context는 독립 NRx
accelerator가 아니며, 수평 확장은 물리적으로 독립된 MIG/GPU endpoint가 필요하다.

### 14.2 MIG/MPS/P2P/GDR fair comparison

아래는 3 trials의 median이다. `last stable`은 offered throughput을 유지하면서 queue가
폭발하지 않은 마지막 측정점이다.

| placement/transport | low-load E2E | last stable | 다음 관측점의 sojourn p99 | 해석 |
|---|---:|---:|---:|---|
| MPS, full GPU | 4.242 ms @50/s | ≥350/s | 2.551 ms @350/s | 단독 compute는 가장 빠르나 isolation 없음 |
| MIG-local | 4.678 ms @50/s | 300/s | 1124.981 ms @350/s | 4g 고정 capacity cliff |
| MIG+MPS | 4.979 ms @50/s | 250/s | 259.106 ms @300/s | 같은 GI context 증가는 scale-out 아님 |
| P2P | 5.269 ms @50/s | 250/s | 451.798 ms @300/s | transport는 작지만 remote service capacity가 지배 |
| zero-copy GDR | 6.109 ms @50/s | 180/s | 1048.695 ms @250/s | 현재 구현은 single-slot speedup이 아니며 더 이른 cliff |

P2P payload transport는 대표적으로 약 78–110 us였지만 E2E는 수 ms이고 overload tail은
수백–수천 ms였다. 따라서 **통신 자체가 주 병목이라는 가설은 기각**한다. GDR은 latency
optimizer가 아니라 격리된 endpoint를 접근 가능하게 만드는 enabling path로만 취급한다.

### 14.3 realistic background contention

trace는 `500 req/s 2.01 s → 1100 req/s 3 s burst → 500 req/s 3 s`이고, primary 4g와
spare 3g는 측정 중 고정했다. adaptive reclaim은 NRx burst 동안 cooperative work-unit
경계에서 background 실행을 억제한다.

| background | naive burst p99 / miss@5ms | adaptive burst p99 / miss@5ms | adaptive/naive background units |
|---|---:|---:|---:|
| ResNet50 inference | 2211.394 ms / 0.6700 | 6.723 ms / 0.0124 | 521/554 = 94.0% |
| BERT-base inference | 1602.273 ms / 0.5727 | 2.772 ms / 0.0039 | 1524/1706 = 89.3% |
| Whisper-base inference | 471.312 ms / 0.4964 | 3.070 ms / 0.0039 | 1909/1936 = 98.6% |
| Qwen-7B decode | 2270.974 ms / 0.6767 | 5.724 ms / 0.0112 | 278/299 = 93.0% |

이는 단순 Qwen-only 현상이 아니다. resident model을 유지하고 MIG topology를 바꾸지 않은
상태에서, 7–11%의 background work 감소로 queue collapse를 제거할 수 있었다. 다만 현재
정책은 연구용 prototype이며 CUDA kernel preemption을 구현한 것은 아니다.

### 14.4 multi-cell/selective NRx problem gate

3개의 물리적으로 독립된 3g MIG endpoint에서 87개 trace를 실행했다. 모든 값은 3 trials
median이며 `no timely NRx`는 현재 실험의 5 ms threshold 기준이다.

| scenario | policy | p99 | no timely NRx | fallback | idle endpoint fraction |
|---|---|---:|---:|---:|---:|
| 1 cell, 1 ms, NRx 100% | static-one | 3293.252 ms | 0.9997 | 0 | 0.667 |
| 1 cell, 1 ms, NRx 100% | predicted-finish | 1.632 ms | 0.0013 | 0.0011 | 0.465 |
| 4 cell, 1 ms, bursty NRx 10% | static-one | 51.392 ms | 0.6485 | 0 | 0.794 |
| 4 cell, 1 ms, bursty NRx 10% | predicted-finish | 5.501 ms | 0.0161 | 0.0025 | 0.781 |
| 4 cell, 0.5 ms, bursty NRx 10% | static-one | 3332.836 ms | 0.9990 | 0 | 0.668 |
| 4 cell, 0.5 ms, bursty NRx 10% | predicted-finish | 5.065 ms | 0.0789 | 0.0518 | 0.590 |

static placement에서는 한 endpoint가 사실상 전부 deadline을 놓치는 동안 다른 두 endpoint가
66.7% idle인 상태가 반복됐다. 반면 predicted-finish routing/admission은 같은 고정 topology로
queue collapse를 크게 줄였다. `problem_gate=True`이므로 fixed-capacity fragmentation의
존재는 실제 hardware에서 확인됐다.

동시에 현재 predicted-finish도 완성된 답은 아니다. 예를 들어 2-cell, 1 ms staggered
trace는 rare NRx service outlier 때문에 median p99 8.185 ms와 no-timely ratio 0.3604를
보였다. scheduler lateness p99는 0.4 us였으므로 단순 host replay artifact로 설명되지 않는다.
최종 scheduler에는 mean-only prediction이 아니라 runtime completion feedback과 tail/outlier
guard가 필요하다.

### 14.5 결과 기반 architecture 방향

한 문장 판정은 다음과 같다.

> **관측된 핵심 문제는 고정 MIG가 만든 NRx capacity fragmentation(A)이며, background AI가
> 같은 NRx partition을 사용할 때 cooperative-unit blocking(C)이 이를 증폭한다. P2P/GDR
> transport latency나 동일 GI 안의 context 수 증가는 이 문제를 해결하지 못한다.**

따라서 다음 prototype의 최소 구성은 다음으로 제한한다.

1. L1을 재시작하지 않는 고정 4g protected partition
2. 여러 고정 3g MIG/GPU에 resident한 NRx endpoint pool
3. measured completion과 tail bound를 사용하는 predicted-finish routing/admission
4. deadline 안에 완료 불가능하면 conventional receiver로 즉시 fallback
5. NRx headroom이 있을 때만 cooperative boundary에서 background work를 실행
6. P2P/GDR은 endpoint reachability를 위한 transport로 사용하고 novelty로 주장하지 않음

### 14.6 아직 남은 증거

- multi-cell 실험은 actual TensorRT compute/queue지만 cuPHY front/back과 payload transport가
  빠진 compute-only gate다.
- selective trace는 IID/Markov sensitivity input이며 실제 radio ground truth가 아니다.
- conventional fallback은 정책상 account만 했고 실제 실행/CRC/BLER utility는 측정하지 않았다.
- 5 ms는 experimental threshold이며 production PHY deadline으로 주장하지 않는다.
- full Aerial/cuPHY↔multi-endpoint GDR chain, radio-derived trace, BLER/CRC utility 및 Nsight
  kernel/copy/sync attribution을 끝내야 최종 architecture claim을 확정할 수 있다.

즉 이번 캠페인은 problem existence와 설계 방향을 지지하지만, 논문 전체 mechanism의 최종
검증을 끝낸 것은 아니다.
