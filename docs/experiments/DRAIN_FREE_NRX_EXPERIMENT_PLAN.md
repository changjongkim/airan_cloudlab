# Drain-Free Elastic NRx 실험 설계

> **상태 주의:** 현재 문제 정의와 실행 체크포인트는
> `../current/MIG_NRX_RESEARCH_CHECKPOINT_KO.md`가 authoritative하다. 본 문서의 routing,
> fallback 및 reclaim 정책은 causal campaign 결과가 Branch A를 지지할 때만 후속
> 설계 후보로 사용한다.

**작성일**: 2026-08-13  
**상태**: Characterization 및 fixed-MIG online-reclaim gate 완료 · full L1/transport/radio integration 미완료  
**대상 시스템**: CloudLab d8545 · A100-SXM4-40GB ×4 · ConnectX-6 Dx · Aerial 25.3.2

## 1. 연구 질문

> 실행 중인 L1을 중단하거나 MIG geometry를 변경하지 않고, 고정된 MIG/GPU
> endpoint 사이에서 NRx 요청을 routing하여 hard real-time RAN deadline과 queue
> stability를 보장하면서 남는 accelerator capacity를 background AI에 제공할 수 있는가?

MIG geometry는 deployment 시 한 번 정하고 measured run 동안 바꾸지 않는다.
동적으로 바뀌는 것은 다음 세 가지뿐이다.

1. 각 PUSCH job을 보낼 NRx endpoint
2. NRx 대신 conventional receiver를 선택할지 여부
3. background AI가 새 GPU work를 enqueue해도 되는지 여부

## 2. 연구 범위와 비범위

### 중심 contribution

- 고정 MIG topology 위의 drain-free NRx service architecture
- local/sibling-MIG/remote-GPU NRx endpoint의 deadline-aware routing
- queue 상태와 예상 radio gain을 함께 사용하는 NRx admission/fallback
- NRx가 사용하지 않는 시간에만 background AI를 실행하는 work-conserving runtime
- Aerial/cuPHY/TensorRT/P2P/NIC GPUDirect RDMA 실제 구현과 tail-latency 평가

### 중심 claim으로 사용하지 않을 것

- MIG geometry를 slot 단위로 동적으로 변경한다는 주장
- NIC GDR가 P2P보다 빠르다는 주장
- transport microbenchmark만으로 end-to-end zero-copy를 주장
- synthetic fixed-slot latency만으로 carrier-realistic 성능을 주장
- 동일 MIG slice 안의 TensorRT context 수를 replica 수로 간주

기존 `MIG vs MPS vs P2P vs GDR` 결과는 새 runtime의 필요성을 보이는
characterization 및 ablation으로 사용한다.

### Closest-work boundary

- CAORA류 AI-RAN orchestration은 고정된 `1g` MIG unit을 task에 재할당하는 simulation이며
  실행 중 GI resize와 RAN service-chain transport를 구현하지 않는다.
- MIGRator는 continuous-learning workload의 drain/reconfiguration을 최적화하지만, L1을
  무중단으로 유지하는 fixed-topology data plane이 아니다.
- Flex-MIG는 host shared-memory collective로 일반 GPU job의 one-to-many 실행과 makespan을
  개선한다. 본 연구는 PHY dependency chain, absolute deadline, radio correctness,
  conventional fallback, GPU-direct transport가 중심이다.

따라서 `reconfiguration-free MIG` 또는 `distributed MIG` 자체를 최초라고 주장하지 않는다.
차별점은 **fixed isolated accelerator endpoint를 hard-real-time receiver service로 묶고,
radio utility와 queue feasibility를 함께 사용해 per-job execution path를 결정하는 것**이다.

## 3. 현재 결과가 제공하는 출발점

### 2026-08-13 · Fixed 4g+3g online reclaim gate 완료

Topology를 변경하지 않고 4g primary NRx와 3g resident overflow NRx/Qwen을 사용해
`low 600/s → burst 1,100/s → low 600/s` trace를 replay했다. Non-oracle detector는
최근 arrival rate만으로 Qwen enqueue를 멈추고 overflow endpoint를 회수했다.

| Policy | Burst p99 | Qwen decode step/s |
|---|---:|---:|
| Static 1 + Qwen | 1,407.99 ms | 35.21 |
| Naive 2 NRx + Qwen share | 2,276.66 ms | 23.72 |
| Oracle reclaim | 5.29 ms | 22.35 |
| Online adaptive, 3 trials | 1.77–6.24 ms | 21.10–21.44 |
| Dedicated 2 NRx | 1.69 ms | 0 |

이 gate의 scope는 NRx compute queue와 Qwen utility다. L1/P2P/GDR는 별도 component
gate로 검증했으며 online policy와 아직 하나의 full-chain datapath로 결합하지 않았다.
MIG reconfiguration warmed-service outage는 3회 평균 3.643 s였다. 상세 수치와 novelty
판정은 `results/20260813_drain_free/FINAL_FINDINGS_KO.md`를 따른다.

### 2026-08-13 · Independent-endpoint gate 완료

Physical A100 GPU 1/2/3을 각각 독립 Direct-TensorRT/CUDA-Graph endpoint로 사용한
open-loop sweep을 완료했다. GPU 0의 `4g+3g` MIG topology는 변경하지 않았다.

| Endpoints | Capacity mean | 최고 안정 rate | 첫 불안정 rate · p99 |
|---:|---:|---:|---:|
| 1 | 1,170.99 slot/s | 1,150/s | 1,200/s · 42.88ms |
| 2 | 2,318.77 slot/s | 2,350/s | 2,400/s · 48.53ms |
| 3 | 3,448.40 slot/s | 3,500/s | 3,600/s · 47.05ms |

독립 endpoint capacity는 `1×/1.98×/2.94×`로 증가했고, capacity를 2~4% 넘으면
queue tail이 수십 ms로 급증했다. 이 결과는 H2의 compute-only gate를 지지하지만 L1,
transport, fixed-MIG endpoint, radio correctness는 아직 포함하지 않는다. 상세 결과는
`results/20260813_drain_free/RESULT_KO.md`에 있다.

| 항목 | 현재 authoritative 결과 | 새 설계에서의 의미 |
|---|---:|---|
| Direct TRT + Graph, 2g | 2.984 ms · 334.9 slot/s | 단일 2g endpoint 용량 |
| Direct TRT + Graph, 4g | 1.340 ms · 745.1 slot/s | 단일 4g endpoint 용량 |
| Direct TRT + Graph, full | 0.892 ms · 1,130.5 slot/s | 단일 full-GPU endpoint 용량 |
| 동일 slice context N=2~16 | 총 capacity 증가 없음 | replica는 독립 resource여야 함 |
| Same 4g L1+NRx | L1 slowdown 1.621× | co-location의 isolation 손실 |
| Cross 2g\|2g P2P | L1 slowdown 1.043× | protected L1 가설의 근거 |
| P2P round-trip | 76.84 us | transport는 주 병목이 아님 |
| Cross NIC GDR vs P2P, depth 1 | +0.438 ms | GDR는 elasticity 수단이지 latency shortcut이 아님 |

위 수치는 성능 map의 seed일 뿐이다. 새 논문의 primary result는 open-loop queue,
deadline, radio correctness 및 background-AI utility가 함께 있는 실험에서 얻는다.

## 4. 시스템 모델

### 4.1 Job 정의

하나의 job은 `(cell, slot, scheduled PUSCH allocation)`이다.

- arrival time: 실제 slot boundary
- absolute deadline: 해당 Aerial/OAI configuration의 processing budget에서 도출
- input: resource grid, channel estimate, DMRS/config metadata
- output: LLR, decoded transport block, CRC
- radio utility: NRx가 conventional receiver 대비 제공한 TB success/BLER 이득

Receiver chain은 공통 L1 front stage, 선택 receiver stage, 공통 L1 back stage로
분해한다.

```text
arrival -> common front/CE -> {conventional | NRx endpoint} -> LDPC/CRC -> completion
```

NRx pool만 늘려도 common front/back stage의 capacity를 넘을 수는 없다. 안정 조건과
prediction에는 `mu_front`, endpoint별 `mu_nrx`, `mu_conventional`, `mu_back`을 모두 포함한다.

arrival interval과 deadline은 같은 값으로 가정하지 않는다. 예를 들어 job은 1 ms마다
도착할 수 있지만 허용 processing deadline은 scheduler/HARQ configuration에 따라 다를 수
있다. Gate R1에서 실제 configuration으로 확정한다.

### 4.2 실행 경로

| 경로 | 설명 | 역할 |
|---|---|---|
| C | protected L1 MIG의 conventional receiver | 항상 사용 가능한 안전 fallback |
| S | 같은 physical GPU의 sibling MIG NRx | 격리된 near endpoint |
| P | 다른 GPU/MIG의 NRx via CUDA P2P/NVLink | 낮은 transport-cost scale-out |
| G | 다른 GPU/MIG의 NRx via NIC GDR | NIC-routed/disaggregated scale-out |
| A | L1+NRx same partition | 성능 baseline · proposed architecture 아님 |

모든 NRx endpoint는 run 전에 TensorRT engine, CUDA Graph, persistent buffers, MR 및 QP를
미리 준비한다. measured interval에는 model load, container restart 또는 MIG 변경이 없다.

현재 P2P benchmark는 한 process가 두 MIG device를 열고 thread queue로 연결한다. 따라서
P2P 수치는 copy/overlap upper bound로는 유효하지만, 별도 L1/NRx process의 service path라고
자동으로 간주하지 않는다. Cross-process CUDA IPC/P2P가 지원되지 않으면 proposed runtime의
process-isolated endpoint transport는 NIC GDR로 고정한다.

### 4.3 Background AI

Qwen은 NRx와 같은 opportunistic pool에 model을 미리 올려두되 token/decode work의 신규
enqueue를 runtime이 gate한다. queue가 증가하면 이미 실행 중인 kernel은 끝내고 새 AI
work 제출을 중단한다. GPU stream priority만으로 preemption된다고 가정하지 않는다.

## 5. 가설

### H1 · Isolation

Protected L1 MIG의 active-time p99는 NRx endpoint 수와 background-AI offered load가
증가해도 own-partition baseline의 1.05× 이내를 유지할 수 있다.

### H2 · Reconfiguration-free elasticity

독립된 고정 MIG/GPU endpoint를 추가하면 동일 slice에 context를 추가하는 것과 달리
aggregate NRx service capacity가 증가하며, topology 변경 없이 burst backlog를 drain한다.

### H3 · Deadline-safe routing

Shortest-queue 또는 round-robin만 사용하는 정책보다 service-time prediction과 absolute
deadline을 사용하는 routing이 deadline miss 및 p99 sojourn time을 줄인다.

### H4 · Radio-aware admission

동일 deadline budget에서 NRx가 유용한 channel/job을 우선 선택하면 queue-only fallback보다
TB success, BLER 또는 goodput이 높다.

### H5 · Work-conserving AI

NRx headroom을 침범하지 않도록 background enqueue를 gate하면 static reservation보다 AI
goodput을 높이면서 RAN deadline과 L1 isolation을 유지한다.

### H6 · GDR의 역할

NIC GDR의 가치는 단일 job latency 단축이 아니라 CPU bounce 없이 독립 endpoint를 NRx
pool에 포함하는 것이다. 동일 depth/endpoint 조건에서 host-staging 대비 CPU traffic과
tail latency를 줄이되, CUDA P2P보다 빠르다고 가정하지 않는다.

## 6. Realistic workload 설계

### 6.1 세 단계 workload

1. **Performance-only gate**
   - 현재 고정 tensor와 동일 engine contract 사용
   - routing/transport/queue 구현 검증용
   - radio-quality claim에는 사용하지 않음
2. **Link-level correctness workload**
   - pyAerial/Sionna의 model-supported TDL/CDL/AWGN channel
   - ground-truth transport bits 보존
   - conventional/NRx 양 경로의 LLR, CRC, TB success 기록
   - 현재 pretrained engine이 지원하는 MCS/QPSK/config를 primary로 사용
3. **Trace-driven system workload**
   - periodic slot arrivals가 primary이며 Poisson arrival는 robustness ablation으로만 사용
   - 실제 traffic trace는 시간별 active cell/UE/PUSCH job 수를 modulation하는 데 사용
   - 여러 cell의 job은 같은 slot boundary에 동시 도착시켜 burstiness를 보존

### 6.2 Arrival sweep

- slot period: 실제 configured numerology에서 확정; 후보 0.5/1.0 ms
- active cells: `1, 2, 4, 8`
- burst length: `10, 100, 1,000` slots
- offered load: aggregate endpoint capacity의 `0.25, 0.50, 0.75, 0.90, 1.00, 1.10, 1.25×`
- channel state: model-supported SNR/channel/Doppler 범위에서 stratified sampling
- 모든 정책에 동일 arrival/channel trace와 seed를 replay

단일 Python loop가 이전 job 완료 후 다음 job을 생성하는 closed-loop 방식은 primary
evaluation에서 금지한다.

### 6.3 Deadline 확정 Gate R1

실험 전 다음을 실제 Aerial/OAI configuration에서 기록한다.

- numerology와 slot duration
- PUSCH arrival point
- LLR/TB가 소비되어야 하는 pipeline point
- scheduler/HARQ가 허용하는 processing budget
- fronthaul 및 downstream stage를 포함할지 여부

Primary deadline은 이 결과로 하나를 고정하고, `0.5×, 1×, 2×` budget sensitivity를
부가 실험으로 수행한다. 임의의 `1 ms`를 deadline과 arrival period 양쪽에 동시에 쓰지 않는다.

## 7. Fixed topology와 endpoint 정의

### 7.1 원칙

- 각 measured run의 시작과 끝에서 MIG UUID/placement manifest SHA를 저장
- run 중 `nvidia-smi mig -cgi/-dgi/-dci`, MIG mode toggle 및 GPU reset 금지
- NRx replica는 별도 GI 또는 별도 physical GPU에 있는 실행 endpoint
- 같은 GI의 여러 TensorRT context는 concurrency ablation일 뿐 replica로 세지 않음

### 7.2 최소 topology set

| ID | 고정 배치 | 목적 |
|---|---|---|
| T0 | Full GPU · L1+NRx+Qwen MPS | 비격리 Pareto baseline |
| T1 | 4g L1+NRx · sibling 3g Qwen | same-partition dependency baseline |
| T2 | 2g L1 · 2g NRx · 3g Qwen | 기존 same-GPU cross-MIG 재현 |
| T3 | protected L1 MIG · sibling-MIG NRx · remote-GPU NRx pool | proposed local+overflow |
| T4 | protected L1 MIG · remote-GPU NRx pool · opportunistic AI | scale-out 및 GDR/P2P 비교 |

T3/T4의 정확한 profile은 각 endpoint가 model을 수용하고 목표 arrival capacity를 만들도록
2g/3g/4g/full service map을 측정한 후 고정한다. 결과를 본 뒤 topology를 config마다 바꾸지
않고, 한 primary topology를 모든 routing 정책에 공통 사용한다.

## 8. 비교 정책

| ID | 정책 | 설명 |
|---|---|---|
| B0 | Conventional only | radio/compute 하한선 |
| B1 | All NRx same | same-partition latency · isolation 없는 기준 |
| B2 | Static remote | 모든 job을 하나의 고정 remote endpoint로 전송 |
| B3 | Round-robin | endpoint capacity/queue를 보지 않음 |
| B4 | Shortest queue | 현재 queue만 이용 |
| B5 | EDF feasible | predicted finish time이 deadline 이전인 endpoint 중 EDF |
| B6 | Queue-threshold fallback | queue가 임계치를 넘으면 conventional |
| P | Utility-aware deadline routing | deadline-feasible endpoint 중 예상 radio gain이 큰 job에 NRx 배정 |
| O | Offline oracle | 실제 두 receiver 결과를 미리 아는 비배포 upper bound |

Policy P의 최소 state는 다음과 같다.

- endpoint별 queue와 in-flight completion estimate
- resource profile별 calibrated service-time distribution
- job absolute deadline
- transport path별 latency distribution
- channel/MCS/DMRS residual 등 NRx benefit predictor feature
- background AI in-flight 상태

첫 구현은 table/quantile predictor로 시작한다. RL은 강한 비선형 이득이 확인된 뒤에만
추가하며 novelty를 위해 억지로 사용하지 않는다.

## 9. 실험 Phase

### Phase A · Correctness와 공정성 gate

#### A1 · Full-chain bit correctness

- 동일 transmitted bits를 conventional과 NRx 경로에 입력
- LLR shape/bit order/scale 검증
- LDPC/CRC까지 실행
- P2P/GDR 전후 output과 local direct output 일치 확인

**통과 기준**: transport별 deterministic vector CRC 일치, finite output, stale buffer 0건.

#### A2 · P2P/GDR 동일-depth 재측정

- 같은 ring depth `1, 2, 4, 8`
- 같은 payload, endpoint profile, warm-up, iterations
- transport latency, e2e latency, CPU time, throughput 동시 기록
- in-process dual-device P2P와 cross-process/container P2P를 분리
- cross-process P2P가 CUDA IPC 또는 driver policy로 불가능하면 실패 사실 자체를 기록하고
  in-process P2P를 deployment baseline이 아닌 upper bound로 표시

**통과 기준**: 현재 P2P-depth2/GDR-depth1 confound 제거.

#### A3 · No-reconfiguration invariant

- topology manifest watcher와 GPU reset/MIG event logger 추가
- run 중 L1 PID/context/UUID가 변하지 않았음을 검증

### Phase B · Service map calibration

- common L1 front/back 및 full conventional path: `2g, 4g, full`
- endpoint resource: `2g, 3g, 4g, full`
- independent endpoint count: `1, 2, 3`
- background AI offered load/cap: `0, 30, 50, 70, 100%`
- input classes: primary supported PUSCH config + supported channel strata
- 각 조합에서 service p50/p95/p99, throughput, interference, memory footprint 측정

같은 slice context `1,2,4,8,16` 결과는 기존 데이터를 사용하고 반복하지 않는다.

**통과 기준**: stage별 service quantile predictor hold-out error를 보고하고, aggregate
NRx capacity가 독립 endpoint 추가 시 실제로 증가함을 확인. Primary offered load는 먼저
`lambda < mu_L1-common`인 영역으로 제한하여 NRx routing 효과와 L1 자체 overload를 혼동하지
않는다.

### Phase C · Open-loop queue와 scale-out

- T3/T4 고정 topology
- 동일 trace를 endpoint count `1,2,3`에 replay
- offered load `0.25~1.25×`
- normal, step burst, periodic burst, synchronized multi-cell arrival
- B2~B6 비교

**핵심 결과**: queue time series, deadline miss, backlog at window end, drain time,
L1 active p99, endpoint utilization.

### Phase D · Radio-aware fallback

1. 각 channel sample에서 conventional과 NRx를 모두 offline 실행
2. `Delta utility = U_NRx - U_conventional` label 생성
3. runtime-visible feature로 lightweight benefit predictor 학습
4. B5/B6/P/O를 동일 trace에서 비교

Primary radio utility 우선순위:

1. decoded TB goodput
2. CRC/TB success
3. BLER
4. LLR-level metric은 보조 지표

**통과 기준**: Policy P가 동일 deadline-miss budget에서 B5/B6보다 radio utility를
개선하거나, 동일 radio utility에서 background-AI goodput을 개선.

### Phase E · Opportunistic background AI

- Qwen model은 미리 load하고 decode enqueue만 gate
- static cap, always-on, queue-threshold pause, proposed headroom controller 비교
- AI burst 시작/중지와 NRx burst가 교차하는 trace 사용
- 마지막 AI kernel 제출 후 NRx가 service budget을 회복하는 quiescence delay 측정

**통과 기준**: protected L1 invariant를 유지하고, static reservation보다 AI token/s 증가,
always-on보다 NRx deadline miss 감소.

### Phase F · MIG reconfiguration 비용 대조군

실시간 정책으로 사용하지 않고 premise를 검증하는 한 번의 disruptive experiment다.

- topology A에서 B로 전환
- workload drain, container stop, CI/GI destroy/create, model reload, CUDA Graph warm-up
- 각 단계 시간과 그동안 처리하지 못한 slot 수 기록
- L1 process를 유지할 수 있는 부분 전환과 전체 drain 필요 전환을 구분
- routing-only endpoint switch와 downtime 비교

**주의**: 이 실험은 measured policy run과 분리하고, topology backup/restore 검증 후 수행한다.

### Phase G · 장시간 안정성

- primary topology와 정책 P를 최소 30분, 최종본은 가능하면 2시간 실행
- 최소 `10^6` job 또는 trace 한 주기 이상
- clock, power, temperature, ECC, NIC counters, queue, deadline miss 동기 기록

## 10. Metric

### RAN correctness/quality

- decoded TB goodput
- CRC success / TB error / BLER
- NRx 선택률과 conventional fallback률
- NRx가 실제 결과를 개선/악화시킨 job 비율

### Real-time

- L1 active p50/p95/p99/max
- queue waiting, service, transport, end-to-end sojourn 분리
- deadline miss ratio와 연속 miss burst 길이
- max outstanding, backlog at window end, drain time

### Elasticity/utilization

- endpoint별 busy fraction
- aggregate NRx slot/s
- accepted/offloaded/fallback job 수
- topology change 0건, L1 restart 0건
- background AI token/s 및 first-token/inter-token latency

### Cost

- CPU utilization과 memory traffic
- PCIe/NVLink/NIC bytes 및 counters
- GPU memory footprint
- energy/power는 가능한 경우 보조 지표

## 11. 통계 원칙

- 기능 smoke를 제외한 primary config는 최소 5 independent trials
- 정책 비교는 동일 trace/seed를 쓰는 paired design
- run order randomized, warm-up과 clock policy 고정
- primary queue run은 config당 최소 5분; tail claim은 합계 `>=10^5` jobs
- mean뿐 아니라 p50/p95/p99/max와 95% bootstrap CI 보고
- deadline miss가 0이면 0만 쓰지 않고 sample 수와 one-sided upper bound 보고
- 시작/종료 시 GPU clock, temperature, power, ECC, MIG manifest, container image hash 저장

## 12. Go/No-Go 기준

### Gate 1 · Production-path validity

Direct TensorRT와 full receiver correctness가 확인되지 않으면 system sweep 중단.

### Gate 2 · Deadline validity

실제 pipeline budget을 확인하지 못하면 “hard real-time deadline” claim 금지. 대신
service-capacity characterization으로 범위를 축소.

현재 Python component harness의 4g L1-only closed-loop capacity도 약 613 slot/s이므로,
1,000 slot/s carrier target을 주장하기 전에 C++/CUDA-Graph production path에서 common
front/back capacity를 다시 측정한다. 실제 arrival rate보다 L1 자체 capacity가 낮으면 NRx
routing 실험보다 L1 path 최적화를 먼저 수행한다.

### Gate 3 · Radio benefit heterogeneity

channel/job에 따라 NRx의 incremental utility가 유의하게 달라지지 않으면 radio-aware
selection claim을 버리고 deadline-safe elastic routing만 평가.

### Gate 4 · Independent endpoint scaling

독립 endpoint 추가가 aggregate capacity를 늘리지 못하면 transport/CPU/PCIe 병목을 먼저
해결하고 policy experiment를 진행하지 않음.

### Gate 5 · Proposed-policy dominance

Policy P가 B5/B6 대비 radio utility, deadline miss, background-AI goodput 중 어느 Pareto
축에서도 개선하지 못하면 central contribution으로 사용하지 않음.

## 13. 필요한 구현 단위

기존 optimized direct-TRT/P2P/GDR 코드를 재사용하고 다음을 새로 분리한다.

- `fixed_topology_manifest.py`: UUID/profile/placement hash와 변경 감시
- `nrx_endpoint.py`: persistent DirectNrx + Graph + ring endpoint
- `slot_arrival_replay.py`: periodic synchronized multi-cell open-loop replay
- `l1_nrx_router.py`: B2~B6/P routing policy와 absolute deadline
- `conventional_fallback.py`: 동일 PUSCH config의 conventional LLR/LDPC/CRC
- `radio_benefit_dataset.py`: paired conventional/NRx label 생성
- `background_ai_gate.py`: Qwen enqueue pause/resume와 quiescence 측정
- `run_drain_free_sweep.sh`: topology 불변 gate와 실험 orchestrator
- `analyze_drain_free.py`: raw log 검증, paired statistics, figure 생성

## 14. 논문 figure 계획

1. **Problem figure**: static MIG isolation과 dynamic NRx demand의 capacity fragmentation
2. **Reconfiguration downtime**: drain/recreate/reload/warm-up vs routing-only switch
3. **Service map**: 2g/3g/4g/full 및 independent endpoint 수별 capacity
4. **Queue phase transition**: offered load 대비 p99/backlog/deadline miss
5. **Policy comparison**: deadline miss vs decoded TB goodput
6. **AI-RAN Pareto**: background token/s vs RAN deadline/radio utility
7. **Ablation**: same partition, P2P, host staging, NIC GDR
8. **Long-run trace**: NRx burst, routing, fallback, AI gating, queue를 한 timeline에 표시

## 15. 최종 성공 조건

다음이 모두 성립할 때 central claim을 지지한다.

1. measured run 중 MIG geometry 변경과 L1 restart가 0건이다.
2. protected L1 active p99가 own-partition baseline에 근접한다.
3. 독립 endpoint scale-out이 overload 구간의 queue를 안정화한다.
4. infeasible job은 conventional fallback되어 stale NRx result나 unbounded backlog가 없다.
5. proposed policy가 deadline/radio utility/AI goodput Pareto에서 기존 정책보다 우수하다.
6. 모든 radio-quality 결과가 ground-truth bits와 full LDPC/CRC로 검증된다.

논문 메시지는 “GDR가 빠르다”가 아니라 다음으로 고정한다.

> Static MIG provides isolation but not elasticity. A drain-free, deadline- and
> radio-aware NRx service layer recovers elasticity through pre-provisioned
> accelerator endpoints, request routing, and safe receiver fallback without
> reconfiguring or restarting the real-time L1.
