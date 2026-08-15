# DART-Rx 전체 실험 계획

> **상태 주의:** 현재 실행 중인 MIG 중심 causal campaign과 결과별 architecture 분기는
> `MIG_NRX_RESEARCH_CHECKPOINT_KO.md`를 따른다. 본 문서의 DART mechanism 실험은
> problem existence가 확인될 때까지 동결한다.

작성일: 2026-08-13  
연결 설계: `ISCA_ARCHITECTURE_V2_KO.md`  
실행 matrix: `ISCA_EXPERIMENT_MATRIX_V2.csv`

## 0. 이 계획이 답해야 하는 질문

모든 실험은 아래 다섯 질문 중 하나에만 연결한다.

1. **Problem:** fixed isolation 때문에 실제로 idle capacity와 NRx deadline miss가 동시에 생기는가?
2. **Cause:** transport, NRx compute, queue, non-preemptible background kernel, host control 중 무엇이 tail을 만드는가?
3. **Mechanism:** deadline-tagged admission, device commit, JIT fallback, lease가 각각 무엇을 해결하는가?
4. **Architecture:** DART-Q hardware assist가 software scheduler보다 왜 필요한가?
5. **Outcome:** radio utility, PHY deadline, AI application SLO를 동시에 개선하는가?

`MIG/MPS/MIG+MPS/P2P/GDR` 비교는 필수다. 다만 `무엇이 더 빠르다`에서
멈추지 않고, 각 결과를 contention, isolation, fragmentation, transport,
control-path 비용으로 분해해야 한다.

## 1. 평가 원칙

### 1.1 두 개의 clock

- host orchestration: `CLOCK_MONOTONIC_RAW`
- GPU event/descriptor/commit: CUDA event와 `clock64()` calibration

host/GPU timestamp를 run 시작과 끝에 교정하고 drift를 저장한다.

### 1.2 두 종류의 run

- **measurement run:** profiler 없이 latency/throughput/queue 측정
- **attribution run:** Nsys/NCU/CUPTI를 켜고 짧은 window만 분석

profiler 수치를 production latency처럼 섞지 않는다.

### 1.3 open-loop arrival

NRx와 background service는 closed-loop 최대속도 외에 open-loop arrival을 반드시 사용한다. producer가 느려져 arrival 자체가 줄어드는 coordinated omission을 막는다.

### 1.4 반복과 통계

- microbenchmark: warm-up 후 최소 10,000 samples 또는 30초, 5 trials
- end-to-end: condition당 최소 5분, 3 trials
- long run: 2시간 이상
- 보고: median trial과 trial 간 범위, sample p50/p95/p99/p99.9/max
- deadline: miss ratio와 longest consecutive miss를 함께 보고
- bootstrap 95% CI와 effect size를 저장

### 1.5 correctness

timing만 맞고 tensor가 틀린 run은 폐기한다.

- input/output checksum
- `(slot_id, epoch)` 일치
- wrapper/direct/graph output allclose
- late-result injection에서 stale commit 0건
- radio replay에서는 CRC/BLER/goodput

## 2. topology와 비교 대상

### 2.1 hardware topology

주 topology는 GPU0의 fixed `4g L1 + 3g expert pool`이다. GPU1--3은 full GPU endpoint와 background workload에 쓴다. topology를 run 중 변경하지 않는다.

### 2.2 canonical five-way comparison

| ID | GPU budget | L1/NRx placement | background placement | transport |
|---|---|---|---|---|
| C-MPS | full A100 | L1+NRx+AI가 full GPU를 MPS 공유 | same GPU | none |
| C-MIG | 4g+3g | L1+NRx가 4g 공유 | isolated 3g | none |
| C-MIGMPS | 4g+3g | L1+NRx가 4g에서 MPS 공유 | isolated 3g | none |
| C-P2P | 4g+3g | L1 4g, NRx 3g | NRx와 3g MPS 공유 | same-GPU P2P |
| C-GDR | 4g+3g 또는 4g+remote | L1 4g, NRx isolated endpoint | NRx pool과 공유 | NIC loopback GDR |

모든 approach는 다음 두 pass를 수행한다.

- **C0 compute-only:** background 없이 L1-only, NRx-only, L1+NRx
- **C1 realistic contention:** 동일 offered background trace와 open-loop NRx

고정 조건:

- raw TensorRT/caller-owned binding 경로
- current QPSK payload: forward 1,415,232 B, backward 314,496 B
- NRx replica 1부터 시작, 이후 동일 replica sweep
- 동일 arrival timestamps와 random seed
- 동일 clock/power/thermal envelope
- 동일 warm-up, samples, trials

보고 항목:

- absolute L1/slot sojourn p50/p95/p99/p99.9
- 해당 topology의 L1-only 대비 slowdown
- NRx service와 queue wait
- deadline miss와 longest consecutive miss
- background application SLO/throughput
- GPU slice utilization, idle GPU-seconds, power
- pack/transport/visibility/unpack/control breakdown

P2P와 GDR은 동일 endpoint/topology에서 transport만 교체하는 paired run을
우선한다. 불가능한 topology는 별도 control로 표기하고 같은 bar에 섞지 않는다.

### 2.3 baseline

| ID | placement/control | 의미 |
|---|---|---|
| B0 | cuPHY only | L1 lower bound |
| B1 | L1+NRx isolated, no background | mandatory NRx service baseline |
| B2 | full GPU + MPS | same-partition work conservation |
| B3 | full GPU + MPS priority | CUDA priority baseline |
| B4 | same MIG GI + MPS | constrained same-partition co-location |
| B5 | split MIG + host shared memory | isolation + CPU copy |
| B6 | split MIG + P2P | transport-only cross-partition |
| B7 | split/remote + GDR | transport-only isolated fabric |
| B8 | static dedicated NRx replicas | overprovisioned deadline baseline |
| B9 | shortest queue | simple scale-out routing |
| B10 | predicted finish, host control | strong software admission baseline |
| B11 | immediate dual execution | always NRx+conventional speculation |
| B12 | block on NRx | no fallback safety baseline |
| B13 | ARCHES-like selected-only/concurrent | expert-selection conceptual baseline |
| B14 | REEF/Tally artifact where compatible | generic sharing/preemption baseline |
| O | oracle | future service time known |

### 2.4 DART variants

| ID | variant |
|---|---|
| S0 | static one-endpoint FIFO |
| S1 | round robin |
| S2 | shortest queue |
| S3 | p99 predicted-finish feasibility |
| S4 | S3 + atomic endpoint/tensor-slot reservation |
| S5 | S4 + slot/epoch single-winner commit |
| S6 | S5 + JIT conventional fallback |
| S7 | S6 + fixed background quantum |
| S8 | S6 + grant-ahead deadline-safe lease |
| S9 | S8 + device observe/commit path |
| S10 | DART-Q trace/RTL model |
| O | ideal zero-control-overhead oracle |

각 variant는 직전 variant에 mechanism 하나만 추가한다. 구현 algorithm과 state
machine은 `DART_RX_SCHEME_DESIGN_V0_KO.md`를 따른다.

## 3. realistic NRx workload

실행 가능한 traffic/radio realism 분리와 현재 자동화 경로는
`DART_RX_REALISTIC_WORKLOAD_GATE_KO.md`를 따른다. Multi-cell arrival timing만
현실적인 run을 BLER/CRC 근거로 사용하지 않는다.

### 3.1 timing trace

세 단계로 진행한다.

1. **deterministic tensor replay:** current shape로 mechanism/correctness 디버깅
2. **recorded channel replay:** SNR, MCS, PRB, layer, UE count가 바뀌는 IQ/CE input
3. **Aerial/OAI traffic replay 또는 OTA:** 가능해지는 즉시 최종 validation

CloudLab에 RU link가 없으므로 1과 2를 main reproducible evaluation으로 삼고, 3은 외부 testbed validation으로 분리한다. random synthetic input을 radio-result 근거로 쓰지 않는다.

### 3.2 traffic pattern

각 service class는 같은 mean만 쓰지 않고 burst structure를 가진다.

- synchronized cells: 1, 2, 4, 8 cells
- periodic slots: 0.5/1.0 ms interval
- UE burst: ON/OFF Pareto 또는 measured trace
- load: endpoint isolated capacity의 30, 50, 70, 85, 95, 100, 105, 120%
- mixed tensor classes: MCS/PRB/layer별 measured service distribution

### 3.3 NRx replicas

replica 수 `N={1,2,4,8}`을 sweep한다. 구분해야 할 것은 다음과 같다.

- one endpoint 안의 concurrent execution
- 여러 MIG/full GPU endpoint로 scale-out
- protected L1 stage의 공통 직렬 병목

replica마다 독립 TensorRT context, stream, input/output buffer, graph exec handle을 둔다. 같은 context를 여러 request가 공유해 serialization되는 오류를 막는다.

## 4. realistic background workload

### 4.1 Generative AI

- model: Qwen2.5-7B 또는 memory-fit instruct model
- inputs: 공개 real prompt trace; prompt length bucket 보존
- phases: prefill과 decode 별도 측정
- load: interactive Poisson/bursty arrival와 saturated throughput 둘 다
- metrics: TTFT, TPOT, request p99, token/s
- lease units: decode step, transformer layer group, prefill tile

기존 random token closed-loop 결과는 kernel characterization에만 사용한다.

### 4.2 Video analytics

- 실제 H.264/H.265 clip을 decode
- TensorRT detector/segmenter의 pretrained weights 사용
- 1/4/8 streams, 15/30 FPS
- metrics: end-to-end frame p99, deadline drop, FPS, accuracy checksum
- lease units: frame 또는 TensorRT engine segment

random tensor + `weights=None` ResNet은 baseline으로 인정하지 않는다.

### 4.3 Streaming speech

- 실제 audio chunk가 wall-clock rate로 도착
- streaming-capable Whisper/RNN-T 경로
- 무조건 30초 padding하는 batch API 금지
- metrics: chunk p99, real-time factor, backlog, word error rate subset
- lease units: encoder chunk 또는 decoder step

### 4.4 Online/FL training

- 실제 dataset subset과 pretrained checkpoint
- micro-batch 1/2/4/8
- forward/backward/optimizer를 NVTX로 분리
- metrics: samples/s, iteration p99, convergence proxy, pause/resume overhead
- lease unit: one micro-batch

### 4.5 RAN-native AI

- CSI/channel/beam predictor
- recorded feature trace
- metrics: prediction deadline와 task metric

이 workload는 NRx와 같은 RAN data rhythm을 가지므로 secondary case study로 사용한다.

### 4.6 혼합 배치

단일 3g에 모든 model을 억지로 넣지 않는다. 세 realistic deployment를 둔다.

- Edge site A: NRx pool + video analytics
- Edge site B: NRx pool + streaming speech + small RAN-native model
- Edge site C: NRx pool + generative AI 또는 online training

GPU1--3을 써서 실제 memory pressure를 보존하고, placement manifest를 공개한다.

## 5. Phase P0 — freeze와 capability gate

### P0.1 Manifest

수집 항목:

- GPU/MIG UUID와 profile
- clocks, persistence, power limit, thermal state
- NIC/GID/link/NUMA/topology
- image digest, driver/CUDA/TensorRT/Aerial version
- git/source hash
- model/data hash

### P0.2 CUDA feature probe

작은 C++ test로 다음을 실제 A100에서 검증한다.

- device graph launch
- conditional IF/SWITCH graph
- graph exec handle pool의 동시 launch
- GPU-visible completion flag ordering
- stream memory operations/atomic support

### P0.3 Transport capability

- same-GPU MIG P2P
- cross-GPU P2P
- GDR MR per GPU/MIG
- GPUDirect flush visibility
- DOCA GPUNetIO 설치/지원 여부

실패한 기능은 숨기지 않고 DART software path의 제한으로 기록한다.

## 6. Phase P1 — NRx를 1.34 ms 아래에서 해부

### P1.1 wrapper/direct/binding/graph

동일 input에 대해 다음을 비교한다.

- public wrapper
- raw TensorRT enqueue
- caller-owned input binding
- caller-owned input/output binding
- persistent buffers
- CUDA Graph
- graph + concurrent contexts

각 variant에서 output allclose와 pointer stability를 검사한다.

### P1.2 Nsys decomposition

NVTX range:

```text
slot / pack / transport_fwd / queue_wait / trt_enqueue /
nrx_gpu / transport_bwd / fallback / commit / ldpc_crc
```

수집:

- CUDA runtime call duration
- memcpy kind/bytes/duration
- synchronization call
- kernel launch-to-start gap
- kernel duration/name/stream/context
- graph launch와 node gap

### P1.3 NCU top kernels

GPU-time 상위 kernel을 대상으로 다음을 수집한다.

- SM/tensor utilization
- memory throughput
- achieved occupancy
- warp stall reasons
- launch dimensions와 register/shared memory

목적은 NRx를 새로 최적화하는 것뿐 아니라 service-time bound가 tensor class에 따라 얼마나 안정적인지 확인하는 것이다.

### P1.4 replica throughput

replica/context `1,2,4,8`, batch `1,2,4`에서:

- per-slot latency
- aggregate slots/s
- interference/serialization
- memory footprint
- graph handle reuse limit

이 결과가 DART endpoint profile table을 만든다.

## 7. Phase P2 — architecture gap을 드러내는 실험

### P2.1 Non-preemption residual test

background kernel 실행 중 임의 offset에 high-priority NRx를 inject한다.

- Qwen prefill/decode
- vision
- speech
- training backward

각 injection에 대해:

- arrival-to-first-NRx-kernel
- 당시 실행 중이던 background kernel의 residual time
- stream priority on/off
- MPS active-thread percentage

예상 signature는 `NRx dispatch delay ~= residual background kernel`이다. 상관이 없다면 lease/non-preemption framing을 수정한다.

### P2.2 True quantum sweep

sequence length 전체를 quantum이라고 부르지 않는다. 실제 graph partition 단위를 바꾼다.

- Qwen layer group 1/2/4/8
- training micro-batch
- frame engine segment
- speech encoder block

각 unit의 p99와 background throughput loss를 측정한다.

### P2.3 Host control-path breakdown

descriptor 생성부터 graph start까지:

- Python router
- C++ host router
- host CUDA Graph launch
- persistent GPU polling
- simulated DART-Q doorbell

host average가 아니라 p99.9와 CPU scheduling perturbation에 대한 민감도를 본다.

## 8. Phase P3 — killer problem experiment

이 phase가 논문 진행 여부를 결정한다.

### P3.1 Same-trace contradiction

동일 multi-cell trace를 다음 provisioning에 replay한다.

- static 1 replica
- static peak-provisioned replicas
- dynamic MIG reconfiguration
- MPS shared full GPU
- fixed isolated endpoint pool

동시에 그린다.

- time-series queue depth/deadline miss
- endpoint utilization
- total idle GPU-seconds
- background application throughput

성공적인 problem figure는 **어떤 endpoint는 queue deadline을 놓치는 순간에도 다른 isolated compute가 idle**임을 보여야 한다.

### P3.2 Demand transition

30→60→90→120→60→30% load와 짧은 2--20 ms burst를 넣는다.

- fixed partition의 queue cliff
- dynamic MIG의 drain/reload outage
- MPS의 interference tail
- remote endpoint의 activation delay

### P3.3 Provisioning frontier

x축을 provisioned NRx capacity, y축을 deadline miss, 색을 idle GPU-seconds로 한다. static isolation이 deadline과 utilization을 동시에 만족하지 못하는 영역이 존재해야 한다.

## 9. Phase P4 — fair transport experiment

### P4.1 payload contract

current QPSK direct path는 forward 1,415,232 B, backward 314,496 B다. 1,257,984 B는 generic eight-value control이며 current NRx output과 혼동하지 않는다.

### P4.2 transport variants

- CPU shared memory
- CPU-buffer RDMA
- same-GPU MIG P2P
- cross-GPU P2P where supported
- GDR staging
- caller-owned zero-copy GDR when binding permits

### P4.3 fair timing boundary

다음을 모두 포함한 end-to-end transport time을 보고한다.

- pack/layout conversion
- publish/doorbell
- DMA
- receiver visibility/flush
- unpack 또는 binding
- return
- completion/epoch validation

`mr.write()`나 `cp.asnumpy()`를 timer 밖에 둔 microbenchmark로 zero-copy를 주장하지 않는다.

### P4.4 depth and concurrency

queue depth `1,2,4,8,16,32`와 in-flight slots를 sweep한다. NIC/P2P의 bandwidth가 아니라 NRx service와 겹칠 수 있는지가 핵심이다.

## 10. Phase P5 — DART software prototype

### P5.1 Host DART

구현:

- 64-byte descriptor
- per-endpoint p99 profile
- tail estimate와 credit
- tensor-slot pool
- epoch-safe completion
- admission reject
- JIT fallback

구현을 한 번에 켜지 않고 `predicted finish -> reservation -> epoch commit -> JIT`
순서로 같은 trace에서 cumulative ablation한다.

### P5.2 Device DART

구현:

- GPU-visible SPSC/MPSC ring
- persistent polling kernel
- pre-uploaded graph exec pool
- device timestamps
- device commit CAS
- stale/late completion drop

### P5.3 Lease controller

background graph unit마다 calibrated p99를 붙인다. earliest admitted NRx latest-start 전에 drain 가능한 unit만 dispatch한다.

fixed gate, fixed quantum, grant-ahead lease를 분리 비교한다. grant-ahead lease는
UL schedule lookahead와 unexpected-arrival bound `Qmax`를 모두 사용한다.

### P5.4 Policy matrix

비교:

- random/round robin
- shortest queue
- least predicted finish
- utility-feasible
- utility-feasible + JIT fallback
- utility-feasible + JIT fallback + lease
- oracle

### P5.5 fault injection

- endpoint hang
- delayed completion
- duplicate completion
- stale epoch
- transport error
- profile underprediction 5/10/20/50%

목표는 stale commit 0건과 bounded fallback이다.

## 11. Phase P6 — DART-Q hardware evaluation

### P6.1 Trace-driven simulator

입력:

- observed kernel/DMA duration trace
- request arrival
- endpoint queue state
- graph quantum trace

변수:

- SRT 16/32/64/128
- graph profile error
- doorbell latency 0.1--50 us
- commit guard
- graph exec pool size
- endpoint count 1--16

검증은 A100 host/device prototype의 동일 policy 결과와 event order를 맞추는 방식으로 한다.

### P6.2 RTL

SystemVerilog로 최소 다음을 구현한다.

- descriptor SRAM interface
- latest-start comparator tree
- feasibility subtract/compare
- credit update
- epoch/CAS state machine
- lease gate

합성 결과:

- max frequency
- logic area/gate count
- SRAM bytes
- dynamic/static power estimate
- operation latency

SM/tensor core modification이 없음을 명확히 한다.

### P6.3 Hardware ablation

- no deadline field
- no endpoint credit
- no epoch guard
- no JIT fallback
- no lease gate
- host doorbell
- polling doorbell
- ideal completion doorbell

### P6.4 Architecture go/no-go

DART-Q가 device polling 대비 tail이나 reserved-SM cost를 실질적으로 줄이지 않으면 architecture contribution을 주장하지 않는다.

## 12. Phase P7 — full-system evaluation

### P7.1 workload-by-workload

각 background class를 동일 NRx trace와 개별 실행한다. policy 간 비교에서 NRx trace와 random seed를 고정한다.

### P7.2 mixed deployment

세 site profile을 30분씩 실행한다.

- site A: 4-cell NRx + 4 video streams
- site B: 4-cell NRx + streaming speech + beam predictor
- site C: bursty 8-cell NRx + Qwen interactive 또는 online training

### P7.3 full metrics

PHY:

- end-to-end p50/p95/p99/p99.9
- deadline miss ratio
- longest consecutive miss
- fallback and late-drop rate
- BLER/CRC/goodput

NRx fabric:

- admitted/rejected/routed requests
- queue wait/service/transport/commit breakdown
- endpoint utilization and idle GPU-seconds
- activation delay
- prediction error

background:

- TTFT/TPOT, frame p99/FPS, speech RTF, training samples/s
- application SLO attainment

architecture:

- doorbell-to-graph-start
- persistent polling SM occupancy
- descriptor traffic
- queue occupancy
- hardware state/area/power

system:

- host CPU utilization and memory traffic
- GPU power/energy per useful NRx result
- NIC bytes/packets

### P7.4 long run

최종 두 configuration을 2시간 이상 실행한다.

- static/strongest baseline
- DART best variant

thermal throttling, clock drift, error recovery, memory leak, epoch wrap/slot reuse를 확인한다.

## 13. Phase P8 — radio utility

### P8.1 fallback calibration

conventional receiver의 tensor class별 p99 bound를 측정한다.

### P8.2 NRx utility map

SNR/MCS/channel condition별로:

- conventional BLER/CRC/goodput
- NRx BLER/CRC/goodput
- incremental GPU time/energy

### P8.3 policy result

deadline만 지키는 정책과 utility-feasible 정책을 비교한다. 동일 deadline miss라면 더 높은 goodput 또는 동일 goodput에서 더 낮은 NRx 사용량을 보여야 한다.

radio ground truth가 없는 synthetic input에서는 이 phase를 수행한 척하지 않는다.

## 14. 핵심 ablation

| ablation | 드러내는 원인 |
|---|---|
| avg profile vs p99 profile | tail-aware admission 필요성 |
| host vs device commit | host jitter/round-trip |
| immediate vs JIT fallback | 중복 compute 비용 |
| epoch guard off | stale-result correctness |
| lease off vs fixed quantum vs predictive lease | background blocking 관리 |
| one vs many endpoints | horizontal elasticity |
| P2P vs GDR with same policy | transport substrate 영향 |
| fixed utility vs radio-aware utility | optional NRx의 실제 가치 |
| polling vs DART-Q model | hardware assist 가치 |

## 15. 예상 paper figures

1. **Isolation-elasticity contradiction:** 시간축 queue miss와 동시에 존재하는 idle compute
2. **CUDA/API critical path:** host submit, queue wait, residual kernel, NRx, transport, commit
3. **Non-preemption:** background residual kernel 대 NRx dispatch delay
4. **Deadline/utilization frontier:** MPS, MIG, host router, DART
5. **Endpoint scale-out:** replicas 대 slots/s/p99/deadline miss
6. **Transport breakdown:** pack/DMA/visibility/unpack; P2P/GDR/host
7. **Policy ablation:** admission, epoch, fallback, lease 누적 효과
8. **Heterogeneous workload:** PHY SLO 대 application SLO
9. **Radio utility:** goodput/BLER 대 NRx compute budget
10. **Architecture:** DART-Q queue depth/area/power/speedup
11. **Long run CDF:** baseline 대 DART p99.9/consecutive miss

## 16. success criterion

절대 숫자를 결과에 맞춰 사후 조정하지 않는다. 실험 전에 다음을 preregister한다.

### G0 Problem gate

동일 realistic trace에서 static provisioning이 deadline miss를 만들면서 다른 eligible endpoint compute가 유의미하게 idle이어야 한다.

### G1 Correctness gate

- stale/wrong epoch commit 0
- output mismatch 0
- injected endpoint failure에서 fallback deadline 보장

### G2 Mechanism gate

DART가 strongest feasible software baseline보다 deadline/utilization/application-SLO Pareto front를 개선해야 한다.

### G3 Generality gate

최소 세 background class에서 방향이 유지되어야 한다. Qwen만 성공하면 general AI-RAN claim을 하지 않는다.

### G4 Architecture gate

DART-Q가 software device polling보다 p99 control delay 또는 dedicated polling resource를 줄이고, 합성 overhead가 현실적이어야 한다.

### G5 Radio gate

NRx 선택이 실제 radio utility를 개선해야 한다. 아니면 architecture를 optional generic accelerator stage로 좁히고 AI-RAN radio claim을 낮춘다.

## 17. 1일차 12시간 실행 순서

오늘은 모든 paper sweep을 얕게 돌리는 날이 아니다. claim을 살리거나 죽이는 증거를 먼저 만든다.

| 시간 | 작업 | 산출물 |
|---|---|---|
| 0:00--0:40 | manifest, NIC/MIG 복구, source freeze | `manifest.json` |
| 0:40--1:30 | CUDA device graph/conditional/doorbell probe | capability report |
| 1:30--3:00 | raw TRT/binding/graph Nsys decomposition | NRx critical path |
| 3:00--4:00 | NCU top kernels + service profile | endpoint WCET table v0 |
| 4:00--5:30 | true replica 1/2/4/8 sweep | capacity curve |
| 5:30--7:00 | non-preemption injections on Qwen+training | residual-delay figure |
| 7:00--8:00 | C++ host control vs GPU polling microbench | control-path CDF |
| 8:00--10:00 | multi-cell open-loop same-trace killer run | idle+miss figure |
| 10:00--11:00 | P2P/GDR fair current-payload depth sweep | transport breakdown |
| 11:00--12:00 | analyze, pass/fail G0/H2, next-day freeze | decision memo |

실제 vision/speech model download와 dataset 준비는 GPU profiling과 병렬로 host에서 수행하되, cache/hash가 끝나기 전 결과를 만들지 않는다.

## 18. 전체 campaign 예상 기간

| 기간 | 내용 |
|---|---|
| Day 1 | P0--P3 killer evidence |
| Day 2--3 | P1 최적화와 P4 fair transport |
| Day 4--6 | Host/device DART 구현 및 fault tests |
| Week 2 | realistic workload suite와 full policy sweep |
| Week 3 | DART-Q simulator/RTL/synthesis |
| Week 4 | radio replay, long run, ablation, artifact cleanup |

## 19. 즉시 구현할 파일

```text
cloudlab_aerial/task1/
  dart/
    dart_descriptor.h
    dart_queue.cu
    dart_commit.cu
    dart_host_router.cc
    cuda_graph_capability.cu
    control_path_bench.cu
    dart_trace_sim.py
  nrx_direct_bench.py
  nrx_replica_sweep.py
  background/
    qwen_trace_server.py
    video_trt_server.py
    speech_stream_server.py
    online_train_server.py
  run_isca_problem_sweep.sh
  run_isca_dart_sweep.sh

cloudlab_results/
  results/isca_v2/<run_id>/
    MANIFEST.json
    events.parquet
    summary.csv
    nsys/
    ncu/
```

## 20. 최종 해석 규칙

- transport가 NRx compute보다 작으면 “GDR가 latency를 해결했다”고 쓰지 않는다.
- wrapper 100 ms 수치는 architecture baseline으로 다시 사용하지 않는다.
- static MIG의 isolation win과 elasticity loss를 모두 보고한다.
- DART의 승리는 L1 latency 하나가 아니라 `deadline miss + radio utility + background SLO + idle GPU-seconds`의 Pareto 개선으로 판단한다.
- hardware proposal은 software prototype으로 검증된 semantic gap만 해결한다. 존재하지 않는 bottleneck을 RTL로 장식하지 않는다.
