# DART-Rx final architecture · measurement-backed specification

> **동결된 후보안:** 이 문서는 현재 canonical architecture가 아니다. MIG 중심 문제와
> 진행 중인 causal campaign의 기준은 `MIG_NRX_RESEARCH_CHECKPOINT_KO.md`를 따른다.
> Multi-endpoint fixed-MIG problem existence가 확인되기 전에는 아래 transaction/pool
> mechanism을 최종 설계 또는 novelty로 간주하지 않는다.

Date: 2026-08-13  
Status: single-endpoint vertical slice implemented and validated; multi-endpoint
pool implementation remains the next architecture milestone.

## 1. 결론

정당화 실험 결과는 DART-Rx의 문제와 설계를 다음처럼 고정한다.

> **Real-time L1이 항상 유효한 conventional 결과를 유지하는 동안, radio
> utility가 양수이고 deadline 안에 완료 가능한 요청만 격리된 resident NRx
> pool로 보내고, 원격 결과는 epoch와 expiry가 유효할 때만 한 번 commit한다.**

논문의 핵심은 MIG, MPS, P2P, GDR 중 하나가 더 빠르다는 것이 아니다. 이들은
placement/transport baseline이다. DART-Rx의 architecture novelty 후보는
**utility-bearing, deadline-expiring alternative-result transaction**이다.

이번 결과로 다음 판단은 가능하다.

- `GO`: radio-utility admission
- `GO`: isolated resident NRx service와 GDR data plane
- `GO`: conventional baseline을 보존하는 epoch/expiry single commit
- `GO`: fixed cell-to-NRx가 아니라 predicted-finish pool
- `NO`: 모든 slot에 NRx를 호출하는 설계
- `NO`: MIG/GDR 자체를 novelty로 주장
- `NO`: 현재 결과로 1 ms PHY deadline을 만족한다고 주장
- `NO`: 아직 측정하지 않은 PRB/tile 단위 partial NRx를 설계에 포함

## 2. 근본 문제

### 2.1 한 문장 problem statement

> **선택적으로만 유용한 neural PHY result가 burst할 때, 정적 accelerator
> isolation은 한 endpoint에 deadline miss를 만들면서 다른 endpoint를 놀게
> 하지만, 기존 inference/RDMA interface는 radio utility, result expiry,
> baseline-preserving commit을 표현하지 못한다.**

이 문제는 세 개의 독립 문제를 병렬로 나열한 것이 아니라 하나의 causal
chain이다.

1. NRx utility는 모든 channel condition에서 같지 않다.
2. hard-channel request만 선택하면 request stream은 bursty해진다.
3. fixed placement에서는 burst가 특정 isolated endpoint queue에 집중된다.
4. remote idle endpoint를 사용하려면 transport만이 아니라 late/stale result의
   commit semantics가 필요하다.
5. 따라서 admission, placement, transport, commit이 하나의 transaction이어야
   한다.

### 2.2 측정으로 확인한 problem existence

#### Conditional radio utility

Aerial conventional receiver와 배포된 TensorRT NRx를 동일 TB, 동일 Sionna
Rayleigh realization에 paired 실행했다. MCS 7에서 median incremental success는:

| Es/N0 | conventional BLER | NRx BLER | NRx incremental success |
|---:|---:|---:|---:|
| -4.0 dB | 1.000 | 1.000 | 0.000 |
| -3.8 dB | 1.000 | 0.800 | +0.200 |
| -3.6 dB | 0.527 | 0.046 | +0.477 |
| -3.4 dB | 0.013 | 0.000 | +0.013 |
| -3.2 dB | 0.000 | 0.000 | 0.000 |
| -3.0 dB | 0.000 | 0.000 | 0.000 |

양의 utility는 waterfall 구간에 집중됐다. 따라서 `always NRx`는 radio gain은
추가하지 않으면서 service capacity를 소비할 수 있다.

#### Fixed-placement fragmentation

4 synchronized cells, cell당 1 request/ms, bursty 50% NRx eligibility, 실제
resident TensorRT endpoint 3개를 사용했다.

- static cell placement: timely NRx 실패 비율 41.1--42.8%, p99 27.9--35.5 ms
- predicted-finish pool: timely NRx 실패 비율 0--0.049%, p99 3.48--3.80 ms
- static miss의 98.5--99.5%에서 다른 endpoint가 arrival부터 deadline까지
  compute-idle

즉 문제는 aggregate compute shortage만이 아니라 isolation 아래의 placement
fragmentation이다.

## 3. Overall architecture

```text
RX slot on protected L1
       |
       +--> Aerial conventional receiver ------------------+
       |          always produces the baseline             |
       |                                                   |
       +--> utility + feasibility admission                |
                 | positive and timely                     |
                 v                                         |
          earliest-finish endpoint                         |
                 |                                         |
       GDR payload -> request descriptor -> doorbell        |
                 |                                         |
       resident caller-owned TRT/CUDA Graph NRx             |
                 |                                         |
       GDR result -> completion descriptor -> doorbell      |
                 |                                         |
                 +--> epoch/health/visibility/expiry/CRC ---+
                                      |
                               exactly one commit
```

논문 architecture section은 세 subsection으로만 구성한다. DART-P/F/R/C/J처럼
메커니즘마다 별도 이름을 만들지 않는다.

### 3.1 Utility and feasibility front-end

입력은 `(cell, slot, epoch, MCS, channel-quality, absolute expiry)`이다.

1. paired radio table에서 expected incremental success와 confidence를 찾는다.
2. confidence-positive utility가 아니면 conventional-only로 종료한다.
3. 각 healthy endpoint의 conservative finish를 계산한다.
4. `predicted_finish + commit_guard <= expiry`인 endpoint 중 가장 빠른 곳에
   tensor credit을 원자적으로 reserve한다.
5. feasible endpoint가 없으면 remote work를 만들지 않는다.

현재 utility gate는 measured SNR bin lookup이다. 최종 시스템에서는 SNR 자체가
아니라 gNB가 이미 계산하는 CQI/DMRS quality/decoder history를 사용해야 한다.

### 3.2 Isolated resident receiver fabric

NRx model과 TensorRT context는 endpoint마다 resident 상태를 유지한다. MIG를
동적으로 재구성하거나 L1을 drain하지 않는다.

- endpoint별 RC QP
- registered GPU input/output ring
- caller-owned TensorRT binding
- persistent CUDA Graph
- endpoint service bound와 health epoch
- predicted tail과 tensor credit

request publish order는 같은 RC QP에서 다음과 같다.

1. GPU payload RDMA WRITE
2. 64-byte request descriptor RDMA WRITE
3. 8-byte doorbell RDMA WRITE

completion은 대칭적으로 `result -> completion -> doorbell` 순서다. NIC completion
자체는 L1 commit이 아니다.

### 3.3 Expiring alternative-result commit

Conventional receiver는 optional fallback이 아니라 **항상 보존되는 baseline**이다.
NRx는 baseline을 대체할 수 있는 expiring alternative다.

NRx commit의 필요조건은 모두 동시에 참이어야 한다.

- slot ID와 epoch 일치
- endpoint ID와 health epoch 일치
- registered result payload visibility 보장
- completion status `OK`
- decision 시각이 absolute expiry 이전
- NRx LDPC/CRC 성공
- transaction이 아직 open

Conventional CRC가 먼저 성공하면 즉시 baseline commit한다. Conventional이
실패하면 timely NRx 결과를 기다릴 수 있고, expiry까지 성공 결과가 없으면
ready baseline 또는 miss를 commit한다. late/stale result는 buffer를 overwrite할
수 있어도 architectural state를 변경할 수 없다.

## 4. Wire contract

### 4.1 64-byte request record

- slot ID, epoch, cell ID
- release, expiry, predicted finish
- payload bytes/checksum
- graph ID, tensor class
- quantized utility
- request flags
- endpoint health epoch

### 4.2 64-byte completion record

- slot ID, epoch, endpoint ID
- worker start/finish/publish timestamp
- completion status
- endpoint health epoch
- payload checksum
- CRC metadata
- payload-published and CUDA-Graph flags

Descriptor는 host-registered MR에 있고 bulk tensor는 GPU MR에 있다. GPU MR에는
Pyverbs `MR.read/write`를 호출하지 않는다.

## 5. 실제 구현 상태

### 5.1 구현된 vertical slice

- protected L1: GPU 0의 A100 MIG 4g.20gb
- remote NRx: GPU 1 full A100
- transport: ConnectX-6 Dx GPUDirect RDMA loopback
- input: 실제 Aerial PUSCH TB와 Sionna Rayleigh channel
- baseline: 실제 Aerial `PuschRx`
- NRx: caller-owned TensorRT binding + CUDA Graph
- result: remote LLR -> 실제 Aerial LDPC/CRC
- control: 64-byte request/completion + ordered doorbell
- commit: utility, epoch, health, visibility, expiry, CRC 검사

주요 소스:

- `dart_rx_core.py`: utility admission, endpoint credit, single commit
- `dart_rx_wire.py`: request/completion wire record
- `gdr_rdma_channel.py`: GPU/descriptor/doorbell RC transport
- `dart_rx_radio_l1.py`: protected conventional + remote alternative pipeline
- `dart_rx_nrx_worker.py`: resident direct-TRT worker
- `run_dart_rx_integrated.sh`: reproducible container runner
- `analyze_dart_rx_integrated.py`: paired validation

### 5.2 아직 구현하지 않은 부분

- 하나의 L1 process가 여러 remote QP를 동시에 구동하는 actual multi-endpoint pool
- ring depth > 1의 asynchronous request pipeline
- multi-cell 1 ms arrival stream과 actual GDR payload의 동시 queueing
- ResNet/BERT/Whisper/Qwen background lease와 preemption/drain
- device-resident scheduler 또는 DART-Q hardware queue
- production channel-quality estimator와 online utility adaptation
- real gNB HARQ deadline을 사용한 validation

따라서 현재 결과는 **real integrated single-endpoint transaction**의 증거이며,
완성된 elastic pool의 최종 결과는 아니다.

## 6. Integrated hardware result

최종 paired campaign은 3 trials, trial당 100 requests, MCS 7, Rayleigh
`-3.8/-3.6/-3.4/-3.2 dB`, 12 ms experimental expiry를 사용했다.

### 6.1 Paired radio outcome

하나의 actual all-NRx run에서 얻은 동일 conventional/NRx 결과에 admission
policy를 적용했다. 별도 프로세스 사이 cuPHY decode가 per-slot bit-identical하지
않았기 때문에 이 방식만 paired radio claim에 사용한다.

| policy | median NRx requests / 100 | median correct-TB ratio |
|---|---:|---:|
| conventional baseline | 0 | 0.64 |
| all NRx | 100 | 0.80 |
| DART utility | 75 | 0.80 |

각 trial에서 DART utility는 all-NRx와 동일한 delivered outcome을 유지하면서 NRx
요청을 정확히 25% 줄였고, baseline보다 11--19개의 TB를 추가 복구했다.

### 6.2 Actual mode timing

각 mode는 실제 GDR/NRx/commit path로 독립 실행했다.

| mode | median-of-trials decision p50 | median-of-trials decision p99 | misses |
|---|---:|---:|---:|
| none | 0.885 ms | 1.074 ms | 0/300 |
| all | 3.985 ms | 5.798 ms | 0/300 |
| utility | 3.757 ms | 5.691 ms | 0/300 |

총 900 requests에서 deadline miss는 없었고 admitted remote completion은 모두
12 ms expiry 전에 도착했다. 이 12 ms는 vertical-slice correctness/timing gate이며
1 ms RAN deadline claim이 아니다.

## 7. 왜 이 설계가 필요한가

| 측정된 pain point | 설계 mechanism |
|---|---|
| NRx utility가 waterfall에 집중 | confidence-positive utility admission |
| selective burst와 fixed-placement fragmentation | earliest predicted-finish pool |
| MIG/P2P isolation boundary | registered GDR receiver fabric |
| late result가 PHY state를 오염할 위험 | epoch/health/expiry commit guard |
| NRx가 항상 필요한 것은 아님 | conventional baseline을 항상 유효하게 유지 |
| dynamic MIG가 L1을 drain | resident fixed endpoint, no reconfiguration |

## 8. Novelty boundary

### 8.1 주장할 수 있는 차별점

DART-Rx는 다음 조합을 하나의 command/commit contract로 만든다.

- radio utility-bearing admission
- absolute expiry가 있는 alternative result
- isolated accelerator pool의 predicted-finish placement
- GPU-memory GDR data path
- mandatory conventional baseline을 보존하는 versioned single commit

이 contract가 architecture contribution이어야 한다. 단순 policy 조합이나
Pyverbs 구현은 contribution이 아니다.

### 8.2 아직 ISCA claim으로 부족한 부분

현재 host runtime은 mechanism의 functional prototype다. ISCA급 주장에는 다음이
추가로 필요하다.

1. multi-endpoint actual GDR pool에서 fixed/static 대비 queue stability
2. host polling의 control latency/CPU overhead와 device/DPU/DART-Q 대안
3. ring credit, descriptor cache, expiry compare, commit table의 구체적
   microarchitecture와 area/throughput model
4. 2/4/8-cell arrival와 selective/burst trace
5. background co-tenant 4종 이상에서 L1 p99와 reclaimed throughput
6. endpoint failure, stale completion, delayed payload fault injection
7. 실제 system deadline에서 miss/BLER/throughput Pareto

## 9. 다음 구현 순서

1. 현재 single-endpoint transaction을 `EndpointSession[]` pool로 확장한다.
2. endpoint별 worker thread와 registered ring depth 4--8을 만든다.
3. actual 2/4/8-cell trace를 1 ms period로 replay한다.
4. static-cell, round-robin, shortest-queue, DART predicted-finish를 같은 actual
   GDR path에서 비교한다.
5. queue slack에 background kernels를 lease하고 guard 이전 drain을 검증한다.
6. control-plane Nsight/CPU profile로 host overhead를 분해한다.
7. 이 결과로 DART-Q의 필요한 hardware state와 critical path를 정한다.

## 10. Artifact

- paired utility evidence: `results/20260813_dart_rx_justification/`
- integrated validation: `results/20260813_dart_rx_integrated/analysis/`
- actual mode logs/results: `results/20260813_dart_rx_integrated/dart_rx_*/`
