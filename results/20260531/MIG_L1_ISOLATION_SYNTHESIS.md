# MIG L1 Kernel Isolation Inefficiency Synthesis

Date: 2026-06-01  
Data path: `/Users/changjongkim/New_research/cloudlab_results/results/20260531`

> Update note: this 20260531-focused synthesis is now superseded by the full 20260531-20260601 deep dive at
> `/Users/changjongkim/New_research/cloudlab_results/results/all_deep_dive/MIG_L1_ALL_EXPERIMENTS_DEEP_DIVE.md`.
> The broader dataset changes the interpretation: early cross-partition p99 spikes should be treated as exploratory,
> while 20260601 F/Dv2 show strong cross-partition isolation under saturation. The clearest confirmed failure mode is
> same-partition L1 + NeuralRx co-location, which causes catastrophic bimodal p99 inflation.

## 1. Bottom Line

NVIDIA A100 MIG는 SM/HBM 같은 하드웨어 처리량 자원은 잘 나누지만, cuPHY L1처럼 작은 kernel과 memcpy/convert가 반복되는 latency-critical pipeline의 tail latency는 충분히 격리하지 못한다.

가장 중요한 결론은 다음과 같다.

1. **MIG partition을 작게 만들면 AI가 없어도 L1이 느려진다.**
   - Full GPU 평균 36.38 ms 대비 2g L1 alone 평균 51.01 ms, **+40.2%**
   - p99도 39.18 ms 대비 52.14 ms, **+33.1%**
   - ncu에서는 7g 대비 2g의 MIO throttle이 0.02%에서 0.66%로 **약 33배 증가**

2. **AI co-tenant가 붙으면 3g L1의 mean은 보통 +31~35%, p99는 +65~72% 증가한다.**
   - Qwen, chanpred, xapp는 모두 비슷한 mean degradation
   - NeuralRx는 outlier로 p99가 41.32 ms에서 196.68 ms로 **+376%**

3. **ncu per-kernel metric은 거의 변하지 않는데, nsys timeline의 inter-kernel gap과 memcpy가 커진다.**
   - 즉, L1 kernel 자체의 실행 시간이 망가진다기보다 kernel 사이 대기, CUDA runtime/driver queue, memory-controller queue에서 leakage가 발생한다.
   - `cupy_copy -> convert_kernel`, `convert_kernel -> noiseIntfEst/windowedCh` 전이가 가장 취약하다.

4. **Throughput만 보면 MIG가 잘 격리되는 것처럼 보인다.**
   - AI throughput v2에서 L1 co-tenant 영향은 qwen_small +1.1%, chanpred +0.1%, xapp +2.2%, neuralrx 0%
   - 그러나 per-operation 또는 L1 frame p99로 보면 tail latency leakage가 드러난다.

5. **최종 paper claim은 "MIG의 평균/처리량 격리는 양호하지만 tail latency 격리는 불충분하다"가 가장 정확하다.**
   - 이전의 "L1만 피해 보고 AI는 완전히 격리된다"는 framing은 throughput-only metric이 만든 착시다.

## 2. Data Coverage

로컬 백업에는 텍스트/JSON/CSV/로그 파일 1123개가 있으며, 이 중 JSON 284개, CSV 231개가 있다. 분석에 사용한 주요 데이터 묶음은 다음과 같다.

| Group | Content | Status |
|---|---|---|
| `n20_baseline_*` | Full GPU, 7g, 4g, 3g, 2g L1 baseline | 직접 재집계 |
| `n20_phase*` | Tier1 L1 + AI co-tenant | 직접 재집계 |
| `ai_throughput_v2` | AI throughput with/without persistent L1 | 직접 재집계 |
| `ncu`, `ncu_csv`, `nsight_csv` | Nsight Compute per-kernel metrics | 기존 분석 문서 및 CSV 확인 |
| `nsys_csv_v2`, `nsys_sqlite_v2_analysis` | Tier1-matched kernel timeline | 기존 분석 문서 및 CSV 확인 |
| `nsys_deep_*_analysis` | wall-clock aligned deep dive | CSV 확인 |
| `p4_l1_timeseries`, `p5_sustained` | longer L1 log tests | 직접 재집계 |
| `p7_pdsch_tx` | PDSCH TX attempt | 전 run 실패, API mismatch |
| `ai_per_op_latency`, `l1_multi_ai` | 디렉터리는 있으나 로컬 run 파일 없음 | 기존 `EXPERIMENTS_20260531.md` 수치만 참조 가능 |

## 3. L1 Baseline: MIG Partition Size Penalty

No AI co-tenant인 baseline만 비교하면, 2g L1이 확실한 병목 지점이다.

| L1 Config | N | Mean ms | p95 ms | p99 ms | Mean vs Full | p99 vs Full |
|---|---:|---:|---:|---:|---:|---:|
| Full GPU v2 | 20 | 36.38 | 37.59 | 39.18 | baseline | baseline |
| 7g MIG single | 20 | 37.21 | 37.79 | 38.60 | +2.3% | -1.5% |
| 4g alone | 20 | 39.07 | 39.82 | 40.47 | +7.4% | +3.3% |
| 3g alone | 20 | 40.20 | 40.60 | 41.32 | +10.5% | +5.5% |
| 2g alone | 20 | 51.01 | 51.39 | 52.14 | +40.2% | +33.1% |

해석:

- 7g MIG single은 Full GPU와 거의 비슷하다. MIG mode 자체의 overhead는 작다.
- 4g와 3g는 manageable한 증가지만, 2g는 평균 +40%로 급격히 나빠진다.
- 이 결과는 ncu의 MIO throttle 증가 및 nsys의 2g p99 gap 증가와 일치한다.

## 4. Tier1: L1 Under AI Co-Tenant

3g L1 baseline은 mean 40.20 ms, p99 41.32 ms다. 여기에 AI를 붙이면 대부분의 real AI workload에서 L1 latency가 크게 오른다.

| Scenario | Mean ms | p99 ms | Mean Delta | p99 Delta |
|---|---:|---:|---:|---:|
| qwen7b_decode | 54.23 | 70.76 | +34.9% | +71.2% |
| qwen7b_prefill | 53.80 | 70.20 | +33.8% | +69.9% |
| qwen7b_stress | 52.97 | 69.72 | +31.8% | +68.7% |
| qwen_small | 52.78 | 67.96 | +31.3% | +64.5% |
| chanpred | 53.97 | 71.06 | +34.3% | +72.0% |
| xapp | 52.93 | 68.85 | +31.7% | +66.6% |
| neuralrx | 60.43 | 196.68 | +50.3% | +376.0% |

Phase 2/3 layout sweep도 같은 메시지를 준다.

| Scenario | L1 Size | Mean ms | p99 ms | Relevant Baseline | Mean Delta | p99 Delta |
|---|---|---:|---:|---|---:|---:|
| M1 3-way balanced | 3g | 53.36 | 71.64 | 3g alone | +32.7% | +73.4% |
| M2 3-way L1 small | 2g | 65.98 | 84.45 | 2g alone | +29.4% | +62.0% |
| M3 3-way asym | 4g | 53.96 | 70.90 | 4g alone | +38.1% | +75.2% |
| M4 4-way 1L1+3AI | 4g | 53.21 | 73.53 | 4g alone | +36.2% | +81.7% |
| D1 L1 starved | 2g | 66.71 | 82.89 | 2g alone | +30.8% | +59.0% |
| D2 L1 boosted | 4g | 53.67 | 73.10 | 4g alone | +37.4% | +80.6% |

해석:

- AI가 붙는 순간 3g/4g L1 모두 mean이 53 ms 근처로 모인다.
- 2g L1은 co-tenant 이전 baseline 자체가 이미 나쁘고, AI가 붙으면 66 ms 수준으로 악화된다.
- 4g로 키우는 것은 2g catastrophe는 피하게 하지만, Tier1 wall-clock leakage 자체를 제거하지는 못한다.

## 5. AI Throughput: Throughput Isolation Looks Good

AI throughput v2는 persistent L1 wrapper를 배경으로 켜고 AI throughput을 30초 단위로 잰 결과다.

| AI | Alone | With L1 | Delta |
|---|---:|---:|---:|
| qwen_small | 46.68 it/s | 47.18 it/s | +1.07% |
| chanpred | 2248.80 pred/s | 2251.60 pred/s | +0.12% |
| xapp | 1820.00 inf/s | 1859.60 inf/s | +2.18% |
| neuralrx | 9.00 inf/s | 9.00 inf/s | 0.00% |

해석:

- throughput 관점에서는 MIG 격리가 매우 좋아 보인다.
- 하지만 이것은 30초 적분 평균이라 burst/tail event가 희석된다.
- 따라서 AI-RAN co-location 판단에는 throughput이 아니라 p95/p99/p999 latency를 같이 봐야 한다.

## 6. NCU: Kernel Execution Itself Is Mostly Isolated

Nsight Compute는 16개 scenario에서 28개 metric을 수집했다. 핵심은 다음과 같다.

Partition size effect:

| Metric | 7g S2 | 3g S5 | 2g S10 | Interpretation |
|---|---:|---:|---:|---|
| L2 traffic per kernel | 12.27 | 377.86 | 289.49 | 작은 partition에서 L2 activity 급증 |
| MIO throttle | 0.02% | 0.13% | 0.66% | 7g 대비 2g 약 33배 |
| SM throughput | 1.65% | 3.69% | 5.08% | 작은 partition에서 각 SM 부담 증가 |
| GPU time/kernel | 8.19 us | 7.97 us | 8.68 us | per-kernel 자체는 큰 변화 아님 |

3g L1 + AI effect:

- L2 hit rate 변화는 대체로 0.4% 미만
- MIO throttle은 AI 종류가 바뀌어도 0.13% 근처로 유지
- GPU time/kernel은 Qwen, NeuralRx, 3AI에서도 약 +1~2% 수준

결론:

- ncu만 보면 "AI co-tenant가 L1 kernel 내부를 거의 방해하지 않는다"가 맞다.
- 그런데 Tier1 wall-clock은 +30~50% 증가한다.
- 따라서 leakage 위치는 kernel 내부가 아니라 kernel 사이 scheduling/gap/memcpy/runtime 쪽이다.

## 7. NSYS: Leakage Is In Inter-Kernel Gaps And Memory Ops

Tier1-matched nsys v2는 16 scenarios x 3 runs, 총 48 traces다. S5 3g alone과 비교하면 다음 패턴이 나온다.

| Scenario | p99 Gap Delta | Total Time Delta | Idle Time Delta | Interpretation |
|---|---:|---:|---:|---|
| S6 3g + Qwen | +13.2% | +9.5% | +12.2% | chaotic AI, moderate leakage |
| S7 3g + NeuralRx | +22.0% | +16.0% | +20.5% | strongest single AI leakage |
| S9 3g + 3AI | +1.3% | +10.2% | +13.1% | p99 gap muted, total still up |
| S13 3g + sat_compute | +3.0% | -0.6% | -0.8% | uniform compute, mostly safe |
| S14 3g + sat_hbm | +7.0% | +2.5% | +3.2% | bandwidth workload, mild |
| S24 3g + 2sat | +0.9% | +4.7% | +6.0% | mild |

가장 취약한 transition:

| Transition | Count | Qwen | NeuralRx | 3AI | sat_compute |
|---|---:|---:|---:|---:|---:|
| `cupy_copy_complex64 -> convert_kernel` | 3840 | +58.7% | +398.6% | +499.3% | -24.8% |
| `convert_kernel -> windowedCh` | 1920 | +22.6% | +134.2% | +136.7% | +15.4% |
| `convert_kernel -> noiseIntfEst` | 1920 | +26.2% | +291.5% | +509.0% | +1.5% |

Memory operation evidence:

| Scenario | memcpy Total | memcpy Median | memcpy p99 |
|---|---:|---:|---:|
| S5 3g alone | 140 ms | 4.19 us | 32.93 us |
| S6 + Qwen | 394 ms, +181% | 10.24 us, +144% | 164.77 us, +400% |
| S7 + NeuralRx | 585 ms, +317% | 14.30 us, +241% | 170.72 us, +418% |
| S13 + sat_compute | 272 ms, +94% | 6.78 us, +61% | 163.90 us, +398% |

해석:

- L1 pipeline에서 `cupy_copy`와 `convert_kernel` 주변이 가장 취약하다.
- 특히 NeuralRx/3AI처럼 kernel launch와 memory operation 패턴이 복잡한 workload는 transition gap을 3~6배까지 키운다.
- memcpy p99는 거의 모든 AI에서 4~5배 튀지만, 최종 L1 frame latency로 누적되는 정도는 workload pattern에 따라 달라진다.

## 8. Deep-Dive: Wall-Clock Alignment And Root Cause

`nsys_deep_*` 분석은 L1과 AI를 wall-clock으로 맞춰서 burst의 동시성을 본다.

Dual-concurrent table:

| Scenario | L1 Kernels | p99 Gap | p999 Gap | Max Gap | Top 1% Idle Share |
|---|---:|---:|---:|---:|---:|
| A1 S35 2g L1 + chanpred 3g | 48976 | 1564 us | 1612 us | 118.8 ms | 26.3% |
| A2 S34 4g L1 + ResNet 2g | 48976 | 811 us | 840 us | 106.5 ms | 28.7% |
| A3 M5c 3g L1 | 48976 | 870 us | 1014 us | 117.4 ms | 28.0% |
| A4 M8a 3g L1 | 48976 | 980 us | 1185 us | 116.1 ms | 29.3% |

핵심:

- A1 worst case는 A2 robust case보다 p99 gap이 약 93% 크다.
- top 1% gap이 idle time의 26~29%를 차지한다. 즉 평균이 아니라 rare burst가 전체 tail을 지배한다.
- burst는 반복적으로 `convert_kernel -> windowedChEstPreNoDftSOfdmKernel` 근처에서 발생한다.

Phase decomposition:

| Phase | p99 Gap | p999 Gap | p999 Delta |
|---|---:|---:|---:|
| D0 3g L1 alone | 977.6 us | 994.5 us | baseline |
| D1 memcpy 8MB | 976.3 us | 1780.7 us | +79.1% |
| D2 memcpy 64MB | 863.0 us | 1645.1 us | +65.4% |
| D3 compute-only GEMM | 812.0 us | 1623.1 us | +63.2% |
| D4 launch storm | 941.7 us | 1217.4 us | +22.4% |
| D5 chanpred full | 939.5 us | 954.2 us | -4.1% |

해석:

- launch storm만으로도 tail을 키우지만, memcpy/compute phase가 p999를 더 크게 키운다.
- 따라서 root cause는 단순히 kernel launch queue 하나가 아니라 memory-controller queue contention과 AI compute/memory activity가 함께 만드는 burst로 보는 것이 더 정확하다.

## 9. Deeper Interpretation: Static Capacity Isolation Is Not Temporal Bandwidth Isolation

MIG의 핵심 가치는 정적 자원 격리다. SM, HBM 용량, L2 slice, memory partition 같은 자원을 GPU instance 단위로 나눠서 한 tenant가 다른 tenant의 capacity를 직접 빼앗지 못하게 한다. 이 점은 AI throughput 결과에서 분명히 보인다. sat_compute, sat_hbm, ResNet, Qwen, chanpred 같은 workload의 평균 처리량은 L1이 같이 있어도 거의 변하지 않는다.

하지만 L1 latency 문제는 capacity isolation만으로 설명되지 않는다. L1은 평균 처리량보다 "언제 다음 kernel이 시작되는가"가 중요하고, 이때 필요한 것은 정적 capacity가 아니라 짧은 순간의 burst bandwidth와 queueing delay 보장이다.

### 9.1 Resource Layers That Matter

MIG 격리를 해석할 때 자원을 한 덩어리로 보면 안 된다. 이번 데이터는 최소 네 층을 구분해야 한다.

| Layer | MIG Isolation | What The Data Says | Why It Matters For L1 |
|---|---|---|---|
| SM capacity | strong static partitioning | AI throughput mostly stable | 평균 compute 처리량은 잘 격리됨 |
| HBM capacity / steady bandwidth | mostly partitioned and predictable | sat_hbm throughput scales cleanly by partition | 장시간 평균 bandwidth는 잘 나뉨 |
| L2 / memory-subsystem effective bandwidth | partitioned but smaller slice means lower headroom | 2g alone has +40% mean latency and much higher MIO throttle | L1 working set and copy/convert stages become sensitive |
| Runtime / copy / memory queue bandwidth | not exposed as a hard real-time reservation | memcpy p99 +300~400%, inter-kernel gap +100~500% in vulnerable transitions | tail latency leaks through burst queueing |

즉, "MIG가 L2와 HBM을 나누니까 격리된다"는 말은 평균 처리량 관점에서는 맞지만, "L1의 p99/p999 latency도 격리된다"는 뜻은 아니다. L1에는 각 slot/frame마다 필요한 순간 대역폭이 있고, 그 순간에 copy/convert/channel-estimation boundary가 AI activity와 겹치면 queueing delay가 생긴다.

### 9.2 Bandwidth Is The Central Bottleneck, But Not Only One Number

이 실험에서 bandwidth는 가장 중요한 설명 축이다. 다만 bandwidth를 HBM GB/s 하나로만 보면 부족하다.

1. **Fixed total device bandwidth**: GPU 하나의 총 메모리/내부 fabric bandwidth는 고정되어 있다. MIG는 이를 동적으로 늘리지 못하고 정적으로 slice한다.
2. **Partition-local bandwidth headroom shrinks**: 2g처럼 작은 partition은 자기 몫의 L2/memory pipeline headroom이 작다. 그래서 AI가 없어도 L1 baseline이 악화된다.
3. **Burst bandwidth is not reserved**: L1은 평균 bandwidth보다 순간 burst가 중요하다. `cupy_copy -> convert_kernel` 같은 boundary에서 짧은 시간 안에 copy/format conversion/next cuPHY kernel launch가 이어지는데, 이 구간에 AI activity가 겹치면 p99 gap이 튄다.
4. **Queue bandwidth can dominate raw bandwidth**: sat_hbm처럼 raw HBM bandwidth를 쓰는 workload는 throughput isolation이 좋아 보이지만, nsys에서는 memcpy p99가 크게 오른다. 이는 "GB/s가 부족하다"라기보다 memory operation이 queue에서 대기하는 시간이 tail을 만든다는 뜻이다.

그래서 이번 결과는 "MIG가 bandwidth를 전혀 격리하지 못한다"가 아니라, 더 정확히는 다음과 같다.

> MIG는 장시간 평균 bandwidth와 capacity는 어느 정도 정적으로 격리하지만, latency-critical L1이 필요로 하는 순간 burst bandwidth, memory-operation queueing, kernel transition timing까지는 SLA 형태로 격리하지 못한다.

### 9.3 Why Static Isolation Requires Workload Knowledge

MIG partition은 정적이다. 한 번 3g/2g/1g로 나누면 각 workload가 실제로 언제 burst를 내는지, 어떤 kernel transition이 민감한지, 어떤 phase가 memcpy-heavy인지에 따라 동적으로 조절하지 않는다.

이번 데이터에서 workload knowledge가 필요한 이유는 명확하다.

- **ResNet/sat_compute/sat_hbm**: 평균 throughput은 안정적이고 L1 leakage도 상대적으로 작거나 stochastic하다.
- **Qwen/chanpred/NeuralRx/xapp**: kernel launch, memcpy, irregular memory access가 많아 L1 tail을 크게 흔든다.
- **NeuralRx**: L1과 유사한 PHY-NN 성격 때문에 p99 outlier가 가장 심하다.
- **2g L1**: co-tenant 이전부터 headroom이 부족하므로 어떤 AI를 붙이든 SLA 위험이 커진다.

따라서 "L1은 3g, AI는 2g" 같은 정적 배치 규칙만으로는 부족하다. 실제 운영에서는 최소한 다음 정보를 알아야 한다.

- AI workload의 memcpy/copy frequency
- kernel launch rate와 kernel size distribution
- burst phase가 L1의 copy/convert/channel-estimation boundary와 겹치는지
- L1 partition의 p99/p999 headroom
- 평균 throughput이 아니라 per-frame/per-op tail latency

### 9.4 The Clear Problem Statement

이번 실험이 드러내는 MIG 비효율성은 세 가지로 정리된다.

1. **Fragmentation inefficiency**: L1을 작은 MIG partition에 넣으면 AI가 없어도 L2/memory-subsystem headroom이 줄어 latency가 증가한다.
2. **Tail isolation failure**: 평균 처리량은 격리되어도 copy/convert/kernel transition의 p99/p999 gap은 co-tenant activity에 의해 커진다.
3. **Static placement mismatch**: MIG는 workload의 시간적 burst pattern을 모르고 정적으로 자원을 나누므로, 실제 AI-RAN workload 조합에서는 안전한 배치를 사전에 알기 어렵다.

이 세 가지를 합치면 paper의 핵심 문제 제기는 다음처럼 갈 수 있다.

> MIG turns one GPU into multiple capacity-isolated devices, but it does not turn one GPU into multiple independently bandwidth-scheduled real-time devices. For AI-RAN, the missing abstraction is temporal bandwidth isolation: predictable burst access to memory, copy, and launch paths at L1 kernel boundaries.

## 10. P4/P5 Follow-Up Logs

P4 timeseries, P5 sustained는 run 수가 적어 paper-grade main evidence보다는 sanity check에 가깝다.

P4, 500 iterations:

| Co-tenant | Mean ms | p99 ms | Mean Delta | p99 Delta |
|---|---:|---:|---:|---:|
| alone | 38.43 | 62.79 | baseline | baseline |
| neuralrx | 41.67 | 114.31 | +8.4% | +82.1% |
| xapp | 42.18 | 95.14 | +9.7% | +51.5% |
| sat_compute | 39.20 | 87.58 | +2.0% | +39.5% |
| chanpred | 39.60 | 63.72 | +3.0% | +1.5% |

P5, 7500 iterations:

| Co-tenant | Mean ms | p99 ms | Mean Delta | p99 Delta |
|---|---:|---:|---:|---:|
| alone | 37.78 | 45.10 | baseline | baseline |
| sat_compute | 40.60 | 88.60 | +7.5% | +96.5% |
| chanpred | 39.78 | 48.16 | +5.3% | +6.8% |
| qwen_small | 39.92 | 45.21 | +5.7% | +0.3% |
| neuralrx | 39.57 | 42.65 | +4.7% | -5.4% |

해석:

- P4에서는 NeuralRx/xapp의 p99 tail이 다시 두드러진다.
- P5에서는 sat_compute p99가 크게 튀지만 run 수가 2개라 stochastic burst 가능성이 크다.
- main claim은 Tier1 N=20, nsys v2/v3, deep-dive 쪽을 우선해야 한다.

## 11. Failed Or Missing Data

- `p7_pdsch_tx`: 45개 log 모두 `get_tb_size() got an unexpected keyword argument 'num_prb'`로 실패했다. 이 데이터는 MIG 격리 결론에 사용하면 안 된다.
- `ai_per_op_latency`, `ai_per_op_latency_b`, `l1_multi_ai`: 로컬 백업 디렉터리는 존재하지만 run/log/json 파일이 비어 있다. 다만 `EXPERIMENTS_20260531.md`에는 Stage 4 및 multi-AI 결과가 요약되어 있다.
- `n20_baseline_fullGPU`: JSON은 old/no-mig 50-iteration 결과로 보이며, 5/31 baseline 비교에는 `n20_baseline_fullGPU_v2` log가 맞다.

## 12. Recommended Paper Framing

### Main Claim

> A100 MIG provides useful static capacity isolation for SMs, memory capacity, and steady-state throughput, but it does not provide temporal bandwidth isolation for a latency-critical cuPHY L1 pipeline. The failure is not primarily due to slowdown inside individual L1 kernels. Instead, cross-partition AI activity consumes burst headroom in memory-operation, copy, runtime, and transition paths, inflating inter-kernel gaps around copy/convert/channel-estimation boundaries and producing rare but severe tail delays.

### Evidence Chain

1. **Baseline partition penalty**: 2g L1 alone is +40% mean vs Full GPU and has much higher MIO throttle.
2. **Application-level leakage**: 3g L1 + AI causes +31~50% mean and +65~376% p99 latency.
3. **Throughput illusion**: AI throughput changes by only 0~2.2%, hiding tail effects.
4. **ncu negative evidence**: per-kernel L2/DRAM/MIO metrics barely change with AI, so kernel execution is mostly isolated.
5. **nsys positive evidence**: inter-kernel gaps and memcpy p99 inflate heavily, especially around `cupy_copy -> convert_kernel` and `convert_kernel -> cuPHY` transitions.
6. **deep-dive causal evidence**: top 1% L1 bursts align with AI GPU activity in wall-clock time, and memory/compute phases raise p999 more than launch-storm alone.
7. **static placement limitation**: safe MIG partitioning requires workload-specific knowledge of burst timing, memcpy frequency, and kernel transition sensitivity; static capacity slices alone are not enough.

## 13. Practical Recommendation

For AI-RAN deployment:

- Avoid putting latency-critical L1 on 2g MIG. Its standalone overhead is already too high.
- Do not rely on AI throughput isolation as evidence of real-time safety.
- Treat bandwidth as a temporal SLA problem, not only a GB/s capacity problem.
- Treat NeuralRx/LLM/LSTM-like irregular workloads as high-risk co-tenants.
- Profile candidate AI workloads for memcpy frequency, kernel launch rate, and p99/p999 gap impact before co-locating them with L1.
- If strict p99/p999 SLA is required, isolate L1 on a separate physical GPU or reserve a larger MIG partition and enforce AI quiet windows.
- Use nsys gap/memcpy tail metrics as the primary diagnostic, not only ncu per-kernel metrics.
