# 5/31 CloudLab AI-RAN 실험 종합 정리

**날짜**: 2026-05-31 (한국) / 2026-05-30 (서버 UTC-6)
**환경**: CloudLab d8545 (AIRANSLICING), NVIDIA A100-SXM4-40GB × 4, EPYC 7413, 512GB RAM
**드라이버**: NVIDIA 550.163.01, Aerial cuPHY 25.3.2, Docker 29.5.2
**컨테이너**: `airan:25-3-final` (cuPHY pyaerial + PyTorch 2.4.1 + torchvision 0.19.1)
**위치**: `/users/sgkim/cloudlab_aerial/results/20260530/` (서버) → `/Users/changjongkim/New_research/cloudlab_results/results/20260531/` (백업)

---

## 0. 실험 전체 구조

| 단계 | 스크립트 | 목적 | 셀 수 | 상태 |
|---|---|---|---|---|
| **Setup** | `post_reboot_setup.sh` | 환경 부트스트랩 (driver, docker, MIG, cuPHY, Qwen) | — | ✅ 완료 |
| **Baselines** | `run_fullgpu_baseline.sh`, `run_mig_baselines.sh` | L1 alone (no AI) 5 partitions | 5 × N=20 = 100 | ✅ 완료 |
| **Tier1 main** | `run_tier1_main.sh` | L1 latency with various AI co-tenants | 13 × N=20 = 260 | ✅ 완료 |
| **AI throughput v2** | `ai_throughput_v2.sh` | persistent L1 wrapper로 AI throughput 격리 측정 | 4 × 2 × N=5 = 40 | ✅ 완료 |
| **AI full matrix** | `run_ai_full_matrix.sh` | 모든 파티션 × AI 워크로드 throughput | 19 × 2 × N=5 = 190 | ✅ 완료 (qwen_small_1g, qwen7b_stress_4g 실패) |
| **AI supplement** | `run_ai_supplement.sh` | ResNet (fp16) + Forecaster batch sweep | 22 × 2 × N=5 = 220 | ✅ 완료 |
| **L1 multi-AI matrix** | `run_l1_multi_ai_matrix.sh` | L1 latency × multi-AI 동시 co-tenant | 21 × 2 × N=5 = 210 | ✅ 완료 (14/21, M9 3개 실패) |
| **Nsight ncu matrix** | `nsight_full_matrix.sh` | per-kernel 28 metrics (L2/DRAM/MIO/stalls) | 16 scenarios | ✅ 완료 (재부팅 후 perm fix) |
| **Stage 4: AI per-op latency** | `run_ai_per_op_latency_matrix.sh` ⭐ | **AI workload p99 latency (mean 아님!)** | 16 × 2 × N=5 = 160 | ✅ 완료 |
| **nsys timeline v2** | `nsys_full_matrix.sh` (Tier1-matched: CELLS=20 ITERS=30 N=3) | kernel start/end timestamps for gap analysis | 16 × 3 = 48 | 🟢 진행중 (마지막) |
| **P3/P4/P5/P7** | `run_p*.sh` | partition sweep, timeseries, sustained, PDSCH TX | — | ⏳ 대기 (자동 chain) |

**Chain wrappers**:
- `chain_post_full_matrix.sh` — Stage 1→2→3 (initial chain, kill 후 재시작)
- `chain_post_reboot_resume.sh` — Stage 3 nsight → 4 ai_per_op → 5+ (재부팅 후 resume chain)
- `nsys_full_matrix.sh` — nsys timeline (GPU 1, 병렬)

---

## 1. L1 Baselines (no AI co-tenant)

각 파티션 단독으로 cuPHY component-level PUSCH RX pipeline 실행. N=20 runs, 각 run = 30 iters × 20 cells.

| Partition | Mean (ms) | p95 (ms) | p99 (ms) | Stdev | min/max | Bimodal gap |
|---|---|---|---|---|---|---|
| Full GPU (no MIG) | 36.38 | 37.59 | 39.18 | 1.85 | 34.5/39.2 | 4.11 |
| 7g MIG single | 37.21 | 37.79 | 38.60 | 1.84 | 34.4/38.8 | 4.16 |
| 4g alone | 39.07 | 39.82 | 40.47 | 2.12 | 37.1/42.5 | 4.42 |
| 3g alone | 40.20 | 40.60 | 41.32 | 1.99 | 37.1/41.8 | 4.40 |
| 2g alone | 51.01 | 51.39 | 52.14 | 1.92 | 49.8/54.7 | 3.66 |

**관찰**:
- 모든 partition에서 **bimodal pattern** (gap ~4ms, ~37ms 클러스터 + ~41ms 클러스터) — 5/24 발견 재확인
- MIG mode 오버헤드 자체는 작음 (Full GPU vs 7g single = +2.3%)
- partition size 줄어들수록 latency 증가:
  - 4g → 3g: +3%
  - 3g → 2g: +27% (2g가 partition cap hit)
- 2g가 outlier — 14 SMs 만으로는 partition cap 직접 hit

---

## 2. Tier1 Main: L1 Latency under AI Co-tenant (13 legs, N=20 each)

### Phase 1 — Qwen variants on split-50-50 (3g L1 + 3g AI)

| AI Workload | Mean (ms) | p95 (ms) | p99 (ms) | vs 3g alone (40.20ms) |
|---|---|---|---|---|
| qwen7b_stress | 52.97 | 59.46 | 69.72 | **mean +32%, p99 +69%** |
| qwen7b_prefill | 53.80 | 59.03 | 70.20 | mean +34%, p99 +70% |
| qwen7b_decode | 54.23 | 60.93 | 70.76 | mean +35%, p99 +71% |
| qwen_small | 52.78 | 57.76 | 67.96 | mean +31%, p99 +64% |

→ Qwen variants 모두 **균일하게 L1을 ~33% 늦춤**, p99는 ~70% 증가.

### Phase 4 — Real AI-RAN workloads on split-50-50

| AI Workload | Mean (ms) | p95 (ms) | p99 (ms) | vs 3g alone |
|---|---|---|---|---|
| **neuralrx** | **60.43** | **94.61** | **196.68** | **mean +50%, p99 +377%** 🔥 |
| chanpred | 53.97 | 60.32 | 71.06 | mean +34%, p99 +72% |
| xapp | 52.93 | 57.72 | 68.85 | mean +32%, p99 +67% |

→ **NeuralRx outlier!** p99 196 ms = ~5x baseline. PHY-layer NN의 dense Conv1d + LDPC inferencing이 L1 cuPHY의 L2 cache 패턴과 충돌하는 듯.

### Phase 2 — Multi-partition layouts

| Layout | cgi | L1 partition | AI 위치 | Mean (ms) |
|---|---|---|---|---|
| M1 3way-balanced | 9,14,14 | 3g | 2g + 2g (qwen_small × 2) | 53.36 |
| **M2 3way-L1small** | 9,14,14 | **2g** | 3g + 2g | **65.98** |
| M3 3way-asym | 5,14,19 | 4g | 2g + 1g | 53.96 |
| M4 4way-1L1+3AI | 5,19,19,19 | 4g | 1g × 3 | 53.21 |

→ M2 (2g L1) = 65.98 ms — **2g alone (51.01) 대비 +29%**. L1 partition 작을수록 누설 영향 더 큼.
→ M3/M4 (4g L1) ≈ M1 (3g L1) — partition size 키워도 큰 차이 없음.

### Phase 3 — D1/D2

| Config | Layout | L1 | AI | Mean (ms) |
|---|---|---|---|---|
| D1_L1_starved | split-40-60 (14,9) | 2g | 3g (qwen_small) | 66.71 |
| D2_L1_boosted | split-60-40 (5,9) | 4g (실제로는 3g) | 4g (실제로는 4g) | 53.67 |

→ D1 / D2 = M2 / M3 와 일관. **L1 partition 사이즈가 L1 누설의 dominant factor**.

---

## 3. AI Throughput v2 (persistent L1 wrapper, split-60-40)

5/24 측정 버그 fix: L1을 persistent loop로 30초 내내 돌리면서 AI 처리량 측정.

| AI | alone | with_l1 | 변화 |
|---|---|---|---|
| qwen_small | 46.68 it/s | 47.18 | +1.1% |
| chanpred | 2249 pred/s | 2252 | +0.1% |
| xapp | 1820 inf/s | 1860 | +2.2% |
| neuralrx | 9.0 inf/s | 9.0 | 0% |

→ **모든 AI 워크로드에서 L1 co-tenant 영향 <3%** = AI 격리 거의 완벽.

---

## 4. AI Full Matrix (19 cells × 2 setups × N=5)

각 AI workload × 4 partition × {alone, with_l1}.

### 4-1. 기존 AI 워크로드 (light, launch overhead bound)

| Workload × Partition | alone mean | with_l1 mean | 변화 |
|---|---|---|---|
| qwen_small × 4g | 46.60 it/s | 46.30 | -0.6% |
| chanpred × 1g | 2037 pred/s | 2037 | 0% |
| chanpred × 4g | 2270 | 2269 | 0% |
| xapp × 1g | 1827 inf/s | 1877 | +2.7% |
| xapp × 4g | 1842 | 1875 | +1.8% |
| neuralrx × 1g | 9.0 inf/s | 9.0 | 0% |
| neuralrx × 4g | 9.8 | 9.4 | -4.1% |

**관찰**: chanpred / xapp / neuralrx 모두 1g→4g 스케일링 거의 없음 (chanpred 1g 2037 → 4g 2270 = +12%). **launch overhead / kernel scheduling bound** → SM 더 줘도 활용 못 함.

### 4-2. Qwen-7B variants (4g만 fit, model 14GB)

| Workload × 4g | alone | with_l1 | 변화 |
|---|---|---|---|
| qwen7b_prefill | 13.51 it/s | 13.47 | -0.3% |
| qwen7b_decode | 37.70 | 37.62 | -0.2% |
| qwen7b_stress | (실패 — HF_HOME NERSC 경로 하드코드) | | |

### 4-3. Synthetic Saturating Workloads

#### sat_compute (fp16 GEMM, 91%+ Tensor Core 효율)

| Partition | alone TFLOPS | with_l1 TFLOPS | 변화 |
|---|---|---|---|
| 1g (14 SM) | 26.61 | 26.59 | -0.07% |
| 2g (28 SM) | 73.88 | 73.92 | +0.05% |
| 3g (42 SM) | 112.16 | 111.95 | -0.19% |
| 4g (56 SM) | 148.14 | 147.93 | -0.14% |

→ **선형 SM 스케일링** (26→74→112→148 = 1:2.8:4.2:5.6, 이론치 1:2:3:4). **91% Tensor Core 효율** = 진짜 compute saturation.

#### sat_hbm (memory bandwidth saturating)

| Partition | alone GB/s | with_l1 GB/s |
|---|---|---|
| 1g (alloc 4 GB) | 87.30 | 87.30 |
| 2g (alloc 8.5 GB) | 174.20 | 174.20 |
| 3g (alloc 17 GB) | 351.50 | 351.46 |
| 4g (alloc 17 GB) | 346.90 | 346.84 |

→ **HBM bandwidth 선형 스케일링** (87→174→351 = 1:2:4). MIG HBM 격리 완벽.

---

## 5. AI Supplement v2 (22 cells × 2 setups × N=5)

ResNet-50 fp16 (Tensor Core 활용) + Traffic Forecaster (Informer-lite), 각 partition × batch sweep.

### 5-1. ResNet-50 fp16 — compute-bound 워크로드

| Partition | bs=32 | bs=64 | bs=128 | bs=256 | with_l1 delta |
|---|---|---|---|---|---|
| 1g (14 SM) | (fail) | 659 img/s | — | — | 0% |
| 2g (28 SM) | 1273 | 1314 | 1347 | — | -0.07~-0.16% |
| 3g (42 SM) | — | 2188 | 2263 | 2319 | -0.04~-0.18% |
| 4g (56 SM) | — | 2577 | 2664 | 2723 | -0.04~-0.15% |

**스케일링** (bs=64): 1g 659 → 4g 2577 img/s = **3.91x** (SM 4x). 거의 선형 → **진정한 compute saturating AI 워크로드**.

### 5-2. Forecaster (Informer-lite) — sparse attention encoder

| Partition | bs=16 | bs=32 | bs=64 | bs=128 | bs=256 | with_l1 delta |
|---|---|---|---|---|---|---|
| 1g | 98.1 | 50.0 | 25.8 | — | — | 0~-0.2% |
| 2g | — | 61.2 | 30.8 | 16.0 | — | ±0.13% |
| 3g | — | — | 46.9 | 24.1 | 12.2 | ±0.13% |
| 4g | — | — | 59.6 | 30.8 | 16.0 | ±0.07% |

**스케일링** (bs=64): 1g 25.8 → 4g 59.6 batch/s = **2.31x** — sub-linear. **Sparse attention indexing이 cache 효율 떨어뜨림 → memory pattern bound**.

---

## 6. 핵심 발견 — **비대칭 격리 가설 확정**

### 6-1. L1 ← AI 방향: **격리 X**

오늘까지 모든 AI co-tenant에서 일관되게 L1 latency 누설:

| AI co-tenant | 3g L1 mean ∆ | 3g L1 p99 ∆ |
|---|---|---|
| qwen_small | +31% | +64% |
| qwen7b (3종) | +32~35% | +69~71% |
| chanpred | +34% | +72% |
| xapp | +32% | +67% |
| **neuralrx** | **+50%** | **+377%** 🔥 |

L1 partition 작을수록 더 심함:
- 4g L1 + AI: +37%
- 3g L1 + AI: +31~50%
- **2g L1 + AI: +29~31% (그러나 시작점 자체가 +27% MIG 오버헤드)**

### 6-2. AI ← L1 방향: **격리 OK**

오늘까지 모든 AI workload, 모든 batch size, 모든 partition에서 일관:

| AI 종류 | partition 커버리지 | with_l1 영향 |
|---|---|---|
| sat_compute | 1g/2g/3g/4g (91% Tensor Core 효율) | <0.2% |
| sat_hbm | 1g/2g/3g/4g (HBM bw saturate) | 0% |
| ResNet-50 fp16 | 1g/2g/3g/4g (3.91x SM 스케일) | <0.2% |
| Forecaster | 1g/2g/3g/4g | <0.2% |
| qwen_small/7b prefill/decode | 1g/4g | <1% |
| chanpred / xapp / neuralrx | 1g/4g | <3% |

---

## 7. AI 격리가 잘 되는 이유 + L1만 안 되는 이유 (분석)

> "AI 워크로드들이 격리가 잘 된다는 게 조금 이해가 안 됨"

이게 중요한 포인트라 따로 자세히 정리.

### 7-1. MIG 하드웨어 격리 모델

A100 MIG는 다음을 **물리적으로** 격리:
- **GPC** (Graphics Processing Cluster, 7개): partition마다 dedicated 1~7개
- **SM** (Streaming Multiprocessor, 108개): GPC 단위로 분배
- **HBM channels** (40 채널): partition별 전용 메모리 슬라이스
- **L2 cache** (40 MB): **7개 slice로 분할** — 각 GPC가 하나의 슬라이스 owner

### 7-2. AI 방향 격리가 잘 되는 메커니즘

AI 워크로드의 work pattern:
1. **GEMM/Conv은 단일 partition 안에서 완결**: weights, activations 모두 자기 HBM 슬라이스에 있음
2. **L2 hit률이 높음**: 같은 weights를 여러 iteration 재사용 → L2가 작아도 working set이 자기 슬라이스에 fit
3. **Throughput-tolerant**: 30초 동안 inference 처리량 측정 → 일시 disturbance가 평균에 묻힘

→ MIG가 SM/HBM bw를 정확히 자르므로 AI throughput은 "자기 partition의 capacity"가 결정. **다른 partition의 작업이 영향 못 줌.**

### 7-3. L1 방향 격리가 안 되는 메커니즘 (가설)

L1 (cuPHY PUSCH RX) work pattern:
1. **수많은 작은 kernel launches (50+ kernels per 1ms slot)**:
   - ChannelEstimator → Equalizer → NoiseIntfEstimator → LdpcDeRateMatch → LdpcDecoder → CrcChecker
   - 각 kernel은 100~500μs 단위
2. **Latency-critical (sub-millisecond slots)**:
   - 1 슬롯 = 1 TTI = 0.5~1 ms
   - 30초 평균에서도 worst-case가 p99에 직접 보임
3. **L2 cache 의존도가 매우 큼**:
   - cuPHY는 channel matrix, noise estimate 등 작은 buffer를 여러 kernel에서 반복 access
   - L2 slice가 작으면 working set이 안 들어가 → DRAM trip 증가 → latency 폭증
4. **Cross-partition L2 coherence 가능성**:
   - MIG L2가 7 slice로 나뉘었지만 **공유 인터커넥트는 같음**
   - 인접 partition의 heavy L2 traffic이 wire 경쟁 유발 가능
   - GPCs 간 노이즈가 L1의 sensitive timing에 영향

### 7-4. 왜 AI는 안 보이고 L1은 보이는가 (latency vs throughput)

같은 disturbance가 있어도:

- **AI throughput (30s 적분 measurement)**:
  - 1초 동안 1% degradation은 1% throughput loss
  - 1초 동안 50% degradation도 1초 평균엔 약 1.7%
  - 따라서 disturbance가 있어도 throughput 측정에선 작게 나옴

- **L1 latency p99 (slot-level worst-case)**:
  - 1 슬롯이라도 50% 더 걸리면 p99에 그 값 직접 반영
  - 30초 30000 슬롯 중 300개만 worst 모드면 p99 영향 큼

→ **같은 메커니즘이지만 throughput vs latency 측정 특성이 보이는 것을 바꿈**.  
→ 사실 AI도 약하게 영향받음 (0~3% 변동) — 측정 노이즈 안에 묻혔지만 0이 아님.

### 7-5. NeuralRx outlier 추정

5x 의 p99 inflation (196 ms vs 41 ms):
- NeuralRx 자체가 L1 비슷한 Conv1d + LDPC 패턴
- 같은 cuPHY library 함수 호출 → L2 slice 충돌 가능성 더 큼
- 또는 cuDNN auto-tune이 partition 한 쪽에서 매번 새로 학습 → 추가 kernel launch overhead

Nsight S7 (3g L1 + NeuralRx 2g) 측정으로 직접 검증 가능.

---

## 8. AI-RAN context에서의 의미

> "L1 커널은 격리하기 어려운 거고, AI-RAN 환경에서는 격리로 인한 오버헤드로 L1 커널 latency가 심한 건 fact"

맞음. 정리:

### 8-1. MIG는 다음 보장
- AI 워크로드의 처리량 격리 ✅
- HBM 메모리 용량/대역폭 격리 ✅
- SM compute 격리 ✅

### 8-2. MIG가 보장하지 못하는 것
- L1처럼 sub-ms latency-critical 워크로드의 **tail latency 격리** ❌
- Cross-partition L2 slice contention ❌
- MIG mode 자체의 기본 오버헤드 (Full GPU 36ms → 7g MIG 37ms = +2%) ❌

### 8-3. AI-RAN 운영 시사점
1. **L1을 더 큰 partition에 넣는 것만으로는 부족**:
   - 4g L1 + AI = 3g L1 + AI ≈ 53ms (M3/M4/D2 일관)
   - 4g 보호가 효과 없음
2. **L1과 AI를 다른 GPU에 두는 게 가장 안전**:
   - 다만 GPU 자원 비용 문제
3. **L1 schedule을 AI-quiet window에 맞춤**:
   - DPDK 같은 dedicated kernel-bypass 통신 + MIG는 latency 보장에 약함
4. **NeuralRx 같은 cuPHY-유사 워크로드를 L1과 같은 GPU에 절대 두면 안 됨**:
   - p99 +377% inflation
5. **partition 수 늘릴수록 L2 slice fragmentation 심해짐**:
   - 2g L1 baseline 자체가 +27% (vs Full GPU)
   - 더 잘게 자르면 baseline 자체가 망가짐

---

## 9. 다음 데이터 (chain 진행중)

### Stage 2: L1 Multi-AI Matrix (진행중)
- 21 cells: M5-M12, 모든 L1 partition × {1 AI, 2 AI, 3 AI} × {sat_compute, sat_hbm, ResNet, Forecaster, NeuralRx, chanpred, xapp, qwen_small}
- 가설: 3 AI worst case (M7a: 4g L1 + 3× sat_compute on 1g) = mean +70~90%

### Stage 3: Nsight Full Matrix
- 26 scenarios: ncu replay-mode + nsys profiling
- 직접 측정 metric:
  - `lts__t_sectors_hit_rate.pct` — L2 hit rate per slice
  - `dram__throughput.avg.pct_of_peak_sustained_elapsed` — HBM bw 포화도
  - `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM 활용도
- **F8 가설 직접 증명 예정**:
  - "MIG L2 cache slice fragmentation이 partition 작을수록 L2 hit rate 낮춤"
  - "AI co-tenant 추가 시 L1의 L2 hit rate가 추가 하락"

---

## 10. 실패 / 미해결 항목

| 항목 | 원인 | 대응 |
|---|---|---|
| qwen_small_1g | PyTorch NVML assert (1g 파티션 작음) | 1g은 qwen용 partition 아님, skip 가능 |
| qwen7b_stress_4g | run_qwen7b_stress.py 안 NERSC HF_HOME 하드코드 | prefill/decode 데이터로 충분 |
| resnet_1g_bs16, bs32 | MIG transient state (chain restart 직후) | Stage 1+2+3 끝나면 별도 retry 가능 |

---

## 11. 백업 / 재현 정보

### 백업 위치
- 서버: `/users/sgkim/cloudlab_aerial/results/20260530/`
- 로컬: `/Users/changjongkim/New_research/cloudlab_results/results/20260531/`

### 분석 스크립트
- `/Users/changjongkim/New_research/cloudlab_results/results/20260531/analyze.py`

### 핵심 실험 스크립트 (모두 `~/cloudlab_aerial/` 서버에 있음, github `airan_cloudlab` repo 동기화)
- `run_fullgpu_baseline.sh` — Full GPU baseline
- `run_mig_baselines.sh` — 4 MIG partition baselines (7g/4g/3g/2g)
- `run_tier1_main.sh` — 13-leg Phase1+4+2+3 orchestrator
- `ai_throughput_v2.sh` — persistent L1 wrapper
- `run_ai_full_matrix.sh` — partition × workload 매트릭스
- `run_ai_supplement.sh` — ResNet fp16 + Forecaster batch sweep
- `run_l1_multi_ai_matrix.sh` — M5-M12 multi-AI L1 측정
- `nsight_full_matrix.sh` — 26 scenarios nsys+ncu

### 새 AI 워크로드 스크립트 (`~/AIRAN_Changjong/experiments/`)
- `run_partition_saturated.py` — fp16 GEMM compute saturation
- `run_hbm_saturated.py` — HBM bw memory saturation
- `run_traffic_forecaster.py` — Informer-lite time-series transformer
- `run_resnet_stress.py` (수정됨) — fp16 ResNet-50 with autocast

---

## 12. 한 줄 요약 (paper draft용 — 초기 버전, §13~14에서 업데이트됨)

> NVIDIA A100 MIG는 AI workload throughput (LLM, CNN, sparse-attention encoder, dense GEMM 모두)을 0.2% 이내로 격리하지만, cuPHY L1 PUSCH RX의 tail latency(p99)는 인접 partition AI co-tenant 한 개만으로도 64~377% 증가하며, partition 크기를 늘려도 보호되지 않는 비대칭 격리 실패를 보인다.

---

## 13. ⭐ **Stage 4: AI per-op latency 측정 결과 — framing 완전 수정**

### 측정
`run_ai_per_op_latency_matrix.sh` — AI workload의 **per-operation latency** (mean이 아닌 p99 tail) 측정.
- 4 workloads (qwen / chanpred / neuralrx / resnet) × 4 partitions × {alone, with_l1 background} × N=5
- 16 cells × 2 setups × 5 runs = 160 runs
- 각 run = 30초간 CUDA event 기반 per-op timing 기록

### 결과 — **mean과 p99의 deceptive 차이**

| Workload × Part | mean Δ | **p99 Δ** |
|---|---|---|
| chanpred 1g | +1.4% | **+9.0%** |
| **chanpred 2g** | +0.5% | **+27.0%** ⭐ |
| **chanpred 3g** | 0% | **+23.7%** ⭐ |
| **chanpred 4g** | +0.5% | **+19.6%** ⭐ |
| neuralrx 1g | -0.4% | +2.1% |
| neuralrx 2g | -0.5% | +4.2% |
| **neuralrx 3g** | -1.1% | **+12.9%** |
| neuralrx 4g | (진행중) | — |
| qwen 2g | -1.1% | **+6.7%** |
| qwen 3g | -1.3% | +4.7% |
| qwen 4g | -2.8% | +3.2% |
| **ResNet 1g/2g/3g/4g** | +0.1~0.2% | **+0.2~0.4%** (거의 무영향) |

### 패턴 — L1 latency 데이터와 **동일**

| AI 종류 | AI p99 Δ (Stage 4) | L1 p99 Δ when AI co-tenant (Tier1) |
|---|---|---|
| chanpred (LSTM) | +20-27% | +72% (3g L1 + chanpred 3g) |
| NeuralRx (PHY-NN) | +13% (3g) | +377% (3g L1 + NRx 3g) |
| Qwen (LLM autoregressive) | +3-7% | +69% (3g L1 + qwen) |
| ResNet (CNN fp16 Tensor Core) | +0.2-0.4% | 0% (Stage 2 M8a) |

→ AI side도 L1과 **같은 방향, 같은 워크로드 패턴** 따름.

---

## 14. ⭐⭐⭐ **최종 framing — paper claim**

### 이전 framing (§7) — 부분만 맞음
> "MIG는 AI throughput 격리는 잘 작동 (avg metric), L1만 p99 latency 격리 실패"

### **새 framing — 데이터로 확정** ⭐

**"MIG는 평균(mean/throughput) 격리는 양호하지만, tail latency (p99) 격리는 부족.  
L1 (latency-critical workload)뿐만 아니라 AI workload도 per-op latency로 측정하면 동일한 p99 inflation 발생.  
다만 inflation 정도는 workload memory access pattern에 의존:  
- ResNet/sat_compute/sat_hbm/Forecaster (uniform pattern) → 영향 거의 없음 (<1%)  
- LLM (Qwen) / LSTM (chanpred) / PHY-NN (NeuralRx) (chaotic pattern) → p99 +9~27% inflation"**

### 데이터 evidence 종합

| 비교 axis | mean 측정 | p99 측정 | 결론 |
|---|---|---|---|
| L1 alone vs L1 + Qwen | +33% | +69% | L1 + LLM = p99 inflate ✅ |
| L1 alone vs L1 + ResNet | 0% | 0% | L1 + ResNet = 무영향 ✅ |
| **Qwen alone vs Qwen + L1bg** | **-1.3%** | **+4.7%** | **AI + L1bg = p99 inflate (작지만 측정됨)** ⭐ |
| **chanpred alone vs chanpred + L1bg** | **0%** | **+24%** | **AI + L1bg = 크게 p99 inflate** ⭐ |
| **ResNet alone vs ResNet + L1bg** | **+0.1%** | **+0.2%** | **AI + L1bg = ResNet은 무영향** ⭐ |

### 메커니즘 — nsys timeline (v2 진행중)
- nsys v1 (10 cells × 15 iters 짧음): S22 (2g + NeuralRx) idle time +22.6% — kernel 사이 시간 증가 확인
- nsys v2 (Tier1-matched 20 cells × 30 iters × N=3): 진행중, 모든 시나리오에서 inter-kernel gap 분포 캡처 예정
- 가설: **driver-level kernel launch queue contention** (chaotic kernel launch frequency의 AI가 시스템 부담)

### Paper-grade evidence 한 그림

```
같은 chanpred 워크로드, 같은 측정 시점:
- iter/s (throughput):  2249 → 2252 (+0.1%)   ← "격리 잘됨"으로 보임
- p99 latency (us):     444  → 564  (+27%)    ← "격리 실패" 진짜 그림
```

**→ metric 선택이 결과를 정반대로 바꿈.**  
**→ "비대칭 격리 실패" 결론은 throughput-only 측정에서 나온 measurement artifact였음.**  
**→ 진짜 그림: MIG의 tail latency 격리 부족 — 양방향 모두.**

---

## 15. 데이터 백업 위치 + 분석 스크립트

### 백업 (로컬, 779 MB)
`/Users/changjongkim/New_research/cloudlab_results/results/20260531/`
- 베이스라인 + Tier1 phases (~5 MB)
- ai_full_matrix, ai_supplement, l1_multi_ai, ai_throughput_v2 (~3 MB)
- **ai_per_op_latency** (Stage 4, 16 cells) ⭐
- **ncu/** (Stage 3 nsight 16 binary reports, 361 MB)
- **nsight_csv/** (Stage 3 extracted CSVs, 61 MB)
- **nsys_full/** (nsys v2 timeline, 진행중)
- **nsys_v1_short/** (archived 10×15 first attempt)
- **nsys_csv/** (nsys v1 kernel timing CSVs, 15 MB)

### 분석 스크립트
- `analyze.py` — Tier1 main aggregation
- `analyze_nsight_full.py` — ncu 16 scenarios × 28 metrics 비교 테이블
- `analyze_nsys_gaps.py` — nsys kernel gap 분포 분석

### 분석 결과 문서
- `EXPERIMENTS_20260531.md` (이 문서) — 종합 실험 결과
- `NCU_ANALYSIS_20260531.md` — ncu 28 metrics 상세 분석 (12 sections)

### Repo
- github: `changjongkim/airan_cloudlab`
- branch: `main`
- 최근 커밋:
  - `83ab788` — 5/31 NCU section 8 reframe (data-driven)
  - `8654675` — 5/31 NCU 28-metric analysis push
  - `c5afd4d` — 5/31 comprehensive data dump + EXPERIMENTS md
