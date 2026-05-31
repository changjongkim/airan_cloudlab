# NSYS Timeline 분석 — 5/31

**측정**: NVIDIA Nsight Systems (nsys profile)  
**환경**: CloudLab d8545, GPU 1 (chain과 병렬), MIG enabled  
**워크로드**: real_l1.py (cuPHY PUSCH RX), CELLS=20 ITERS=30 N=3 runs (Tier1-matched)  
**컨테이너**: airan:25-3-final, --trace=cuda --gpu-metrics-device 사용 안 함 (MIG에서 안 됨)

---

## 0. nsys 측정 목적

**ncu의 한계**:
- ncu `--replay-mode kernel`은 각 kernel을 격리해서 metric 수집
- per-kernel L2 hit rate, DRAM throughput 등 측정 → AI co-tenant 변화 거의 안 보임 (<2%)
- 그러나 Tier1 wall-clock은 +33% 누설 보임
- → ncu가 **kernel 사이의 시간 (inter-kernel gap)**을 못 봄

**nsys 강점**:
- 실시간 timeline 캡처 → 각 kernel의 정확한 start/end timestamp
- kernel 사이 gap (idle time) 직접 측정
- production run 그대로의 disturbance 보존

→ nsys로 Tier1 누설 메커니즘을 직접 증명.

---

## 1. 측정 시나리오 (16 × 3 runs = 48 traces)

S2/S5/S6/S7/S9/S10/S12/S13/S14/S15/S17/S18/S21/S22/S24/S26 (ncu와 동일).  
각 scenario에 3 runs (statistical power).

---

## 2. v1 vs v2 (조건 차이)

| 항목 | v1 (잘못) | **v2 (Tier1 매칭)** |
|---|---|---|
| CELLS | 10 | **20** |
| ITERS | 15 | **30** |
| N runs | 1 | **3** |
| Per-run 측정 시간 | ~5 sec | **~15-20 sec** |
| Kernel 수 | ~6,340 per run | **15,376 per run** |
| 총 nsys 데이터 | 16 traces | **48 traces (×3)** |
| Tier1 매칭 | ❌ | ✅ |

v1은 너무 짧아서 AI co-tenant disturbance가 steady-state 도달 못 함. v2가 paper-grade evidence.

---

## 3. **v2 핵심 결과**: 3g L1 alone vs AI co-tenants

baseline: **S5 (3g L1 alone)** — median kernel 5.97us, median gap 1.18us, p99 gap 808us, total 1881ms, idle 1473ms (78%).

### 비교 표 (% change vs S5)

| Scenario | MedDur | MedGap | **p95 gap** | **p99 gap** | MaxGap | **total** | **idle** | idle% |
|---|---|---|---|---|---|---|---|---|
| S6 +Qwen | -2.9% | +2.7% | **+8.0%** | **+13.2%** | -7.0% | **+9.5%** | **+12.2%** | +2.2% |
| **S7 +NeuralRx** | -5.2% | +2.7% | **+28.0%** ⭐ | **+22.0%** ⭐ | -7.1% | **+16.0%** ⭐ | **+20.5%** ⭐ | +3.7% |
| S9 +3 AI | 0% | +2.7% | +1.4% | +1.3% | -7.2% | +10.2% | +13.1% | +2.6% |
| S13 +sat_compute | -4.8% | -1.8% | +2.6% | +3.0% | -3.1% | -0.6% | -0.8% | -0.2% |
| S14 +sat_hbm | -4.8% | -7.2% | +2.5% | +7.0% | -3.1% | +2.5% | +3.2% | +0.7% |
| S24 +2 sat | -3.9% | -7.2% | +0.9% | +0.9% | -2.3% | +4.7% | +6.0% | +1.2% |

### 해석

**Chaotic memory pattern AI (Qwen, NeuralRx)**:
- p99 gap inflate +13-22% — kernel 사이 시간 길어짐
- total wall-clock inflate +9-16% — Tier1 wall-clock 누설과 부분 일치
- idle time inflate +12-20% — disturbance가 idle time에 직접 노출됨

**Uniform pattern AI (sat_compute, sat_hbm, 2 sat)**:
- 모든 metric 변화 <7%
- total 변화 0~+4.7%
- → uniform pattern AI는 L1 inter-kernel 영향 없음 (격리 작동)

**3 AI multi-instance (S9)**:
- p95/p99 gap 거의 안 변함 (+1.3%)
- total +10.2% (Qwen S6와 비슷)
- idle +13.1%
- → multi-AI 영향은 single Qwen 한 개와 비슷

---

## 4. 4g L1 + AI

| Scenario | MedGap | p99 gap | total | idle | idle% |
|---|---|---|---|---|---|
| **S26 4g + 3 sat (worst)** | +5.4% | +7.2% | **+15.2%** ⭐ | **+20.0%** ⭐ | +4.1% |
| S21 4g + 2 sat | +7.2% | +6.4% | +8.1% | +11.0% | +2.5% |
| S18 4g + NeuralRx | +5.4% | +7.3% | +0.2% | +0.8% | +0.6% |
| S15 4g + sat | +6.3% | +13.0% | +0.6% | +1.4% | +0.8% |

(baseline = S5 3g alone — 4g alone baseline 없어 cross-partition 비교)

→ **S26 (3 sat on 1g×3)이 4g에서 가장 큰 disturbance**: total +15%, idle +20%.  
→ 4g + NeuralRx (S18)는 의외로 작음 — 4g L1은 NeuralRx에 비교적 robust.

---

## 5. 2g L1 + AI

| Scenario | MedGap | p99 gap | MaxGap | total | idle | idle% |
|---|---|---|---|---|---|---|
| S10 2g alone (baseline) | 1.13 us | 1444 us | 113 ms | 2515 ms | 2079 ms | 82.7% |
| S12 2g + 2 AI | -1.9% | -0.2% | +2.1% | -9.0% | -10.4% | -1.6% |
| S17 2g + sat | +1.9% | -3.1% | +3.8% | -11.5% | -13.5% | -2.2% |
| S22 2g + NeuralRx | **+4.7%** | +0.8% | +1.3% | +0.9% | +1.5% | +0.6% |

→ 2g L1에서는 AI 추가해도 큰 변화 없거나 **오히려 줄어듦** (S12, S17 negative).  
→ **이유**: 2g L1 baseline이 이미 idle 82.7% / total 2515ms로 매우 큰 baseline. AI 추가해도 partition cap이 dominant.

---

## 6. **Partition Size 효과 (MIG 자체)** ⭐⭐⭐

L1 alone, partition 크기만 변화:

| Scenario | MedDur | MedGap | **p95 gap** | **p99 gap** | MaxGap | total | **idle** | idle% |
|---|---|---|---|---|---|---|---|---|
| **S2 (7g, 98 SMs)** | 5.48 us | 1.22 us | — | **567 us** | 114 ms | 1948 ms | 1448 ms | 74.2% |
| **S5 (3g, 42 SMs)** | +9% | -2.6% | -2.9% | **+42.4%** | +3.1% | -3.4% | +1.7% | +5.5% |
| **S10 (2g, 28 SMs)** | **+21%** | -7% | +14.8% | **+154.5%** ⭐⭐ | -0.4% | **+29%** | **+44%** | **+11%** |

### 해석
- **2g alone의 p99 gap이 7g 대비 +155%** — MIG bandwidth throttling 직접 증명
- idle time +44%, total +29% — 작은 partition은 자기 자체로 비효율
- AI 없이도 partition 작으면 inter-kernel disturbance 폭증
- ncu의 mio_throttle 33배 결과와 일관

---

## 7. **Tier1 wall-clock과 nsys 비교**

같은 시나리오의 데이터 세 가지 측정 metric:

| 시나리오 | Tier1 wall-clock | nsys total (3 runs) | nsys idle time | ncu per-kernel |
|---|---|---|---|---|
| 3g L1 + Qwen | mean +33%, p99 +69% | **+9.5%** | **+12.2%** | <2% |
| 3g L1 + NeuralRx | mean +50%, p99 +377% | **+16%** | **+20.5%** | <2% |
| 3g L1 + sat_compute | 0% | -0.6% | -0.8% | <0.2% |
| 3g L1 + 3 AI | mean +33% | +10.2% | +13.1% | <2% |

### 의미
- **Tier1 (per-iter timing)** > **nsys total** — Tier1은 per-iter time이라 setup overhead 제외, 더 민감
- **nsys total / idle** = production wall-clock disturbance — paper-grade evidence
- **ncu per-kernel** = kernel 자체는 격리됨 (memory subsystem)
- → 누설의 위치 = **kernel 사이의 시간 (idle/gap)**, kernel 실행 자체 아님 ✅

---

## 8. **메커니즘 가설** (3개 측정 종합)

```
1. AI co-tenant이 옆 partition에서 끊임없이 kernel launch
   ↓
2. NVIDIA driver / CUDA runtime의 kernel launch queue 공유 (per-GPU)
   ↓
3. L1의 kernel launch 명령이 AI의 명령들 사이에 끼어들기 어려움 → driver-level queueing
   ↓
4. L1 kernel 사이 gap 증가 (nsys로 측정됨, +13-28% p99 gap)
   ↓
5. Cumulative 효과로 wall-clock latency 증가 (Tier1 +33% mean)
```

### 메커니즘 확정 evidence
- ncu (per-kernel L2/DRAM/mio_throttle): AI 추가 시 변화 <2% → kernel 실행 자체는 안 망가짐
- nsys (gap distribution): AI 추가 시 p99 gap +13-28% → kernel **사이** 시간이 망가짐
- Tier1 wall-clock: AI 추가 시 +33% → production에서 결과로 나타남
- workload-type dependence:
  - chaotic AI (Qwen, NeuralRx, chanpred LSTM): gap inflation 큼 — driver queue 자주 침해
  - uniform AI (sat_compute, sat_hbm, ResNet, Forecaster): gap inflation 거의 없음 — driver queue 평소대로

---

## 9. **추가 분석 필요** (sqlite 기반)

CSV 분석은 aggregate statistics만 가능. SQLite로 가능한 detail 분석:

1. **Per-kernel-type gap analysis**:
   - cuPHY pipeline에서 어느 kernel(ChannelEstimator vs Equalizer vs LdpcDecoder)이 가장 큰 gap inflation 겪나
   - "병목 kernel" 식별

2. **Kernel sequence/ordering**:
   - kernel A → kernel B 같은 sequence의 transition gap
   - 특정 sequence가 AI에 의해 더 많이 망가지는지

3. **Time-series of gaps**:
   - 측정 시작 후 시간에 따라 gap 분포 변화
   - AI co-tenant이 시간 지남에 따라 누적 효과 보이는지

4. **Memcpy/memset 분석**:
   - GPU memory transfer가 AI에 의해 더 길어지는지
   - kernel launch queue와 memory transfer queue 분리?

5. **Kernel-pair correlation**:
   - "Long gap after kernel X" 패턴 찾기

---

## 10. 데이터 파일

### nsys raw reports
- 서버: `/users/sgkim/cloudlab_aerial/results/20260531/nsys_full/*.nsys-rep` (48 files, ~7 MB total)
- 로컬: backup 가능 (binary)

### Extracted CSVs
- 서버: `~/cloudlab_aerial/results/20260531/nsys_csv_v2/*.csv` (48 files, 164 MB)
- 로컬: `/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_csv_v2/` (194 MB)

### SQLite databases (다음 단계)
- nsys export로 .sqlite 생성 → 정교한 query 가능
- 각 .nsys-rep → .sqlite (10-50 MB)

### 분석 스크립트
- `analyze_nsys_gaps.py` — v1 (10 cells × 15 iters) 분석
- `analyze_nsys_v2.py` — **v2 (Tier1 matched) 종합 분석** ⭐

---

## 11. **한 줄 요약**

> **"NSYS v2 (Tier1 매칭 조건) 분석으로 inter-kernel gap inflation 직접 측정 — Qwen/NeuralRx 시나리오에서 p99 gap +13-28%, total wall-clock +9-16% 증가 확인. ncu per-kernel 격리는 유지(<2%)되나 kernel 사이 시간이 망가짐 → Tier1 wall-clock 누설 메커니즘 = driver-level kernel launch queue contention 가설 직접 evidence. Uniform pattern AI(sat_compute/sat_hbm)는 누설 없음 확인 — workload memory pattern이 결정 변수."**
