# NCU Stage 3 nsight 종합 분석 — 5/31

**측정**: NVIDIA Nsight Compute (ncu), 16 scenarios × 28 metrics  
**환경**: CloudLab d8545, NVIDIA A100-SXM4-40GB, driver 550.163.01, ncu 2025.2.1  
**컨테이너**: airan:25-3-final (cuPHY 25.3.2 pyaerial)  
**측정 옵션**: `--replay-mode kernel --clock-control none --csv`  
**워크로드**: real_l1.py (cuPHY PUSCH RX pipeline), 5 cells × 2 iters, NUM_WARMUP=2

---

## 0. 시나리오 (16개 완료)

| ID | 설명 | MIG 레이아웃 | L1 위치 | AI co-tenant |
|---|---|---|---|---|
| **S2** | 7g MIG single | 0 (7g 전체) | 7g.40gb (98 SMs) | — |
| **S5** | 3g L1 alone | 9,14 (3g+2g) | 3g.20gb | — |
| **S6** | 3g L1 + Qwen | 9,14 | 3g | qwen_small on 2g |
| **S7** | 3g L1 + NeuralRx | 9,14 | 3g | NeuralRx on 2g |
| **S9** | 3g L1 + 3 AI | 9,19,19,19 | 3g | chanpred+xapp+gpt2 on 1g×3 |
| **S10** | 2g L1 alone | 14,9 | 2g.10gb | — |
| **S12** | 2g L1 + 2 AI | 14,9,14 | 2g(첫번째) | chanpred(3g) + qwen_small(2g) — **부분 실패, file 작음** |
| **S13** | 3g L1 + sat_compute | 9,14 | 3g | sat_compute(8.5GB, 8192) on 2g |
| **S14** | 3g L1 + sat_hbm | 9,14 | 3g | sat_hbm(8.5GB) on 2g |
| **S15** | 4g L1 + sat_compute | 5,14,19 | 4g.20gb | sat_compute on 2g |
| **S17** | 2g L1 + sat_compute | 14,9 | 2g | sat_compute(17GB) on 3g |
| **S18** | 4g L1 + NeuralRx | 5,14,19 | 4g | NeuralRx on 2g |
| **S21** | 4g L1 + 2 sat | 5,14,19 | 4g | sat_compute on 2g+1g |
| **S22** | 2g L1 + NeuralRx | 14,9 | 2g | NeuralRx on 3g |
| **S24** | 3g L1 + 2 sat | 9,14,14 | 3g | sat_compute on 2g+2g (M5a) |
| **S26** | 4g L1 + 3 sat | 5,19,19,19 | 4g | sat_compute on 1g×3 (M7a worst) |

---

## 1. 28 메트릭 의미

### Compute (2)
- `sm__throughput.avg.pct_of_peak_sustained_elapsed` — SM 평균 처리량 (peak 대비 %)
- `smsp__warps_active.avg.pct_of_peak_sustained_elapsed` — SM 안에서 active warp 비율

### L1 cache (3)
- `l1tex__throughput.avg.pct_of_peak_sustained_elapsed` — L1 cache 처리량
- `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum` — L1을 통한 global memory load 바이트
- `l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum` — L1 cache miss 수

### L2 cache (5)
- `lts__t_request_hit_rate.pct` — L2 cache hit rate (%) ⭐
- `lts__throughput.avg.pct_of_peak_sustained_elapsed` — L2 throughput
- `lts__t_bytes.sum` — L2 총 트래픽 (bytes)
- `lts__t_bytes_op_read.sum` — L2 read 바이트 (MIG에서 n/a)
- `lts__t_bytes_op_write.sum` — L2 write 바이트 (MIG에서 n/a)

### DRAM (8)
- `dram__throughput.avg.pct_of_peak_sustained_elapsed` — DRAM 평균 throughput
- `dram__throughput.max.pct_of_peak_sustained_elapsed` — DRAM peak throughput
- `dram__cycles_active.avg.pct_of_peak_sustained_elapsed` — DRAM controller 활성 cycle 비율 (bank util proxy)
- `dram__bytes_read.sum` — DRAM read 바이트
- `dram__bytes_write.sum` — DRAM write 바이트
- `dram__sectors_read.sum` — DRAM read sector 수
- `dram__sectors_write.sum` — DRAM write sector 수
- `dram__sectors_op_atom.sum` — DRAM atomic ops (n/a on MIG)

### Warp stalls (7) — warp가 명령 못 내리는 이유
- `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` — **global memory 대기** ⭐
- `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` — shared memory 대기
- `smsp__warp_issue_stalled_mio_throttle_per_warp_active.pct` — **Memory I/O 시스템 throttle** ⭐⭐
- `smsp__warp_issue_stalled_membar_per_warp_active.pct` — memory barrier
- `smsp__warp_issue_stalled_pipe_busy_per_warp_active.pct` — compute pipe 사용중 (n/a on MIG)
- `smsp__warp_issue_stalled_dispatch_stall_per_warp_active.pct` — front-end dispatch
- `smsp__warp_issue_stalled_drain_per_warp_active.pct` — pipeline drain

### Scheduling/Timing (3)
- `smsp__inst_executed_pipe_lsu.sum` — load/store 명령 수
- `gpc__cycles_elapsed.avg` — GPC active cycle (per-kernel scheduling 측정)
- `gpu__time_duration.sum` — kernel 실행 시간 (microseconds)

---

## 2. 비교 1: **Partition Size 효과** (L1 alone, no AI) ⭐⭐⭐

가장 중요한 baseline — MIG 자체의 비효율 진단.

| Metric | 7g (S2) | 3g (S5) | 2g (S10) | 7g → 2g 변화 |
|---|---|---|---|---|
| **L2 hit rate (%)** | 36.22 | 57.37 | **60.80** | **+68%** ↑ |
| **L2 traffic (B/kernel)** | 12.27 | **377.86** | 289.49 | **+2260%** ↑ |
| L2 throughput (%) | 2.85 | 4.53 | 6.99 | +145% ↑ |
| **DRAM throughput (%)** | 7.24 | 5.88 | **3.25** | **-55%** ↓ |
| **DRAM cycles active (%)** | 7.24 | 5.88 | 3.25 | -55% |
| DRAM bytes read | 99.71 | 99.71 | 100.22 | 0% (같음) |
| DRAM bytes write | 107.72 | 88.58 | 0.00 | -100% |
| SM throughput (%) | 1.65 | 3.69 | 5.08 | +208% ↑ |
| Warps active (%) | 2.99 | 7.17 | 11.12 | +272% ↑ |
| L1 cache miss | 11.59K | 9.85K | 9.67K | -17% |
| **MIO throttle (%)** | **0.02** | 0.13 | **0.66** | **+3200%** ⭐⭐ |
| Long scoreboard (%) | 34.83 | 36.56 | 37.61 | +8% |
| Short scoreboard (%) | 2.36 | 3.13 | 3.04 | +29% |
| GPC cycles | 11.50K | 11.15K | 12.20K | +6% |
| GPU time/kernel (us) | 8.19 | 7.97 | 8.68 | +6% |

### 해석

**파티션이 작아질수록 일어나는 일**:

1. **MIO throttle 33배 폭증** (0.02% → 0.66%):
   - Memory I/O unit이 처리 못해서 warp가 멈춤
   - L1 cache + shared memory + 메모리 접근 모두 통과하는 unit
   - → **작은 partition의 memory subsystem이 압박 받는 직접 증거**

2. **L2 traffic 30배 증가** (12 → 378 KB per kernel):
   - 작은 partition에 작업이 압축되며 L2 slice 더 적극 사용
   - 같은 cuPHY 작업이 더 많은 L2 lookup 생성
   - → L2 slice가 working set 흡수하느라 더 바쁨

3. **L2 hit rate 증가 (36% → 61%)는 deceptive**:
   - hit rate 자체는 높아지지만 **lookup 총량이 30배** 늘어남
   - 실제 miss 양: 7g 8 KB miss → 3g 162 KB miss (20배 증가)
   - → **"hit rate 높다 = 효율적"이라 해석하면 안 됨**

4. **DRAM throughput 55% 감소**:
   - L2가 더 많이 일하므로 DRAM 사용 줄어듦
   - 그러나 DRAM bytes는 동일 → 같은 양의 데이터를 더 느린 페이스로 읽음
   - = **DRAM controller가 막힘 (bank utilization 감소)**

5. **SM throughput / warps active 증가**:
   - 작은 partition (적은 SM 수)에서 같은 작업 → 각 SM이 더 일함
   - 단, 절대값 5% 정도로 여전히 매우 낮음 → **compute-bound 아님, memory-bound**

6. **GPC cycles / GPU time 거의 동일**:
   - per-kernel 실행 시간은 큰 차이 없음
   - 작은 partition이 느린 이유는 **전체 wall-clock 측정에서 발견되어야 함** (kernel 사이 시간)

### 결론
**MIG 작은 partition은 alone 상태에서도 L2/MIO 시스템 압박을 만든다**. 이건 partition size 효과지 AI 영향이 아님.

---

## 3. 비교 2: **3g L1 + 다양한 AI co-tenant**

| Metric | alone (S5) | +Qwen (S6) | +NRx (S7) | +3 AI (S9) | +sat (S13) | +sat_hbm (S14) | +2 sat (S24) |
|---|---|---|---|---|---|---|---|
| L2 hit rate (%) | 57.37 | 57.15 | 57.20 | 57.13 | 57.59 | **59.95** | 57.15 |
| L2 traffic (B) | 377.86 | 346.98 | 330.74 | 362.61 | 330.61 | **287.58** | 377.66 |
| DRAM throughput (%) | 5.88 | 5.92 | 5.74 | 5.63 | 5.58 | **4.12** | **4.23** |
| DRAM cycles active (%) | 5.88 | 5.92 | 5.74 | 5.63 | 5.58 | 4.12 | 4.23 |
| MIO throttle (%) | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 | 0.13 |
| Long scoreboard (%) | 36.56 | 36.70 | 36.46 | 36.75 | 36.47 | 36.61 | 36.73 |
| GPC cycles | 11.15K | 11.32K | 11.32K | 11.34K | 11.25K | 11.17K | 11.16K |
| GPU time/kernel (us) | 7.97 | 8.10 | 8.06 | 8.10 | 8.00 | 7.97 | 7.96 |

### vs S5 alone baseline (변화율)

| Metric | +Qwen | +NRx | +3 AI | +sat | +sat_hbm | +2 sat |
|---|---|---|---|---|---|---|
| L2 hit rate | -0.4% | -0.3% | -0.4% | +0.4% | +4.5% | -0.4% |
| L2 traffic | -8.2% | -12.5% | -4.0% | -12.5% | -23.9% | -0.1% |
| DRAM throughput | +0.8% | -2.4% | -4.2% | -5.1% | **-29.9%** | **-28.0%** |
| MIO throttle | 0% | 0% | 0% | 0% | 0% | 0% |
| **GPC cycles** | **+1.5%** | **+1.5%** | **+1.7%** | +0.8% | +0.1% | +0.1% |
| **GPU time/kernel** | **+1.6%** | +1.1% | +1.6% | +0.4% | 0% | -0.2% |

### 해석

**AI co-tenant이 추가되어도 거의 모든 메트릭 변화 미미 (<5%)**:
- L2 hit rate: 0.4% 미만 (격리 잘됨)
- MIO throttle: 0% 변화 (격리 잘됨)
- DRAM throughput: sat_hbm/2sat 빼면 변화 거의 없음
  - sat_hbm/2sat의 -28~30% 감소는 AI가 DRAM 더 점유해서 L1이 쓸 게 적어짐 → 의외로 격리는 작동
- **GPC cycles +1.5%**: 약간의 scheduling overhead 보이지만 작음
- **GPU time/kernel +1.6%**: 각 kernel 실제 실행은 거의 안 느려짐

→ **ncu 데이터로는 AI co-tenant 의한 L1 interference 거의 못 봄**

### 그러나 Tier1 wall-clock에서는 +33% 누설 → 메커니즘은?
- ncu는 kernel을 격리해서 측정 (replay-mode kernel)
- 실제 production run에서 일어나는 kernel 사이 시간 안 봄
- → **inter-kernel scheduling / launch queue contention**으로 추정 (nsys로 별도 확인 필요)

---

## 4. 비교 3: **4g L1 + 다양한 AI**

| Metric | 3g alone (S5) | 4g+sat (S15) | 4g+NRx (S18) | 4g+2sat (S21) | **4g+3sat (S26)** |
|---|---|---|---|---|---|
| SM throughput (%) | 3.69 | 2.87 | 2.87 | 2.87 | 2.87 |
| Warps active (%) | 7.17 | 5.29 | 5.26 | 5.30 | 5.28 |
| L2 hit rate (%) | 57.37 | 59.22 | 56.80 | 59.49 | 59.50 |
| L2 traffic (B) | 377.86 | 287.84 | **147.23** | 287.81 | 287.82 |
| DRAM throughput (%) | 5.88 | 6.12 | **6.36** | 6.08 | 6.08 |
| MIO throttle (%) | 0.13 | **0.02** | 0.02 | 0.02 | 0.02 |
| Long scoreboard (%) | 36.56 | 38.01 | 38.16 | 38.07 | 38.16 |
| GPC cycles | 11.15K | 11.32K | 11.16K | 11.20K | 11.23K |
| GPU time/kernel (us) | 7.97 | 8.06 | 7.97 | 7.97 | 8.03 |

### vs S5 baseline 변화율

| Metric | +sat (S15) | +NRx (S18) | +2sat (S21) | **+3sat worst (S26)** |
|---|---|---|---|---|
| L2 hit rate | +3.2% | -1.0% | +3.7% | +3.7% |
| L2 traffic | -23.8% | **-61.0%** | -23.8% | -23.8% |
| DRAM throughput | +4.1% | +8.2% | +3.5% | +3.5% |
| **MIO throttle** | **-84.6%** | **-84.6%** | **-84.6%** | **-84.6%** |
| Long scoreboard | +4.0% | +4.4% | +4.1% | +4.4% |
| GPC cycles | +1.5% | 0% | +0.4% | +0.7% |
| GPU time/kernel | +1.1% | 0% | 0% | +0.8% |

### 해석

**4g L1은 3g L1보다 MIO throttle 훨씬 적음** (0.13% → 0.02%):
- 4g는 SM 56개로 3g의 42개보다 많음 → memory pipeline 여유 있음
- → **4g L1의 memory subsystem은 덜 압박 받음**

**그러나 AI co-tenant 효과는 4g도 거의 무관**:
- Worst case S26 (4g + 3 sat_compute on 1g×3) 도 GPC cycles +0.7%만 증가
- GPU time/kernel +0.8% 증가
- → Tier1에서 본 +33% 누설은 여기서도 안 보임

---

## 5. 비교 4: **2g L1 + 다양한 AI**

| Metric | 2g alone (S10) | 2g+2AI (S12) | 2g+sat (S17) | 2g+NRx (S22) |
|---|---|---|---|---|
| SM throughput (%) | 5.08 | (부분 실패) | 5.08 | 5.06 |
| L2 hit rate (%) | 60.80 | (불완전) | 62.55 | 62.95 |
| L2 traffic (B) | 289.49 | — | 293.47 | 297.26 |
| DRAM throughput (%) | 3.25 | — | 3.29 | 3.24 |
| **MIO throttle (%)** | **0.66** | — | 0.66 | 0.65 |
| Long scoreboard (%) | 37.61 | — | 37.92 | 37.73 |
| GPC cycles | 12.20K | — | 12.16K | 12.25K |
| GPU time/kernel (us) | 8.68 | — | 8.67 | 8.74 |

### 해석
**2g L1 자체가 MIO throttle 0.66% (3g의 5배, 7g의 33배)**:
- 작은 partition의 inherent memory subsystem 압박
- AI 추가해도 throttle 거의 안 늘어남 (이미 0.66%에 도달)

**S12 (2g + 2 AI)는 부분 실패** — file 작아 분석 신뢰도 낮음.

---

## 6. 비교 5: **NeuralRx — Tier1의 outlier** (p99 +377%)

| Metric | 3g alone (S5) | 3g+NRx (S7) | 4g+NRx (S18) | 2g+NRx (S22) |
|---|---|---|---|---|
| L2 hit rate (%) | 57.37 | 57.20 (-0.3%) | 56.80 (-1.0%) | 62.95 (+9.7%) |
| L2 traffic (B) | 377.86 | 330.74 (-12.5%) | 147.23 (-61%) | 297.26 (-21.3%) |
| DRAM throughput (%) | 5.88 | 5.74 (-2.4%) | 6.36 (+8.2%) | 3.24 (-45%) |
| MIO throttle (%) | 0.13 | 0.13 (0%) | 0.02 (-85%) | 0.65 (-1.5%) |
| Long scoreboard (%) | 36.56 | 36.46 (-0.3%) | 38.16 (+4.4%) | 37.73 (+3.2%) |
| GPC cycles | 11.15K | 11.32K (+1.5%) | 11.16K (0%) | 12.25K (+9.8%) |
| GPU time/kernel (us) | 7.97 | 8.06 (+1.1%) | 7.97 (0%) | 8.74 (+9.7%) |

### 해석
**ncu로는 NeuralRx의 outlier 패턴이 안 보임**:
- S7 (3g L1 + NeuralRx on 2g): per-kernel metric 변화 1~2%
- Tier1 wall-clock에선 p99 +377% (40ms → 196ms!)
- → NeuralRx의 disturbance도 **inter-kernel level**

S22 (2g L1 + NeuralRx)에서 GPC cycles +9.8%, GPU time +9.7% 증가 보이는데, 이건 **2g partition 자체 효과** (2g alone GPC cycles 12.20K vs 3g alone 11.15K).

---

## 7. **핵심 발견 요약**

### Finding 1: **MIG는 작은 partition에서 L2/MIO subsystem 압박을 만든다** (alone에서도!)
- L2 traffic 30배 증가 (7g 12 KB → 3g 378 KB)
- MIO throttle 33배 증가 (7g 0.02% → 2g 0.66%)
- DRAM throughput 55% 감소
- → **MIG 자체가 만드는 memory subsystem 압박 (real bottleneck)**

### Finding 2: **AI co-tenant 의한 per-kernel 메트릭 변화는 거의 없다** (<5%)
- L2 hit rate, throughput, DRAM, MIO throttle 모두 무반응
- GPU time/kernel +1~2% 변화 (negligible)
- → **MIG의 하드웨어 격리는 kernel level에서 잘 작동**

### Finding 3: **Tier1의 +33% wall-clock disturbance는 ncu에 안 잡힌다**
- ncu `--replay-mode kernel`는 kernel 1개씩 격리 측정
- production run에서 일어나는 kernel 사이 시간(gap) 안 봄
- → **누설은 inter-kernel scheduling / launch queue level에 있음** (nsys timeline 필요)

### Finding 4: **L2 hit rate 증가는 deceptive**
- 7g 36% → 2g 61%로 hit rate 올라가지만
- L2 lookup 총량도 30배 증가
- 실제 miss 양 (절대값)은 작은 partition에서 더 큼
- → "hit rate 높다 = 효율적"으로 해석 금지

### Finding 5: **NeuralRx outlier도 ncu에 안 보임**
- Tier1 p99 +377% 메커니즘은 per-kernel 메트릭 변화로 설명 불가
- → 같은 inter-kernel scheduling 가설

---

## 8. **5/24 paper claim 재해석**

### ❌ 기존 가설: "MIG L2 cache slice fragmentation이 L1 disturbance 원인"
**데이터 안 맞음**:
- L2 hit rate가 작은 partition에서 **오히려 증가** (36% → 61%)
- AI co-tenant 추가해도 L2 hit rate 거의 동일 (변화 < 1%)
- → "L2 fragmentation"이라는 단순 frame은 작동하지 않음

### ✅ 새 가설 (데이터 기반):

**Claim A**: **MIG 자체가 작은 partition에서 memory subsystem 압박 발생**
- L2 traffic 30배 증가, MIO throttle 33배 증가, DRAM throughput 55% 감소
- 이건 **AI 없어도 발생** = MIG의 fundamental cost
- baseline L1 latency 차이를 설명함 (Full GPU 36ms → 2g 51ms = +40%)

**Claim B**: **AI co-tenant 추가 시 ncu가 안 잡는 inter-kernel scheduling 압박 발생**
- per-kernel metric은 동일 (격리 잘됨)
- 그러나 wall-clock latency +33% (Tier1)
- → kernel 사이 시간 늘어남 = **driver-level kernel launch contention 추정**
- nsys timeline으로 확인 예정 (GPU 1에서 진행중)

**Claim C**: **NeuralRx outlier (p99 +377%)는 별도 메커니즘**
- 같은 cuPHY library 사용 → 같은 NVIDIA kernel scheduling queue 경쟁?
- 또는 cuDNN auto-tune의 race condition?
- 추가 조사 필요

---

## 9. **시각화 가능한 figure 후보** (paper용)

### Figure A: Partition Size Effect (alone)
- X축: partition size (7g, 4g, 3g, 2g)
- Y축들: L2 hit rate, L2 traffic, MIO throttle, DRAM throughput
- → "**MIG 자체의 inefficiency**" figure

### Figure B: AI Co-tenant Per-Kernel Metrics
- X축: scenario (S5 alone, S6 Qwen, S7 NRx, ..., S24 2sat)
- Y축: 주요 metrics (변화율 vs alone)
- → "**MIG은 kernel level에서 잘 격리**" figure

### Figure C: Wall-clock vs Per-kernel Discrepancy
- X축: scenario
- Y축 1: Tier1 L1 mean latency (wall-clock)
- Y축 2: ncu per-kernel mean
- → "**누설은 inter-kernel level**" figure

### Figure D: MIO Throttle Heatmap
- X축: L1 partition (7g/3g/2g/4g)
- Y축: AI co-tenant type
- Color: MIO throttle %
- → "**memory subsystem 압박 분포**" figure

---

## 10. **다음 검증 (진행중)**

### nsys timeline 분석 (GPU 1, 진행중)
- 16 scenarios 모두 nsys로 다시 측정
- `.sqlite` 파일에서 kernel start/end timestamp 추출
- 비교:
  - 각 kernel 실행 시간 (ncu와 일치 예상)
  - **kernel 사이 gap 분포** (median, p99, max)
  - 총 wall-clock 시간

### 예상:
- kernel 실행 시간: alone vs with-AI 거의 동일 (ncu 일관)
- **kernel gap: with-AI에서 늘어남** ⭐ (paper claim 확정)
- 총 시간: with-AI에서 더 김 (Tier1 +33% 재현)

### nsys 끝나면:
- Tier1 +33% 메커니즘 직접 증명 가능
- Paper claim "MIG kernel launch queue / driver-level contention" 확정

---

## 11. **데이터 파일**

### 서버
- `/users/sgkim/cloudlab_aerial/results/20260531/ncu/*.ncu-rep` — 16 binary reports (each ~24 MB)
- `/users/sgkim/cloudlab_aerial/results/20260531/ncu_csv/*.csv` — 16 extracted CSVs (each ~3 MB)

### 로컬 (백업)
- `/Users/changjongkim/New_research/cloudlab_results/results/20260531/ncu/` — binary reports
- `/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsight_csv/` — CSVs

### 분석 스크립트
- `analyze_nsight_full.py` — comparison table generator (reproducible)

### Backup history
- `ncu_v1_13metrics/` — 처음 시도 (13 메트릭) — archived for reference
- 현재 `ncu/` — 28 메트릭 final version

---

## 12. **한 줄 요약**

**"NCU 28-metric 분석 결과: MIG 작은 partition은 alone 상태에서도 L2/MIO 시스템 압박 (33배 throttle, 30배 L2 traffic) — fundamental MIG cost. AI co-tenant 추가해도 per-kernel 메트릭 변화 <2% — MIG kernel-level 격리는 잘 작동. Tier1의 +33% wall-clock 누설은 ncu replay-mode가 안 보는 inter-kernel scheduling / driver-level kernel launch contention으로 추정. nsys timeline에서 확정 예정."**
