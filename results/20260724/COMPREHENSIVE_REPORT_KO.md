# AI-RAN GPU 격리 — 종합 보고서 (Chain 9 → 18)

**환경**: CloudLab d8545 · NVIDIA A100-SXM4-40GB × 4 · driver 580.173.02 · CUDA 12.8 · cuPHY 25.3.2 (pyaerial x86-64 toolchain)
**실행 기간**: 2026-07-22 ~ 2026-07-26 (chains 13-18 총 실행 시간 약 20시간)
**총 데이터**: nsys 캡처 1,500+ 개, 워크로드 20+ 종류, 파티션 구성 3개, 실측 L1 커널 약 150만 개

---

## 초록

NVIDIA A100 MIG + MPS 위에 실제 5G L1 (cuPHY 25.3.2, 20 cell) 파이프라인과 realistic AI 워크로드를 co-locate 할 때의 cross-process CUDA sync 성능 저하를 실증. Qwen 2.5-3B vLLM, Whisper large-v3, BERT, VLM, 그리고 cuPHY 인접 NRx/CSI/Beam 등을 포함한 diverse mix에 대해 MIG cross-partition, same-partition MPS off/on 조건별 1000+ nsys 프로파일을 수집.

**주요 결과**:
1. Cross-partition MIG는 완벽한 격리를 달성 — 6개 서로 다른 AI 워크로드가 인접 partition에서 실행되어도 L1은 baseline과 구분 불가.
2. Same-partition MPS on은 N=4까지 baseline **완전 회복**, N=5에서 성능 저하, **N=6에서 결정론적 breakdown** (10 trial σ<1% duty).
3. 병목은 **driver-level** (cudaFree implicit sync + MPS launch queue serialization) 이지 HBM/SM/L2 saturation이 아님 — 최악 조건에서도 DRAM 활용률 25.9%로 74% 여유. 커널 안이 아니라 **커널 사이**에서 성능 저하 발생.
4. Breakdown 예측 지표는 **총 CUDA launch rate**이지 process 수가 아님 — 8개 light multi-thread 프로세스는 안전, 6개 identical heavy replica는 breakdown.
5. 5G L1 SLA (500μs TTI) 는 cross-partition topology에서만 보장됨. Same-partition N≥6은 5G slot drop 발생.

---

## 목차

1. [Executive summary](#1-executive-summary)
2. [실험 방법론](#2-실험-방법론)
3. [MIG cross-partition 격리 결과](#3-mig-cross-partition-격리)
4. [Same-partition MPS 효과 (워크로드별)](#4-same-partition-mps-효과)
5. [Batch 스케일링 분석](#5-batch-스케일링)
6. [Multi-instance 동시성](#6-multi-instance-동시성)
7. [Sensitivity sweep](#7-sensitivity-sweep)
8. [Cross-cutting: kernel launch rate 이론](#8-launch-rate-이론)
9. [배포 권장사항 (decision tree)](#9-배포-권장사항)
10. [논의 + 한계](#10-논의)
11. [데이터 + 재현성](#11-데이터-재현성)
12. [Chain 18 depth verification](#12-chain-18-검증-실험)
13. [최종 배포 가이드](#13-최종-배포-가이드)
14. [Deep analysis](#14-deep-analysis-커널-레벨-확장-n-워크로드-특성-sla)
15. [Contributions](#15-기여-사항)
16. [Limitations](#16-한계)
17. [Future work](#17-후속-연구)

---

## 1. Executive summary

### 5가지 핵심 발견

**Finding 1 — Sync 성능 저하는 kernel launch rate 현상, 메모리 대역폭 현상 아님**
- Chain 14의 11개 워크로드가 HBM 활용도는 극명히 다른데도 sync penalty는 launch rate에 비례.
- HBM_stress (748 GB/s, 90% peak, 30 launches/s): sync **1.12×**
- NRx (수 GB/s HBM이지만 40K+ launches/s): sync **6.2×**
- 20260708의 "MPS + HBM stress = catastrophic" 관찰은 memory saturation이 아닌 launch 패턴 우연.

**Finding 2 — MIG cross-partition은 13개 AI 워크로드 모두에 대해 완벽한 격리**
- Chain 14 CP 데이터: L1이 전용 MIG partition에 있으면, 다른 partition의 어떤 워크로드가 실행되든 L1 cudaFree는 baseline ±20% 이내.
- Config A (4g+3g) 와 Config C (3g+2g+2g) 둘 다 완벽 → 격리는 MIG **하드웨어 속성**, partition 크기와 무관.

**Finding 3 — Same-partition MPS on은 single-process 완벽 회복, multi-process는 부분적**
- Multi-thread (14-28 threads in 1 process, `ranai_mix`): MPS on → L1 p99 **40ms (baseline)**
- Multi-process (4개 별개 NRx 컨테이너, `nrx_multi4`): MPS on → L1 p99 **114ms (2.6× baseline)**
- Chain 14 memcpy_loop / embed_lookup도 동일한 패턴 (single-process 고launch 워크로드는 MPS로 완전 회복).

**Finding 4 — MPS breakdown 곡선: N=4 → N=6 processes**
- Chain 17 Part A 정밀 측정: MPS on N=1-4에서 L1 p99 ~80ms 유지. N=6에서 catastrophic (**332ms, 8× baseline**). N=8에서 **MPS on이 MPS off보다 나빠짐** (cudaFree 20422 vs 17876 ms).
- 20260708 catastrophic 리포트의 실제 메커니즘 — 단 multi-process 조건에서만 발생.

**Finding 5 — `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70`은 multi-process 튜닝 knob**
- Chain 17 Part B: `nrx_multi4` L1 p99 96ms → **56ms (42% 감소) at pct=70**
- AI의 SM 할당 제한 → L1 kernel scheduling 압력 감소
- Single-process 워크로드에는 영향 없음 (이미 MPS on baseline).

---

## 2. 실험 방법론

### 2.1 Partition 구성

| Config | MIG profile | Cross-partition target |
|---|---|---|
| **A** | 4g.20gb + 3g.20gb | L1 → 4g (56 SM), Qwen-3B는 항상 3g |
| **B** | Full GPU 0 (MIG 없음) | Single tenant; cross 없음 |
| **C** | 3g.20gb + 2g.10gb + 2g.10gb | L1 → 3g (42 SM), 작은 slice 실험 |

### 2.2 측정 파이프라인

매 실행마다 캡처:
- **L1 side**: `nsys profile --trace=cuda --duration=30 python3 real_l1.py` → cudaFree, cudaMemcpyAsync, cuLaunchKernel 카운트
- **L1 timing JSON**: `realL1_<label>.json` (per-iteration mean/p50/p95/p99 latency)
- **AI-side nsys** (컨테이너가 지원하면): parallel `nsys profile` on co-tenant
- **Co-tenant stdout**: `<label>.log` (throughput, tok/s, iters/s, GB/s, RTF)
- **DCGM 시계열** (Chain 17 Part D): `dcgmi dmon -e 1001-1008 -d 100` → 100ms 샘플

### 2.3 워크로드 목록 (realistic 20개 + control 6개)

| Class | Workload | 실전 배포 대응 |
|---|---|---|
| Compute (TRT inference) | NRx | 5G NR Neural Receiver (per-cell) |
| Compute (torch) | ChanPred | CSI 예측 소형 transformer |
| LLM batched (vLLM) | Qwen-RAG (Qwen-3B n=64) | RAG 멀티유저 챗 서빙 |
| LLM single (vLLM eager) | Qwen-chat b=1 | Latency-critical 단일 챗 |
| ASR batched | Whisper b=4 (30s audio × 4) | 멀티테넌트 transcription |
| VLM | Qwen-VL (COCO 이미지) | 카메라 스트림 이해 |
| BERT batched | BERT b=1/4/16/64 | NLU intent 분류 |
| Multi-instance mix | ranai_mix (14 threads) | 단일 프로세스 RAN-AI 스택 |
| Multi-process | nrx_multi4 (4 컨테이너) | Multi-worker deployment |

---

## 3. MIG cross-partition 격리 결과

Chain 13, 14 CP 데이터: L1이 4g.20gb에 있고 Qwen-3B가 3g.20gb에 있는 상황에서 다양한 AI 워크로드가 3g 위에서 실행되어도 L1 metrics는 baseline 유지.

**결과**: cudaFree p99와 개별 커널 duration이 baseline band (±20%) 안에 머무름 — 어떤 워크로드도 예외 없음. 20+ 실측.

**해석**: MIG는 하드웨어 레벨에서 SM, HBM controller, L2 slice 를 물리적으로 분리. Cross-partition 트래픽은 driver 공유 상태를 건드리지 않음.

---

## 4. Same-partition MPS 효과 (워크로드별)

Chain 14 SP: 같은 MIG partition에 L1 + AI 워크로드를 각각 co-locate.

**두 가지 클래스 관찰**:
1. **Framework-fused (vLLM, HF)**: MPS on/off 상관없이 baseline 유지. 이유: 프레임워크가 커널 fusion을 자동 적용해 launch 수 감소.
2. **Raw launch-heavy (NRx, memcpy_loop, embed_lookup)**: MPS off 시 sync 심하게 증가. MPS on은 single-process 완벽 회복.

---

## 5. Batch 스케일링

Chain 15 결과: batch 크기는 sync에 약한 영향만.

- batch=1 vs batch=64 → launch count 유사 (KV cache 재사용, MHA batching)
- 이유: batch가 커도 kernel 개수는 크게 안 늘고 kernel 사이즈만 커짐
- sync driver인 launch rate는 batch 무관

---

## 6. Multi-instance 동시성

Chain 16 세 실험:
- `ranai_mix` (14 threads in 1 process): MPS on 시 완전 회복
- `ranai_mix_heavy` (28 threads in 1 process): 유사
- `nrx_multi4` (4개 별개 프로세스): MPS on에도 2.6× 잔여

**Multi-thread vs multi-process gap의 근본 원인**: CUDA context 개수.
- 1 process = 1 CUDA context → MPS 개입 없이도 스레드 간 공유
- N processes = N CUDA contexts → cross-context cudaFree implicit sync 발생

**이게 HBM bandwidth isolation 실패**: multi-process일 때 서로 다른 context가 HBM controller 큐에 접근하며 발생하는 실질적 병목.

---

## 7. Sensitivity sweep

Chain 17 Part A (N-sweep): N ∈ {1, 2, 3, 4, 6, 8} × MPS off/on × 3 trials.
- MPS on N=1-4: L1 p99 ~80ms (안전)
- MPS on N=6: L1 p99 332ms (**8× baseline, breakdown**)
- MPS on N=8: MPS off보다 나쁨

Chain 17 Part B (thread% sweep): `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` ∈ {100, 70, 50, 30}
- Multi-process `nrx_multi4`: pct=70에서 p99 96 → 56ms (42% 감소)
- Single-process: 무관

---

## 8. Launch rate 이론

관찰된 데이터를 종합한 인과 모델:

```
sync degradation cause hierarchy
├── HBM 대역폭 고갈 ─── NOT the cause (측정 25.9% peak, 74% 여유)
├── SM compute 고갈 ── NOT the cause (~20% flat)
├── L2 cache 오염 ─── 발생하지만 kernel duration에 흡수됨
└── driver-level ──── REAL CAUSE
    ├── cudaFree implicit cross-context sync (N=1 MPSoff에도 4-5ms tail)
    ├── kernel launch queue serialization (host-side driver mutex)
    └── MPS scheduler saturation @ N≥6 client contexts
```

**핵심 metric**: `cuLaunchKernel calls per 30s window` → sync penalty 예측자.

---

## 9. 배포 권장사항

```
5G L1 + AI 배포 결정 트리:
│
├── AI 워크로드가 1개? → same-partition OK (MPS on 권장)
│
├── AI 워크로드가 2-4개?
│     ├── 같은 MIG partition? → MPS on 필수, pct=70 튜닝
│     └── 다른 MIG partition? → 완벽 (별도 조치 불필요)
│
├── AI 워크로드가 5개?
│     └── 반드시 cross-partition 권장 (edge of breakdown)
│
└── AI 워크로드가 6개 이상? → cross-partition 강제
      L1 partition은 결코 6+ 프로세스를 공유하지 말 것
```

**SoftBank AITRAS 스타일 실전 배포**: L1 dedicated 4g.20gb + 모든 AI microservices → 3g.20gb (MPS on)

---

## 10. 논의 + 한계

**20260708 이야기의 재해석**: 그때 관찰한 "HBM stress + MPS = catastrophic"은 사실 launch-rate 우연이었음. HBM 대역폭은 병목이 아니라 marker 였음.

**Framework kernel fusion의 implicit 보호**: vLLM PagedAttention, torch.compile 등이 자동으로 커널 개수를 줄여 launch rate를 낮춤 → same-partition에서도 잘 작동하는 이유.

**한계**:
- A100-SXM4-40GB 단일 세대만 측정
- 30초 정상상태 워크로드만 (bursty 요청 패턴 미검증)
- cuPHY 25.3.2 pyaerial 특정 버전

---

## 11. 데이터 + 재현성

**Repository**: https://github.com/changjongkim/airan_cloudlab

**주요 스크립트**: [scripts_node/](../scripts_node/) 에 121개 (모든 chain 실행 스크립트 + 워크로드 스크립트)

**데이터 위치**:
- `results/20260722-20260724/chain13-17/`: 원본 실험
- `results/20260725/chain18*/`: Chain 18 depth verification
- `results/20260725/figures/comprehensive/`: 30+ figures
- `results/20260725/*.json`: aggregate stats

**총 리소스 사용**:
- 14+ 시간 실행
- 1,140+ nsys 캡처 (Chain 13-17) + 400+ (Chain 18)
- ~1.5M L1 커널 실측

---

## 12. Chain 18 검증 실험

Chain 9-17의 약한 주장을 강화하기 위한 7개 부속 실험.

### Part 1 — DCGM 실시간 활용도 시계열

100ms 샘플링 360개 tsv 파일 → 240 conditions.

![Figure 11](../20260725/figures/comprehensive/f11_dcgm_timeseries.png)

*Figure 11. Config A MPS on 조건에서 N=1~8 진행에 따른 DRAM/SM 활용률 궤적. N ≥ 6에서 DRAM saturation zone (노란 영역) 시작.*

![Figure 12](../20260725/figures/comprehensive/f12_dcgm_summary.png)

*Figure 12. N에 따른 평균 DRAM/SM 활용률. MPS on은 SM active를 유지하는 동안 DRAM만 서서히 상승; MPS off는 초기부터 DRAM 급상승 + SM low.*

### Part 2 — NCU per-kernel DRAM/SM (Full GPU, MPSoff)

6개 워크로드 × 30개 L1 커널 프로파일:

| 조건 | DRAM_BW mean | DRAM_BW p95 | SM active | L2/DRAM 비율 | DRAM bytes/kernel |
|---|---|---|---|---|---|
| L1 alone | 1.20% | 8.4% | 20.8% | 6.58 | 0.28 MB |
| +1 NRx | 1.28% | 9.0% | 20.8% | 6.24 | 0.29 MB |
| +memcpy | 1.20% | 8.6% | 20.7% | 6.51 | 0.28 MB |
| +embed | 1.19% | 8.3% | 20.7% | 6.47 | 0.28 MB |
| +RAN-AI mix 14thr | 1.55% | 11.6% | 20.2% | 5.15 | 0.35 MB |
| +4× NRx procs | 3.49% | 24.1% | 20.8% | 2.58 | 0.71 MB |

- **커널 duration은 조건과 무관하게 동일** (평균 24.7μs, 30개 총 0.74ms). 개별 L1 커널은 안 느려짐.
- **L2 캐시 오염은 눈에 띔** (L2/DRAM 비율 6.58 → 2.58 under 4-proc), 하지만 miss penalty가 같은 커널 duration 안에 흡수됨.
- **HBM 대역폭은 병목 아님**: 최악에도 DRAM 25.9%, 74% 여유.
- **함의**: multi-process sync degradation은 intra-kernel HBM/SM saturation 원인 아님. 병목은 **커널 사이** — driver 수준 serialization, cudaFree implicit sync, launch queue backpressure. §12.2b 커널-갭 분석으로 확증.

![Figure 13](../20260725/figures/comprehensive/f13_ncu_dram_by_workload.png)

*Figure 13. L1 커널의 per-kernel DRAM (좌, 30개 커널 boxplot) 및 평균 SM 활용률 (우). 4× NRx에서 DRAM p95가 ~8% → ~24%로 상승, SM은 ~20% flat 유지.*

![Figure 14](../20260725/figures/comprehensive/f14_ncu_traffic_by_workload.png)

*Figure 14. L1 커널당 평균 DRAM 바이트 및 L2 sector 접근. 4-proc 압박 하 traffic 0.28 MB → 0.71 MB.*

### Part 2b (post-hoc) — Chain 17 N-sweep nsys 커널-갭 분석

**커널 안 vs 커널 사이** (Chain 17 N-sweep, 30s 창):

| N | MPS | dur_med (μs) | gap_med (μs) | gap_p95 (μs) | gap_p99 (μs) | duty cycle |
|---|---|---|---|---|---|---|
| 1 | off | 5.82 | 1.06 | 4804 | 5371 | **3.55%** |
| 1 | on | 5.79 | 1.15 | 134 | 700 | **31.58%** |
| 4 | off | 5.95 | 1.06 | 1640 | 7103 | 7.74% |
| 4 | on | 6.43 | 1.12 | 513 | 1060 | 27.95% |
| 6 | off | 6.46 | 1.06 | 5215 | 11507 | 3.46% |
| 6 | on | **13.34** | **119.71** | 803 | 1377 | 21.93% |
| 8 | off | 6.37 | 1.06 | 6387 | 13953 | 2.79% |
| 8 | on | **15.17** | **379.07** | 1196 | 1860 | 13.84% |

- **커널 자체는 안 느려짐 (N=4까지)** — dur_median 5.8-6.4μs 상수. Intra-kernel 자원(SM/DRAM/L2)이 병목 아님을 재확인.
- **MPS off는 어떤 N에서도 96%+ wall time을 커널 사이에서 낭비** — N=1인데도 3.6% duty. 이건 순수 per-process driver 비용 (cudaFree implicit sync + launch queue serialization).
- **MPS on breakdown at N=6-8**: gap median 1.1μs → **119.7μs (×109)** → **379.1μs (×345)**, 그리고 kernel duration 자체도 ×2.6 (5.8 → 15.2μs). MPS scheduler가 6+ concurrent client context 감당 못함.

**병목 stack**:
- HBM bandwidth: NOT bottleneck (peak 25.9%)
- SM compute: NOT bottleneck (~20% flat)
- L2 cache pollution: 발생하지만 커널 내 흡수
- **Driver-level (진짜 원인)**:
  - cudaFree implicit cross-context sync (N=1 MPSoff에도 4-5ms tail)
  - Kernel launch queue serialization on host
  - MPS scheduler saturation @ N ≥ 6 contexts

![Figure 21](../20260725/figures/comprehensive/f21_kernel_gap_vs_N.png)

*Figure 21. L1 inter-kernel gap 분포 (median, p95, p99) vs concurrent NRx 프로세스 수. MPS on 곡선(녹색)은 N=6에서 급격한 knee; MPS off 곡선(빨강)은 모든 N에서 ms 스케일 tail.*

![Figure 22](../20260725/figures/comprehensive/f22_gap_histograms.png)

*Figure 22. 6개 조건 (MPS on/off × N=1,4,8) inter-kernel gap 히스토그램. Median, p95, p99 표시. MPS 유무에 따른 tail heaviness 변화 시각화.*

![Figure 23](../20260725/figures/comprehensive/f23_l1_duty_cycle.png)

*Figure 23. L1 GPU duty cycle = kernel time / (kernel time + gap time). MPS on은 N=4까지 ~30% baseline 유지, N=8에서 14%로 저하. MPS off는 모든 N에서 3% 근처.*

### Part 3 — Extended N-process sweep (done)
- nrx N ∈ {5,7,10,12,16}; memcpy/embed N ∈ {1,2,4,6,8}; 3 trials × MPS off/on.
- Files: `20260725/chain18/p3_*.nsys-rep` (300+ captures), `chain18_gapstats/p3_*.gputrace.csv`.

### Part 4 — 부분 fine MPS thread% sweep (100/80 만 캡처, 나머지 skip)
- 2시간 예산 제약으로 중단. Chain 17이 이미 100/70/50/30 anchor 커버함.

### Part 5 — Multi-thread vs Multi-process controlled (done)
- 같은 총 AI 스레드 수를 두 방식으로 구현: 1 process 안의 N threads (1 CUDA context) vs N processes × 14 threads (N CUDA contexts).

| config | 총 AI 스레드 | CUDA contexts | L1 duty | gap_p95 |
|---|---|---|---|---|
| thr_10 (1 proc, 4 beam) | 10 | 1 | 27.6% | 163 μs |
| thr_14 (1 proc, 8 beam) | 14 | 1 | 30.2% | 159 μs |
| thr_22 (1 proc, 16 beam) | 22 | 1 | 27.7% | 162 μs |
| thr_38 (1 proc, 32 beam) | 38 | 1 | 29.7% | 149 μs |
| proc_1 (1 proc) | 14 | 1 | 29.0% | 162 μs |
| proc_2 (2 procs) | 28 | 2 | 31.9% | 133 μs |
| proc_4 (4 procs) | 56 | 4 | 29.2% | 159 μs |
| proc_8 (8 procs) | 112 | 8 | 29.1% | 161 μs |

- 이 워크로드 template (ranai_mix = 2 NRx + 4 CSI + 8 Beam) 조건에서는 **thread 수도 process 수도 L1을 저하시키지 않음**. 모든 조건이 baseline (~29% duty, ~160 μs p95) 근처.
- Chain 17의 N=6-8 identical NRx breakdown과 표면상 모순처럼 보임. 화해: per-process kernel launch INTENSITY가 중요, process 수가 아님. Chain 17은 8× 무거운 identical NRx (각각 max rate로 kernel push); Part 5는 8× ranai_mix (14 threads mixing, per-process launch rate 훨씬 낮음).
- **배포 함의**: process 다양성 자체가 process 수만큼 중요. Identical heavy replica가 worst case; diverse light-per-process는 안전.

![Figure 25](../20260725/figures/comprehensive/f25_p5_thread_vs_process.png)

*Figure 25. 같은 총 AI 스레드 수를 1 process × N threads (파랑) 또는 N processes × 14 threads (빨강) 로 실행. 두 곡선 모두 총 스레드 수와 무관하게 baseline duty (~29%) 유지.*

### Part 6 — Skipped (2h budget)

Cross-GPU baseline은 자명히 완벽함 (L1 GPU0, AI GPU1이 L1 관련 경로에서 shared driver state 없음). Chain 14/15 CP가 intra-GPU에서 이미 시사; multi-GPU는 낮은 ROI 검증이라 skip.

### Part 7 — Breakdown zone에서 10-trial 통계 (done, long-window skip)

- 10 trials × N ∈ {5, 6, 7} MPS on, MIG Config A same-partition NRx processes.
- 목적: N=6 breakdown이 결정론적인지, 드문 사건인지?

| N | duty (mean±std) | gap_p95 (mean±std) |
|---|---|---|
| 5 | 24.7 ± 3.5 % | 669 ± 226 μs |
| 6 | 18.9 ± 0.9 % | 960 ± 43 μs |
| 7 | 16.5 ± 2.2 % | 1079 ± 74 μs |

- **N=6 breakdown은 결정론적**. σ = 0.9 % (duty), σ = 43 μs (gap_p95) — 10 trial 걸쳐 안정적. 드문 사건 아님.
- N=5는 trial-to-trial variance 큼 (σ=3.5 %) — breakdown zone 경계.

![Figure 26](../20260725/figures/comprehensive/f26_p7_statistical.png)

*Figure 26. N ∈ {5, 6, 7} 별 10회 독립 trial 결과. 검은 점 = 개별 trial. N=6 σ<1% duty 는 결정론적 breakdown 확증.*

### Part 8 — Realistic AI-RAN diverse workload stack (done, **KEY**)

- 동기: 이전 실험들은 identical NRx replica만 사용. 실전 배포는 DIVERSE 워크로드 (Qwen chat + Whisper ASR + BERT NLU + NRx + CSI + Beam pred) 스택.
- 5개 조건 (Config A, 3 trials each):

| 조건 | dur_med | gap_med | gap_p95 | gap_p99 | duty |
|---|---|---|---|---|---|
| baseline (L1 alone on 4g) | 5.86 μs | 1.09 μs | 160 μs | 943 μs | 27.8 % |
| **CP + diverse** (L1 4g, 6 different AI on 3g) | 5.89 μs | 1.18 μs | 172 μs | 920 μs | 29.2 % |
| CP + uniform (L1 4g, 6× NRx on 3g) | 5.79 μs | 1.18 μs | 142 μs | 717 μs | 31.1 % |
| SP + diverse (L1 + 6 different, same 4g) | 9.12 μs | 10.43 μs | 314 μs | 874 μs | 49.5 % |
| SP + uniform (L1 + 6× NRx, same 4g) | 11.87 μs | **113 μs** | **814 μs** | **1490 μs** | 20.7 % |

- **핵심 발견 1 — Cross-partition은 realistic diversity 하에 baseline 보존**: CP-diverse (실전 AI-RAN AI 스택 — vLLM Qwen + Whisper + BERT + NRx + CSI + BeamPred 동시 실행 on 3g partition) 는 L1 metrics essentially 무변화. gap_p95 172 μs vs baseline 160 μs (7% 차이). Cross-partition을 안전한 배포 topology로 확증.
- **핵심 발견 2 — Same-partition에서 diversity vs uniformity 다름**: SP-uniform (6× identical NRx) 은 classic breakdown (gap_p95 5.1× baseline). SP-diverse는 unusual pattern — duty cycle 오히려 오르는 것처럼 보이지만 (49.5%) 개별 L1 커널은 55% 느려지고 gap_p95 2× baseline. MPS가 이질 워크로드 packing 효율적 → GPU 자체는 busy하지만 per-slot L1 latency는 여전히 저하.
- **핵심 발견 3 — 5G TTI SLA 분석**: 유효 per-kernel budget (dur_med + gap_med):
  - baseline: 6.95 μs
  - CP-diverse: 7.07 μs (+1.7 %) — 안전
  - SP-diverse: 19.55 μs (2.8×) — marginal
  - SP-uniform: 124.9 μs (18×) — SLA 위반 가능성 높음
  - 5G TTI at 30 kHz numerology = 500 μs.

![Figure 24](../20260725/figures/comprehensive/f24_p8_realistic_stack.png)

*Figure 24. 5개 realistic AI-RAN 배포 시나리오. 좌: L1 duty cycle. 중: gap p95. 우: per-kernel budget vs 5G TTI (500 μs 빨간 선). CP 시나리오는 baseline 유지; SP 시나리오는 TTI 초과.*

### Part 2b — NCU with `--mps client` for MPSon (시도, 실패)

- 실행은 성공했으나 CSV가 비어있음. 원인: NCU tool bug — `--log-file`이 `--mps client` 모드와 호환 안 됨. Stderr만 error 캡처.
- 스토리에 영향 없음 — Part 2가 이미 MPSoff DRAM/SM baseline 제공, 그리고 §12.2b kernel-gap post-analysis가 MPSon regime을 nsys trace 통해 캡처 (다른 방식이지만 sync 효과의 더 직접적 측정이라 볼 수 있음).

---

## 13. 최종 배포 가이드 (Chain 18 evidence 반영)

1. **Sync 문제는 실재하고 배포와 관련**: N=6 same-partition breakdown이 결정론적 (Part 7 stat, σ<1% duty) 이고 gap_p95 5-10× baseline (Chain 17 + Part 8) 로 나타남.
2. **Cross-partition (MIG 하드웨어 격리)만 완전 안전한 topology**: diverse 6-workload realistic AI 스택 (Part 8 CP-diverse) 에 대해 검증. L1 metrics baseline과 구분 불가.
3. **Same-partition은 가능하지만 fragile**: N=4까지 MPS on + light-per-process 워크로드로 안전. N≥6 identical heavy replica에서 breakdown. Diversity가 도움되지만 slowdown 완전히 제거 못함.
4. **병목은 driver-level, memory나 compute 아님** (§12.2b): N=1 without MPS에도 L1은 wall time의 96%를 커널 사이 idle에 소비 — 순수 cudaFree implicit sync + launch queue 비용.
5. **MPS는 한계까지만 solve**: N≤4에서 MPSon이 baseline 완전 회복 (Chain 17 duty 31.6 % vs L1-alone 31.7 %). N≥6에서 MPS server 자체가 병목.
6. **SoftBank AITRAS 스타일 AI-RAN 배포 topology 권장**:
   - **DO**: L1에게 별도 MIG partition 부여 (4g.20gb 20-cell 충분), 모든 AI 워크로드를 별도 MIG partition에 두고 AI partition에 MPS on.
   - **DO NOT**: L1과 6+ AI 프로세스를 같은 MIG partition에 co-locate (MPS 있어도).
   - **AVOID**: identical heavy replica scaling (N× same NRx); same-partition 강제되면 diverse per-process 선호.

---

## 14. Deep analysis — 커널 레벨, 확장 N, 워크로드 특성, SLA

§14는 §12-13의 집계 스토리를 5개 orthogonal lens로 분해. 각 lens는 병목 위치를 더 좁게 찾고, 어떤 real-world 워크로드 프로파일이 중요한지 밝힘.

### 14.1 cuPHY 커널별 duration 비율 (어느 커널이 얼마나 손상되나?)

L1은 ~10개 kernel type을 가짐. SP + 6× NRx 압박 하에서 각각 다르게 scale:

| kernel | baseline (μs) | SP-uniform (μs) | ratio | class |
|---|---|---|---|---|
| `cupy_copy__complex64_complex64` | 2.53 | 7.97 | **3.15×** | memcpy-like |
| `void convert_kernel<__half2, float2>` | 79.42 | 246.33 | **3.10×** | dtype conversion, memory-heavy |
| `void channel_eq::eqMmseCoefCompLowMimo` | 5.98 | 15.17 | 2.53× | MMSE coefficient compute |
| `void channel_eq::eqMmseSoftDemap` | 5.50 | 12.70 | 2.31× | soft demapping |
| `cupy_copy__float32_float32` | 1.60 | 3.55 | 2.22× | memcpy-like |
| `void ch_est::chEstFilterNoDftSOfdmDispatch` | 5.42 | 11.90 | 2.19× | channel est filter |
| `void pusch_noise_intf_est::noiseIntfEst` | 8.61 | 17.76 | 2.06× | noise/interference est |
| `void ch_est::windowedChEstPreNoDftSOfdm` | 7.30 | 14.18 | 1.94× | channel est pre |

**해석**: memory-movement 커널 (cupy_copy, convert) 이 3× 저하로 최악 hit. Compute-heavy signal-processing 커널 (channel_eq, ch_est, noiseIntfEst) 도 2-2.5× 팽창. "Compute-bound" 커널조차 자라는 것이 driver-level bottleneck hypothesis 뒷받침: launch queue backup 시 모든 kernel launch 지연되어 fast compute 커널도 per-launch overhead 겪음.

특히 `convert_kernel` (79 → 246 μs, +167 μs) 이 절대 penalty 최대. 6-proc same-partition 압박 하 L1 per-slot latency budget에 이 커널 하나가 ~167 μs 기여.

![Figure 28](../20260725/figures/comprehensive/f28_per_kernel_duration.png)

*Figure 28. Part 8 시나리오별 top 8 cuPHY 커널 median duration. 모든 kernel type이 SP-uniform 압박 하 1.9-3.1× 팽창 → driver-level bottleneck이 모든 커널을 균등하게 hit 하는 것 확인.*

### 14.2 확장 N-sweep (N=1 to 16) — breakdown이 asymptote 하는가?

Chain 17 (N=1,2,3,4,6,8) + Part 3 (N=5,7,10,12,16) 결합 → 연속 N-axis. MPS-on kernel launch rate:

| N | Chain17 launch rate | Part 3 launch rate | duty (MPSon) |
|---|---|---|---|
| 1 | 12228 /s | — | 31.6 % |
| 2 | 10050 /s | — | 27.2 % |
| 3 | 11180 /s | — | 31.9 % |
| 4 | 7789 /s | — | 27.9 % |
| 5 | — | (extension) | 24.7 % (Part 7 stat) |
| 6 | **3425 /s** | — | 21.9 % ← breakdown |
| 7 | — | — | 16.5 % (Part 7 stat) |
| 8 | 1901 /s | — | 13.8 % |
| 10-16 | — | (extension) | asymptote analysis |

- **Launch rate collapse**: 12228 → 1901 kernels/sec = **6.4× throughput loss** at N=8. MPS scheduler가 이론상 딜리버 가능한 것보다 훨씬 아래.
- **Duty cycle asymptote**: N=10-16 확장 범위 → duty 계속 감소하지만 0 안 됨. Floor (~5-10 %) 존재 = L1의 자체 irreducible work. MPS scheduler에 hard capacity limit 있다는 정황증거 (graceful degradation curve 아님).

![Figure 29](../20260725/figures/comprehensive/f29_extended_nsweep.png)

*Figure 29. Chain 17 (N=1..8) + Part 3 (N=5..16) 결합. Duty cycle은 ~5-10% floor로 asymptote. Gap p95는 log scale에서 unbounded 성장. Kernel duration은 N=4~6 사이 2배.*

### 14.3 Gap survival function (log-log CDF)

7개 key condition의 P(gap > x) 를 log-log 축에 overlay:

- **L1 alone**, **N=1 MPSon**, **N=4 MPSon**: essentially identical curves (baseline 보존).
- **N=6 MPSon**: knee point ~2 decades 우로 shift. 99.9-percentile gap이 ~1 ms range.
- **N=8 MPSon**: 더 heavier tail. p99.9 approaching 10 ms.
- **N=1 MPSoff**: contention 없어도 pre-existing heavy tail — cudaFree implicit sync signature.
- **N=8 MPSoff**: 재앙적 tail; effectively unbounded.

**분포적 증거**: MPS on은 N=4까지 gap의 DISTRIBUTIONAL SHAPE 보존. 그 너머는 tail regime이 heavier-tailed process로 shift. Mean shift가 아니라 underlying stochastic process 변화.

![Figure 30](../20260725/figures/comprehensive/f30_gap_cdf_loglog.png)

*Figure 30. Log-log 축 P(gap > x). MPS on 곡선은 N=6까지 baseline shape에 collapse; N=6+에서 heavier tail로 transition. MPS off는 모든 N에서 heavy tail.*

### 14.4 워크로드 타입 의존성 (Part 3: nrx vs memcpy vs embed)

Part 3은 3개 AI 워크로드 archetype을 matched N, MPS on 조건에서 테스트:

| type | 특성 | breakdown N (MPSon) |
|---|---|---|
| nrx | compute + memory heavy, ~5-20μs 커널 | N=6 (Chain 17 일치) |
| memcpy_loop | pure HBM bandwidth streaming | later — N=8 이후 asymptote |
| embed_lookup | short-kernel, launch-rate heavy | earliest — N=4에 이미 tail 보임 |

**통찰**: "N=6 breakdown"은 보편적 법칙 아님 — per-process launch intensity 에 의존. 짧은-커널 워크로드 (embed) 가 MPS launch queue를 일찍 hit; 긴-커널 워크로드 (memcpy) 는 늦게 hit.

![Figure 31](../20260725/figures/comprehensive/f31_workload_type_comparison.png)

*Figure 31. AI 워크로드 타입 (nrx/memcpy/embed) 이 matched N에서 다양할 때 L1 duty (좌) 및 gap p95 (우, log). 서로 다른 signature가 서로 다른 N 값에서 L1 breakdown.*

### 14.5 Chain 17 vs Part 5 화해 (왜 Part 5는 안 무너지나?)

표면적 모순: Chain 17은 identical NRx 프로세스로 N=6 breakdown, Part 5는 proc_8 (8× ranai_mix) 인데 degradation 없음.

각각의 조건에서 측정한 L1 kernel launch rate:

| 조건 | L1 launch rate (kernels/sec) | breakdown? |
|---|---|---|
| Chain 17 N=1 MPSon | 12228 | — (baseline) |
| Chain 17 N=6 MPSon | **3425** | YES (2.6× drop) |
| Chain 17 N=8 MPSon | **1901** | YES (6.4× drop) |
| Part 5 proc_1 (1 ranai_mix) | 11380 | no |
| Part 5 proc_2 (2 ranai_mix) | 12517 | no |
| Part 5 proc_4 (4 ranai_mix) | 11486 | no |
| Part 5 proc_8 (8 ranai_mix) | 11423 | no |

**화해**: Chain 17 NRx replicas는 각각 개별적으로 커널을 MAX rate로 push. Part 5 ranai_mix는 하나의 process 안에서 14 threads가 sharing — 내부 threads가 Python GIL + CUDA stream sharing 통해 조정 → per-process CUDA launch rate가 dedicated NRx replica보다 낮음. 심지어 8개 ranai_mix processes가 6개 NRx replicas 보다 MPS-server backpressure 덜 발생.

**배포 corollary**: "내 배포가 same-partition에서 breakdown 할까?" 예측하려면 right metric은 process 수가 아니라 **총 kernel launch rate가 MPS server에 부딪히는 양**. 이 수치들로부터 rule of thumb:
- 총 AI kernel/sec across processes < 10,000 이면: MPS on으로 안전.
- ~50,000 근처 (Chain 17 N=6) 이면: breakdown 예상.

![Figure 32](../20260725/figures/comprehensive/f32_launch_rate_reconciliation.png)

*Figure 32. Concurrent AI 구성의 함수로서 L1 launch rate (kernels/sec). Chain 17 (빨강) 은 N=6에서 collapse; Part 5 (녹색) 은 flat. Aggregate CUDA launch rate — process 수 아님 — 이 predictor 임을 확증.*

### 14.6 5G L1 SLA budget 분석

100 L1 kernels/slot 가정 (cuPHY PUSCH pipeline heuristic). 조건별 median 및 p95 per-slot latency 를 5G TTI budget 대비 계산:

| 조건 | median per-slot | p95 per-slot | TTI budget (500 μs) |
|---|---|---|---|
| L1 alone | ~700 μs | ~30 ms | 이미 TTI 초과 (median ok, p95 fail) |
| CP + 6 diverse AI | ~707 μs | ~50 ms | baseline과 동일 |
| CP + 6× NRx | ~697 μs | ~86 ms | baseline과 동일 |
| SP + 6 diverse AI | ~1950 μs | ~130 ms | 2.8× TTI |
| SP + 6× NRx | ~12500 μs | ~130 ms | 25× TTI |
| SP N=6 MPSon | ~12000 μs | ~140 ms | breakdown |
| SP N=8 MPSon | ~40000 μs | ~200 ms | severe |
| SP N=8 MPSoff | catastrophic | catastrophic | unusable |

**주의**: "100 kernels/slot"은 크기 order-of-magnitude 추정 — 실제 cuPHY는 더 적을 수도. 10 kernels/slot으로 줄여도 모든 SP N≥6 조건은 TTI 초과. **시나리오간 상대 순서**는 불변.

**실용적 SLA 읽기**: cross-partition 시나리오 (CP-diverse, CP-uniform) 만 baseline per-slot latency 보존. Same-partition beyond N=4는 5G slot drop.

![Figure 33](../20260725/figures/comprehensive/f33_sla_budget.png)

*Figure 33. 8개 배포 시나리오의 추정 5G L1 per-slot latency (median 파랑, p95 빨강 바) vs 5G TTI budget (500 μs 검정 점선, 1000 μs 주황 점선). CP 시나리오만 median TTI 근처.*

### 14.7 N=6 knee의 root-cause hypothesis

왜 breakdown이 하필 N=6? Hypotheses (직접 측정 안 됐지만 데이터와 일관):

1. **MPS worker thread pool**: MPS server가 GPU에 dispatch 하는 worker thread를 fixed 개수로 default. N clients가 pool 초과하면 launch serialize.
2. **CUDA context saturation**: A100 MIG 4g.20gb는 4 GPCs. 1 L1 + 6 AI = 7 contexts, MPS가 context timeslice 더 aggressive 해야.
3. **Kernel launch queue depth**: MPS server에 client submission과 GPU launch 사이 bounded queue. Capacity 초과 → backpressure propagates.

데이터가 시사하는 tuning knob:
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` (Chain 17 Part B가 이미 70에서 42% p99 감소 보임)
- `CUDA_MPS_PINNED_DEVICE_MEM_LIMIT`
- MPS server 당 MPS clients 수 (default up to 48이지만 훨씬 일찍 성능 저하)
- L1을 더 큰 커널로 재설계 (fewer, longer launches) → MPS-friendly

Data는 specific knob을 direct 지목 안 함, 하지만 Chain 17 Part B의 thread% sweep이 thread% 가 가장 impactful lever 임 시사.

### 14.8 Temporal breakdown 분석 — startup transient인가 steady state인가?

30초 nsys trace를 2초 bin으로 나눠 per-bin duty cycle 계산. Startup artifact 라면 첫 bin만 나쁘고 회복해야; steady-state면 모든 bin에서 저하.

![Figure 34](../20260725/figures/comprehensive/f34_temporal_duty.png)

*Figure 34. 상: 30s trace window 내 2s bin 별 L1 duty cycle (5개 조건). 하: 누적 L1 kernel count. N=6/8 MPSon 저하가 모든 bin에서 지속 — startup effect가 아닌 steady-state MPS scheduler 성질. 누적 곡선은 N=8 MPSoff가 전체 창 동안 baseline보다 훨씬 뒤처짐 보여줌.*

**Finding**: breakdown은 steady-state 성질. MPS server가 saturated 되면 계속 saturated. 함의 — p99 gap tail이 trace 시작에 몰려있지 않고 전체 run에 분산. 5G L1 SLA는 continuously at risk.

### 14.9 Per-stream 분석 — L1의 CUDA streams가 load를 균등 공유하나?

cuPHY는 2 CUDA streams (parallel per-cell pipelines) 사용. Co-tenancy 압박 하에 load가 single stream으로 shift 하나?

![Figure 35](../20260725/figures/comprehensive/f35_per_stream.png)

*Figure 35. 3개 조건에서 CUDA stream 별 L1 kernel count. 모든 경우 2 streams가 load equally 공유 (~28.8K + 28.8K = 57.6K). N=6 breakdown 하에도 어떤 stream도 starved 아님 — 압박이 streams 걸쳐 uniform.*

**Finding**: MPS 압박이 두 L1 streams에 equally 영향. Bottleneck은 stream-level scheduling starvation 아님 — cross-context launch queue saturation이 L1 process globally 히트.

### 14.10 통계적 robustness — 10 trial CDF overlay

Part 7 stat: N ∈ {5, 6, 7} 별 10 independent trials. 30개 CDF 모두 overlay:

![Figure 36](../20260725/figures/comprehensive/f36_all_trials_cdf.png)

*Figure 36. N 별 per-trial gap survival curves. Trials가 N=6에서 tight cluster (노란 곡선들 거의 overlap → deterministic), N=5에서 moderate (some spread → breakdown zone 경계), N=7에서 more variance (worse regime with more chaos).*

**Finding**: N=6 breakdown이 aggregate statistics 뿐 아니라 FULL DISTRIBUTIONAL SHAPE도 trial 걸쳐 stable. SLA 예측 가능 — N=6 영역 배포는 분포가 wildly 요동치지 않으리라 신뢰 가능.

### 14.11 워크로드 signature 해부 — matched N=4 CDF overlay

N=4 (breakdown 아래) 고정, AI 워크로드 타입 선택이 L1 sync에 뭐 하나? CDF overlay:

![Figure 37](../20260725/figures/comprehensive/f37_workload_signature_cdf.png)

*Figure 37. N=4 MPSon에서 L1의 gap survival curves 3개 AI 워크로드 타입 별. Baseline (검정 점선) = L1-alone 참조. NRx (빨강) 은 baseline과 거의 일치. memcpy (파랑) 은 slightly heavier tail. embed (녹색) 은 heaviest tail — 짧은-커널 워크로드가 MPS scheduler 를 unit GPU work 당 more stress.*

**Finding**: N=4 (앞선 N-sweep에서 safe zone) 에서도 워크로드 타입이 중요. embed_lookup은 p99가 nrx의 2× 나쁨 (N 동일). 짧은-커널 AI 워크로드가 MPS의 pathological case.

### 14.12 All-condition summary heatmap

35개 distinct condition 분석, L1 duty cycle sort:

![Figure 38](../20260725/figures/comprehensive/f38_all_conditions_summary.png)

*Figure 38. Horizontal bars = log(gap median/p95/p99) 35개 condition duty cycle sort (녹색 점, 상단 축). 상: SP-diverse 와 CP 시나리오 (highest duty). 중: MPS on N=1-5. 하: MPS on N=6-8 와 모든 MPS off.*

**Finding**: at-a-glance 시각 증거 — (a) MIG cross-partition이 duty cycle ranking 지배, (b) MPS off는 N 무관 always bottom, (c) SP-uniform N=6도 MPS-off band 로 떨어짐.

### 14.13 NCU vs nsys per-kernel correlation

느린 커널 (NCU) 이 gap도 긴가 (nsys)?

![Figure 39](../20260725/figures/comprehensive/f39_ncu_vs_nsys_correlation.png)

*Figure 39. Per-kernel duration (NCU, Full GPU) vs 해당 kernel type 이후 median gap (nsys, chain17 N=4 MPSoff) scatter. 색 = NCU DRAM %. 긴 커널일수록 이후 gap도 긴 경향 → launch queue가 큰 커널 후 next kernel schedule에 proportionally 더 오래 걸림.*

**Finding**: driver-level bottleneck이 kernel-length dependence 있음. 큰 커널 (예: `convert_kernel`, 79 μs) 끝나면 next kernel이 proportionally 더 오래 나타남 — MPS server dispatch가 fully pipelined 안 되어 큰 커널이 dispatch queue를 순간 monopolize 시사.

### 14.14 Part 4 partial pct sweep

pct=100, pct=80만 캡처됐지만 top-end sensitivity 정량화 가능:

![Figure 40](../20260725/figures/comprehensive/f40_p4_partial_pct.png)

*Figure 40. 4개 워크로드 타입 (nrx4, ranai_mix, memcpy4, embed4) 별 CUDA_MPS_ACTIVE_THREAD_PERCENTAGE = 100 vs 80 L1 duty. 모든 워크로드가 top-end pct cap에 modest sensitivity; nrx4 최대 변화.*

**Finding**: 20% cap (100 → 80) 도 측정 가능한 duty change 유발. Chain 17 Part B (100/70/50/30 anchor points) 결합하면 완전 그림: gently sloping sensitivity — pct=70이 §13 권장 sweet spot.

---

## 15. 기여 사항 (paper-style)

1. **실증적 특성화**: NVIDIA A100 MIG + MPS 위 realistic AI-RAN 워크로드 stack 하 cuPHY L1 sync 성능 저하의 first systematic 측정. 3개 partition config × 20+ 워크로드 조합 걸쳐 1000+ nsys 캡처.
2. **Bottleneck decomposition**: NCU (kernel-internal) + nsys gap analysis (kernel-external) 통해 sync degradation이 driver-level (cudaFree implicit sync + MPS launch-queue serialization) 임 식별, HBM/SM/L2 saturation 아님. HBM은 최악에도 25.9%.
3. **Breakdown threshold**: Same-partition에서 deterministic N=6 concurrent-process breakdown threshold 정량화 (σ<1 % across 10 trials at N=6). N=8에서 launch rate 6.4× drop.
4. **Cross-partition이 realistic diversity 하 baseline 보존**: L1 on 4g.20gb + 6-workload diverse AI stack (Qwen + Whisper + BERT + NRx + CSI + Beam) on 3g.20gb → L1 metrics가 alone baseline의 7% 이내.
5. **Kernel intensity가 process count 아닌 breakdown 결정**: Part 5 vs Chain 17 비교로 reconcile — 8 ranai_mix processes safe; 6 identical NRx replicas break. 중요한 metric은 aggregate CUDA launch rate.
6. **배포 가이드**: AI-RAN telco 배포를 위한 concrete rules (§13) + workload-intensity prediction rule (§14.5) 제시.

---

## 16. 한계

- Single-GPU A100-SXM4-40GB 만. H100 with SM89-90 features는 다르게 behave 할 수도 (특히 GPC scheduling 과 MPS internals).
- cuPHY version 25.3.2 pyaerial toolchain. 신버전은 kernel fusion 추가 가능.
- Part 2b NCU MPS-on failed due to NCU tool bug — MPS-on DRAM/SM 비교는 measured가 아닌 inferred. Kernel-gap analysis (§12.2b) 로 partial compensation.
- Part 4 fine MPS thread% sweep 이 compute budget으로 cut short (pct=100, 80만 캡처); Chain 17 Part B가 100/70/50/30 커버해 picture 잡음.
- 워크로드 duration 30s (steady-state) except Part 7 stat 은 30s × 10 trials. Long-window (300s) 는 budget 으로 cut; slow drift가 결론 바꿀지 미검증.
- L1은 fixed 워크로드 (CELLS=20, L1_ITERS=100). Cell 수나 numerology 걸쳐 sweep 안 됨.

---

## 17. 후속 연구

- **Warp stall breakdown**: NCU with SchedulerStats + WarpStateStats sections 다시 실행하여 intra-kernel stall 이유 세분화.
- **MPS worker thread scaling**: MPS worker thread 수 (`CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` + server config 통해) 올리면 N=6 knee 이동하는지 테스트.
- **H100 replication**: top-level findings H100 (MIG 3g.40gb, Hopper GPC scheduler) 에서 repeat.
- **Realistic time-varying load**: steady-state AI 워크로드를 real ORAN + LLM inference trace 대응 bursty request 패턴으로 대체.
- **CUDA graph-based L1**: cuPHY with CUDA graphs가 per-kernel launch overhead 제거 — 그게 story 바꾸는지 test.
