# AI-RAN GPU Isolation — 10-slide 프레젠테이션

**저장소**: https://github.com/changjongkim/airan_cloudlab
**환경**: CloudLab d8545 · NVIDIA A100 × 4 · cuPHY 25.3.2 · 1,500+ nsys captures · 20h 실험
**논리 흐름**: 문제 → 방법 → 핵심 발견 3개 → 병목 규명 → realistic 검증 → 신규 발견 → 배포 답변

---

## Slide 1 — Title & 문제 정의

# AI-RAN GPU Isolation
### cuPHY L1 (5G PHY) + AI 워크로드 GPU 공유의 안전한 배포 topology

**Setting**: NVIDIA A100 × 4 · MIG + MPS · cuPHY 25.3.2 · Chain 9-18 (20h, 1,500+ nsys captures, ~1.5M measured L1 kernels)

**연구 질문**:
1. 5G L1 + diverse AI 워크로드 co-locate 시 어디서 뭐가 병목?
2. MPS로 해결되나? MIG로 해결되나?
3. SoftBank AITRAS-style 실전 배포에서 safe topology는?

**결론 preview**:
- 병목 = **driver-level** (HBM 아님)
- **Cross-partition 만 safe**
- Same-partition은 **N=6에서 결정론적 breakdown**

*(This slide has no figure — pure text intro)*

---

## Slide 2 — TL;DR: 5가지 핵심 발견

![Executive dashboard — 5 key findings visual summary](../20260725/figures/comprehensive/f01_executive_dashboard.png)

**Figure 설명**: 상단 3-패널이 5개 finding을 시각적으로 요약. (좌) 커널 launch rate ∝ sync penalty, (중) MIG cross-partition 격리 완벽, (우) MPS on/off 효과 워크로드 클래스별.

| # | 발견 | 증거 chain |
|---|---|---|
| 1 | Sync는 kernel launch rate 현상, 메모리 대역폭 아님 | Chain 14 (11 workloads) |
| 2 | MIG cross-partition = 완벽 격리 | Chain 14 CP, Part 8 |
| 3 | Same-partition N=6에서 결정론적 breakdown | Chain 17, Part 7 (σ<1%) |
| 4 | 병목은 driver-level (kernel gap), HBM/SM 아님 | Chain 18 NCU + gap analysis |
| 5 | Breakdown 예측 = process 수 아닌 aggregate launch rate | Part 5 vs Chain 17 |

**Bottom line**: 5G L1 SLA (500 μs TTI)는 **cross-partition topology에서만** 만족.

---

## Slide 3 — 실험 방법론 (Setup)

![Figure 5 · Config × Workload heatmap](../20260725/figures/comprehensive/f05_config_workload_heatmap.png)

**Figure 설명**: 3개 MIG config × 20+ 워크로드의 MPSoff L1 p99 (baseline 배수). 녹색 = safe, 빨강 = catastrophic. Class A (NRx, ChanPred, memcpy, embed) 이 모든 config에서 red; Class B (Qwen, Whisper, BERT, VL) 는 framework fusion 덕에 green.

**3 Configs**:
| Config | MIG | L1 위치 |
|---|---|---|
| A | 4g.20gb + 3g.20gb | 4g (56 SM) |
| B | Full GPU (108 SM) | Full |
| C | 3g.20gb + 2g + 2g | 3g (42 SM) |

**Measurement stack**:
- L1: `nsys profile --trace=cuda --duration=30 real_l1.py` (cuPHY 20-cell PUSCH, 100 iters)
- AI: 20+ realistic (Qwen 2.5-3B vLLM, Whisper large-v3, BERT, Qwen-VL, NRx, CSI, Beam, ranai_mix, ...)
- Deep: NCU per-kernel, DCGM 100ms, nsys cuda_gpu_trace

**Scale**: 108 unique conditions × 3 trials = 324 base + Chain 18 depth experiments

---

## Slide 4 — Finding 2: Cross-partition = 완벽 격리

![Figure 3 · MIG cross-partition isolation across all workloads](../20260725/figures/comprehensive/f03_mig_cross_isolation.png)

**Figure 설명**: L1이 dedicated MIG partition, 다른 partition에 13개 realistic AI 워크로드. 세로축 L1 cudaFree p99 (ms), baseline과의 편차. 모든 워크로드 (Qwen-7B, Qwen-VL 14GB 포함) 가 baseline band (±20%) 안에 유지.

**핵심 데이터**:
- Config A CP cudaFree: 1,687-2,065 ms (baseline 1,706)
- Config A CP L1 p99: 38-42 ms (baseline 40)
- **13개 워크로드 모두 편차 <20%**

**메커니즘**:
- MIG가 하드웨어 레벨에서 SM/HBM controller/L2/memory controller 분리
- Cross-partition traffic이 shared driver state 접근 안 함
- **Partition 크기와 무관** (Config A와 C 둘 다 완벽)

**함의**: 이건 MIG의 하드웨어 속성. 배포의 golden path.

---

## Slide 5 — Finding 3+4: Same-partition N=6 결정론적 breakdown

![Figure 23 · L1 duty cycle collapses at N=6](../20260725/figures/polished/P23_duty_cycle_headline.png)

**Figure 설명**: L1 duty cycle (kernel time / (kernel + gap)) vs concurrent NRx process 수. **MPS on (녹색)** 이 N=4까지 baseline 31% 유지하다가 **N=6에서 22%, N=8에서 14%로 급락**. MPS off (빨강) 은 어떤 N에서도 3-8%. 노란 zone = breakdown region.

**Chain 17 N-sweep** (Config A, 3 trials each):
| N | MPS off duty | MPS on duty |
|---|---|---|
| 1 | 3.55% | **31.58%** |
| 4 | 7.74% | 27.95% |
| **6** | 3.46% | **21.93%** ← breakdown |
| 8 | 2.79% | 13.84% |

**Part 7 stat** (10 trials × N∈{5,6,7}): N=6 duty = **18.9 ± 0.9%** → **결정론적** (rare event 아님).

---

## Slide 6 — 병목 정체 규명: HBM/SM 아닌 driver-level

![Figure 28 · Every cuPHY kernel slows down 1.9-3.1× uniformly](../20260725/figures/polished/P28_per_kernel_ratios.png)

**Figure 설명**: 8개 dominant cuPHY 커널의 median duration. 검정 = baseline, 빨강 = SP + 6× NRx. 오른쪽 빨간 박스 = 배수. **모든 커널이 균등하게 1.9-3.1× 팽창** — 이것이 driver-level 병목의 증거.

**핵심 데이터**:
- `convert_kernel` (fp16↔fp32): 79 → **246 μs (+167μs)** — largest 절대 penalty
- `cupy_copy(complex64)`: 2.5 → 8.0 μs (**3.15×**)
- `channel_eq::eqMmseCoef`: 6.0 → 15.2 μs (2.53×)

**병목 stack**:
- ❌ HBM 대역폭: **peak 25.9%, 여유 74%** (NCU 실측)
- ❌ SM compute: ~20% flat (모든 조건)
- ❌ L2 cache: 오염 있지만 커널 duration 안 흡수
- ✅ **Driver-level**: cudaFree implicit sync + launch queue serialization + MPS scheduler saturation

**결정적 관찰**: 모든 커널이 균등하게 slow down → 특정 자원 고갈 아닌 **범용 launch queue 지연**

---

## Slide 7 — 병목 확증: Kernel-gap 분석 (money shot)

![Figure 21 · Inter-kernel gap explodes 100-300× at N=6](../20260725/figures/polished/P21_gap_vs_N.png)

**Figure 설명**: L1 커널 사이 gap의 median (좌) 및 p95 (우) vs N. log scale. **MPS on (녹색)**은 N=4까지 gap 1.1μs 유지하다가 N=6에서 **120μs (×109)**, N=8에서 **379μs (×345)** 로 폭발. MPS off (빨강)은 모든 N에서 ms scale tail.

**Chain 17 kernel-gap 실측** (12 nsys files, ~700K kernels):

| 조건 | gap median | gap p99 | duty |
|---|---|---|---|
| L1 alone | 1.12 μs | 700 μs | 31.7% |
| N=1 MPSoff | 1.06 μs | 5371 μs | **3.55%** (96% idle) |
| N=6 MPSon | 119.71 μs | 1377 μs | 21.93% |
| N=8 MPSon | 379.07 μs | 1860 μs | 13.84% |

**핵심 insight**:
- **MPS off는 N=1에서도 96% wall time을 커널 사이 idle** — 순수 per-process driver 비용
- **MPS on은 N≤4에서 baseline 완전 회복** — driver-level sync 해결
- **N≥6에서 MPS server 자체가 병목** — worker thread saturation

---

## Slide 8 — Realistic AI-RAN stack 검증 (Part 8)

![Figure 24 · Cross-partition preserves L1 baseline under realistic diversity](../20260725/figures/polished/P24_part8_realistic_stack.png)

**Figure 설명**: 5개 realistic AI-RAN 시나리오의 L1 gap p95. **녹색 (SAFE)** = cross-partition, baseline과 동일. **빨강 (UNSAFE)** = same-partition, 5.1× breakdown. Bar 아래 배수 (1.1×/5.1×) 로 정량화.

**Part 8 5개 조건** (Config A, 3 trials each):

| 조건 | gap p95 | 배수 | 판정 |
|---|---|---|---|
| L1 alone (baseline) | 160 μs | 1.0× | 기준 |
| **CP + 6 diverse AI** (Qwen+Whisper+BERT+NRx+CSI+Beam) | **172 μs** | **1.1×** | ✅ Safe |
| CP + 6× NRx | 142 μs | 0.9× | ✅ Safe |
| SP + 6 diverse AI | 314 μs | 2.0× | ⚠️ Marginal |
| SP + 6× NRx | **814 μs** | **5.1×** | ❌ SLA violation |

**의미**:
- **Cross-partition은 realistic diverse 워크로드 스택에서도 baseline 완전 보존**
- SP-uniform (6× identical NRx) 이 worst case → 실전 배포 avoid pattern
- SP-diverse는 MPS packing 덕에 partial recovery, 그러나 여전히 SLA 위험

---

## Slide 9 — 🆕 신규 발견: Breakdown threshold = partition size 함수

![Figure 54 · Larger MIG partition = more resilient to N-scaling](../20260725/figures/polished/P54_3config_duty.png)

**Figure 설명**: 3-config × N-sweep duty cycle (MPS on, error bar = 3-trial std). **Config B (Full GPU, purple)** 은 N=6에도 43%, N=8에서만 34% (breakdown 없음). **Config A (MIG 4g, blue)** 는 classic 31%→15% breakdown. **Config C (MIG 3g, orange)** 는 30%→10% (worst).

**Chain 17 360 sqlite files 전체 분석** (108 unique conditions, 이전엔 13개만):

| Config | 크기 | N=1 duty | N=6 duty | N=8 duty | Breakdown? |
|---|---|---|---|---|---|
| **B (Full GPU)** | **108 SM** | 39% | 43% | 34% | **No** ✅ |
| A (MIG 4g) | 56 SM | 31% | 22% | 15% | Yes |
| C (MIG 3g) | 42 SM | 30% | 15% | **10%** | Yes (worst) |

**신규 인사이트**: **N=6 breakdown은 universal MPS 속성 아니고 resource-count 함수**.

**MIG partitioning의 hidden trade-off**:
- Full GPU: same-partition에 가장 safe, tenant isolation 없음
- MIG 4g: moderate breakdown, some isolation
- MIG 3g: earliest breakdown, most isolation slice

---

## Slide 10 — 배포 답변 & Decision tree

![Figure 33 · 5G L1 SLA budget analysis](../20260725/figures/polished/P33_sla_budget.png)

**Figure 설명**: 8개 배포 시나리오의 estimated per-slot L1 latency vs 5G TTI budget (500μs, 검정 점선). **녹색 (CP scenarios)** 만 TTI 근처. SP N=6+ 는 12-40ms (25-80× TTI 초과) — SLA 위반 확정.

### Decision tree

```
AI workload 개수?
├─ 1개    → same-partition OK (MPS on 권장)
├─ 2-4개  → same-partition (MPS on + pct=70) 또는 cross-partition
├─ 5개    → CROSS-PARTITION 권장 (edge of breakdown)
└─ 6+개   → CROSS-PARTITION 강제 (same-partition은 SLA 위반)
```

### DO ✅
- **L1 → dedicated MIG partition** (4g.20gb, 20-cell 충분)
- **모든 AI → 별도 MIG partition + MPS on**
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70` 튜닝 (multi-process p99 42%↓)
- Framework fusion 활용 (vLLM, torch.compile → launch rate 자동 감소)

### DO NOT ❌
- L1 + 6+ AI 같은 partition (MPS 있어도 breakdown)
- MIG 없이 Full GPU sharing
- Identical heavy replica scaling (N× same NRx)

### SLA 예측 rule of thumb
- 총 AI kernel/sec < 10,000 → safe with MPS on
- ~50,000 근처 → breakdown 예상

**핵심 메시지**: *"Cross-partition은 always safe. Same-partition은 partition 크기와 N에 따라 breakdown. 병목은 driver-level."*

---

## Backup slides (optional appendix)

### A1 — Kernel launch cadence: bimodal → broad

![Figure 41 · Kernel launch cadence shift](../20260725/figures/polished/P41_launch_cadence.png)

*Baseline/N=4는 bimodal (10μs + 500μs mode = L1 자연 리듬). N=6+는 broad 분포로 smear — MPS scheduler가 arbitrary delay 주입.*

### A2 — Gap distribution shape shift

![Figure 30 · Gap survival function log-log](../20260725/figures/polished/P30_gap_cdf.png)

*MPS on N=4까지 baseline과 같은 shape 유지 → N=6+에서 heavier-tailed regime으로 transition. Mean shift 아닌 distributional change.*

### A3 — 100ms 활동 timeline (baseline vs breakdown)

![Figure 50 · 100ms GPU activity timeline](../20260725/figures/polished/P50_activity_timeline.png)

*같은 100ms window에 baseline 1,282 커널 (packed) vs N=6 breakdown 528 커널 (visible gaps). 2.4× fewer 커널.*

### A4 — Wall clock completion time

![Figure 57 · L1 workload wall time](../20260725/figures/polished/P57_wall_completion.png)

*L1은 항상 ~57K 커널 launch. 변하는 건 얼마나 오래 걸리나. N=8 MPSoff에서 baseline 대비 5-10× 오래.*

### A5 — MPS effectiveness ratio

![Figure 61 · MPS benefit ratio](../20260725/figures/polished/P61_mps_benefit_ratio.png)

*MPSon/MPSoff duty 비율. N=1 근처 9× benefit, N 커질수록 감소 (MPS 자체가 병목).*

### A6 — Chain 17 vs Part 5 화해 (launch rate 이론)

![Figure 32 · Launch rate reconciliation](../20260725/figures/polished/P32_launch_rate_reconciliation.png)

*Chain 17 identical NRx N=6에서 3425 kernels/sec로 collapse. Part 5 ranai_mix N=8은 11423 kernels/sec 유지. Process 수 아닌 aggregate launch rate가 predictor.*

---

## 발표 시간 배분 (15분 기준)

| Slide | 시간 | 핵심 note |
|---|---|---|
| 1 | 1분 | 문제 정의 + 결론 preview |
| 2 | 1분 | TL;DR 5 findings 표 |
| 3 | 1분 | Setup 방법론 (heatmap 소개) |
| 4 | 1분 | Cross-partition 격리 (Chain 14) |
| 5 | **2분** | N=6 breakdown 강조 (Fig 23) |
| 6 | **2분** | 병목 정체 규명 (Fig 28, NCU 데이터) |
| 7 | **2분** | Kernel gap money shot (Fig 21) |
| 8 | **2분** | Realistic 검증 (Fig 24) |
| 9 | **2분** | 신규 partition-size 발견 (Fig 54) |
| 10 | 1분 | 배포 답변 (Fig 33 + tree) |

**핵심 반복 메시지**: *"Cross-partition은 always safe. Same-partition은 partition 크기와 N에 따라 breakdown. 병목은 driver-level."*

---

## Figure 파일 위치 요약 (슬라이드 만들 때 참조)

| Slide | Figure 파일 |
|---|---|
| 2 | `results/20260725/figures/comprehensive/f01_executive_dashboard.png` |
| 3 | `results/20260725/figures/comprehensive/f05_config_workload_heatmap.png` |
| 4 | `results/20260725/figures/comprehensive/f03_mig_cross_isolation.png` |
| 5 | `results/20260725/figures/polished/P23_duty_cycle_headline.png` |
| 6 | `results/20260725/figures/polished/P28_per_kernel_ratios.png` |
| 7 | `results/20260725/figures/polished/P21_gap_vs_N.png` |
| 8 | `results/20260725/figures/polished/P24_part8_realistic_stack.png` |
| 9 | `results/20260725/figures/polished/P54_3config_duty.png` |
| 10 | `results/20260725/figures/polished/P33_sla_budget.png` |
| A1 | `results/20260725/figures/polished/P41_launch_cadence.png` |
| A2 | `results/20260725/figures/polished/P30_gap_cdf.png` |
| A3 | `results/20260725/figures/polished/P50_activity_timeline.png` |
| A4 | `results/20260725/figures/polished/P57_wall_completion.png` |
| A5 | `results/20260725/figures/polished/P61_mps_benefit_ratio.png` |
| A6 | `results/20260725/figures/polished/P32_launch_rate_reconciliation.png` |

**전체 폴더**:
- Polished figures (v3): https://github.com/changjongkim/airan_cloudlab/tree/main/results/20260725/figures/polished
- Original figures: https://github.com/changjongkim/airan_cloudlab/tree/main/results/20260725/figures/comprehensive
