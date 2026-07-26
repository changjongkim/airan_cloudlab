# AI-RAN GPU 격리 — 10-slide 프레젠테이션

**논리 흐름**: 문제 → 방법 → 핵심 발견 3개 → 병목 정체 규명 → realistic 검증 → 신규 발견 → 배포 답변

Slide 별로: **Headline (한 문장) · Key figure · 3 punch points**

---

## Slide 1 — Title & Problem statement

### AI-RAN GPU Isolation
**cuPHY L1 (5G) + AI workloads 공유 시 sync degradation의 정체와 안전한 배포 topology**

Subtitle: *CloudLab d8545 · NVIDIA A100 × 4 · cuPHY 25.3.2 · 1,500+ nsys captures · 1.5M measured L1 kernels · Chain 9-18 (20h experiment)*

- **문제**: SoftBank AITRAS-style에서 5G L1과 diverse AI 워크로드를 같은 GPU에 co-locate 했을 때 L1이 얼마나 안전한가?
- **미지수 (연구 시작 시)**: 병목이 HBM 대역폭인가 컴퓨트인가 driver인가? MPS로 해결되나? 어떤 topology가 안전한가?
- **결론 preview**: cross-partition만 안전. 병목은 driver-level. N=6에서 결정론적 breakdown.

---

## Slide 2 — TL;DR (핵심 발견 5개)

### 5가지 핵심 발견 (chain별 독립 검증)

| # | 발견 | 증거 |
|---|---|---|
| 1 | Sync degradation은 kernel launch rate 현상, 메모리 대역폭 현상 아님 | Chain 14 (11 workloads), P32 |
| 2 | MIG cross-partition = 완벽 격리 (diverse 6-workload에도) | Chain 14, Part 8 CP-diverse (Fig 24) |
| 3 | Same-partition N=6에서 결정론적 breakdown (σ<1% duty across 10 trials) | Chain 17, Part 7 stat (Fig 23, 26) |
| 4 | 병목은 driver-level (kernel gap), HBM/SM/L2 아님 | Chain 18 NCU + gap analysis (Fig 21, 28) |
| 5 | Breakdown 예측 = process 수 아닌 aggregate CUDA launch rate | Part 5 vs Chain 17 (Fig 32) |

**Bottom line**: 5G L1 SLA (500 μs TTI)는 **cross-partition topology에서만** 만족.

---

## Slide 3 — 실험 방법론

### Setup

| Config | MIG 구성 | L1 위치 |
|---|---|---|
| A | 4g.20gb + 3g.20gb | 4g (56 SM) |
| B | Full GPU (108 SM) | Full |
| C | 3g.20gb + 2g + 2g | 3g (42 SM) |

**측정**:
- L1 side: `nsys profile --trace=cuda --duration=30 real_l1.py` (cuPHY 20-cell PUSCH)
- AI side: 20+ realistic workloads (Qwen 2.5-3B vLLM, Whisper large-v3, BERT, Qwen-VL, NRx, CSI, Beam pred, ranai_mix, embed_lookup, memcpy_loop 등)
- Depth: NCU per-kernel (Chain 18 Part 2), DCGM 100ms sampling, nsys cuda_gpu_trace

**Scale**: 1,500+ nsys captures, 108 unique conditions (3 configs × 6 N × 2 MPS × 3 trials), ~1.5M measured L1 kernels

*Reference figure: methodology diagram or workload table*

---

## Slide 4 — Finding 2: Cross-partition = 완벽 격리 (Chain 13, 14 CP)

### MIG cross-partition은 13개 realistic AI 워크로드 모두에서 baseline 유지

**Key figure**: `figures/comprehensive/f03_mig_cross_isolation.png`

**결과**:
- Config A CP: cudaFree 1,687-2,065 ms (baseline 1,706)
- Config A CP: L1 p99 38-42 ms (baseline 40)
- **Qwen-7B, Qwen-VL 14GB 모델조차 L1 baseline 유지**

**이유**: MIG가 SM/HBM controller/L2를 하드웨어 레벨로 분리 → cross-partition traffic이 shared driver state 접근 안 함

**함의**: 이건 MIG의 하드웨어 속성 (partition 크기와 무관). 배포의 golden path.

---

## Slide 5 — Finding 3+4: Same-partition N=6에서 결정론적 breakdown

### N=6이 magic number — MPS on에도 duty cycle 30% → 14%로 급락

**Key figure**: [`figures/polished/P23_duty_cycle_headline.png`](../20260725/figures/polished/P23_duty_cycle_headline.png)

**Chain 17 N-sweep**:
- MPS on N=1-4: baseline duty 유지 (~30%)
- **MPS on N=6: 21.9%** (breakdown 시작)
- MPS on N=8: 13.8%

**결정론적 (Part 7 stat, 10 trials × N∈{5,6,7})**:
- N=6: duty **18.9 ± 0.9%**, gap_p95 **960 ± 43μs**
- σ<1% duty → deterministic, rare event 아님

---

## Slide 6 — 병목 정체 규명: HBM/SM 아닌 driver-level

### GPU 자원은 여유 — 커널 사이 driver stall이 진짜 원인

**Key figure**: [`figures/polished/P28_per_kernel_ratios.png`](../20260725/figures/polished/P28_per_kernel_ratios.png)

**NCU per-kernel 측정 (Part 2, Full GPU MPS off)**:
- HBM peak utilization: **25.9%** (74% 여유)
- SM active: ~20% (flat, 모든 조건)
- 커널 duration: N=4까지 unchanged (5.8-6.4 μs)

**모든 cuPHY 커널이 균등하게 1.9-3.1× 팽창** (6-proc same-partition 압박 하):
- `convert_kernel` (fp16↔fp32): 79 → **246 μs** (+167 μs — largest 절대 penalty)
- `cupy_copy` (memcpy-like): **3.15×**
- `channel_eq::eqMmseCoef`: 2.53×
- **모든 커널이 균등하게 slower** → driver-level launch queue 지연이 EVERY kernel을 hit

---

## Slide 7 — 병목 정체 확증: Kernel gap 분석 (money shot)

### MPS off는 N=1에서도 96% wall time을 커널 사이 idle에 낭비

**Key figure**: [`figures/polished/P21_gap_vs_N.png`](../20260725/figures/polished/P21_gap_vs_N.png)

**Kernel-gap 분석 (Chain 17 12 nsys files, ~700K kernels)**:

| N | MPS | gap median | duty cycle |
|---|---|---|---|
| 1 | off | 1.06 μs | **3.55%** |
| 1 | on | 1.15 μs | 31.58% |
| 6 | on | **119.71 μs** (×109 폭발) | 21.93% |
| 8 | on | **379.07 μs** (×345) | 13.84% |

**병목 stack (최종)**:
- ❌ HBM 대역폭 (25.9% peak, 여유 74%)
- ❌ SM compute (~20% flat)
- ❌ L2 cache (오염 있지만 커널 안 흡수)
- ✅ **cudaFree implicit cross-context sync + MPS launch queue serialization + MPS scheduler saturation @ N≥6**

---

## Slide 8 — Realistic AI-RAN stack 검증 (Part 8)

### 6-워크로드 diverse AI 스택으로 cross-partition의 실전 안전성 확증

**Key figure**: [`figures/polished/P24_part8_realistic_stack.png`](../20260725/figures/polished/P24_part8_realistic_stack.png)

**5개 시나리오 (Config A, 3 trials each)**:

| 조건 | gap_p95 | 판정 |
|---|---|---|
| L1 alone (baseline) | 160 μs | 기준 |
| **CP + 6 diverse AI** (Qwen+Whisper+BERT+NRx+CSI+Beam on 3g) | **172 μs (1.1×)** | ✅ Safe |
| CP + 6× NRx on 3g | 142 μs (0.9×) | ✅ Safe |
| SP + 6 diverse AI on 4g | 314 μs (2.0×) | ⚠️ Marginal |
| SP + 6× NRx on 4g | **814 μs (5.1×)** | ❌ SLA violation |

**의미**: 실전 AI-RAN AI 스택 (LLM chat + ASR + NLU + PHY-AI + CSI + Beam) 을 cross-partition에 두면 L1은 baseline과 구분 불가.

---

## Slide 9 — 신규 발견: Breakdown threshold는 partition size 의존

### N=6 breakdown이 universal 아님 — Config B (Full GPU)는 안 무너짐

**Key figure**: [`figures/polished/P54_3config_duty.png`](../20260725/figures/polished/P54_3config_duty.png)

**Chain 17 360 sqlite files 모두 분석 (108 unique conditions)**:

| Config | 크기 | N=1 duty | N=6 duty | N=8 duty | Breakdown? |
|---|---|---|---|---|---|
| B (Full GPU) | **108 SM** | 39% | 43% | 34% | **No** ✅ |
| A (MIG 4g) | 56 SM | 31% | 22% | 15% | Yes (classic) |
| C (MIG 3g) | 42 SM | 30% | 15% | **10%** | Yes (worst) |

**함의**: Breakdown은 **partition의 SM/resource 예산 함수**. MIG partitioning의 hidden trade-off — **isolation 얻지만 same-partition breakdown threshold 낮춤**.

- Full GPU: same-partition에 가장 safe, 그러나 tenant isolation 없음
- MIG 4g: moderate breakdown threshold, some isolation
- MIG 3g: earliest breakdown, most isolation slice

---

## Slide 10 — 배포 답변 & Decision tree

### 실전 AI-RAN topology 결정 규칙

**Golden path (SoftBank AITRAS-style)**:
```
L1 → 별도 MIG partition (4g.20gb, 20-cell 충분)
6-8개 AI workload → 별도 MIG partition (3g.20gb) + MPS on
```

**Decision tree**:
```
AI workload 개수?
├─ 1개 → same-partition OK (MPS on 권장)
├─ 2-4개 → same-partition (MPS on + pct=70) 또는 cross-partition
├─ 5개 → CROSS-PARTITION 권장 (edge of breakdown)
└─ 6+ → CROSS-PARTITION 강제 (same-partition은 SLA 위반)
```

**DO ✅**
- L1 dedicated MIG partition (필수)
- AI partition에 MPS on
- `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE=70` 튜닝 (multi-process p99 42% 감소)
- Framework fusion 활용 (vLLM, torch.compile → launch rate 자동 감소)

**DO NOT ❌**
- L1 + 6+ AI 같은 MIG partition (MPS 있어도 breakdown)
- MIG 없이 Full GPU sharing
- Identical heavy replica scaling (N× same NRx)

**SLA 예측 rule** (from Chain 17 vs Part 5 화해):
- 총 AI kernel/sec < 10,000 → safe
- ~50,000 근처 → breakdown

---

## Backup / Appendix Slides (optional)

- Slide A1: 20260708 catastrophic 이야기의 재해석
- Slide A2: MPS effectiveness ratio (P61, N=1에서 9× → N=8에서 1×)
- Slide A3: 5G TTI SLA budget (P33, cross-partition만 500μs 이내)
- Slide A4: Wall clock completion time (P57, N=8 MPSoff에서 5-10× 오래)
- Slide A5: Extended N-sweep asymptote (P29, ~5-10% floor)
- Slide A6: Statistical robustness (P56 3-trial scatter)

---

## Slide 별 발표 시간 배분 (총 15분 기준)

| Slide | 시간 | 화자 note |
|---|---|---|
| 1 | 1분 | 문제 정의 + 결론 preview |
| 2 | 1분 | TL;DR 5 findings |
| 3 | 1분 | Setup 간단히 |
| 4 | 1분 | Chain 14 CP 결과 |
| 5 | 2분 | N-sweep breakdown 강조 |
| 6 | 2분 | NCU + per-kernel 데이터 |
| 7 | 2분 | Kernel gap analysis (핵심) |
| 8 | 2분 | Part 8 realistic 검증 |
| 9 | 2분 | 신규 partition-size 발견 |
| 10 | 1분 | 배포 답변 요약 |

**핵심 메시지 반복**: "cross-partition은 always safe. same-partition은 partition 크기와 N에 따라 breakdown. 병목은 driver-level."
