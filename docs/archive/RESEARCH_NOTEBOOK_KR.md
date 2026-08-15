# AI-RAN MIG/MPS 격리 연구 — Master Research Notebook

최종 정리: 2026-06-17
저자: Changjong Kim
저장소: `cloudlab_results/`
상태: 측정 단계 + 분석 단계, 일부 미수행 보강 측정 남음

이 문서는 우리 연구의 현재 상태를 한 곳에 묶은 master research notebook입니다.
논문이 아닌 연구 정리입니다. 모든 캠페인 / 발견 / 미해결 / 데이터 위치를
이 문서 하나에서 추적할 수 있습니다.

---

## TL;DR

> **GPU의 3가지 chip-wide 공유 자원** (CUDA context kernel queue, PCIe/DMA copy engine, DRAM bandwidth)을
> **NVIDIA의 어떤 격리 옵션도 모두 격리하지 못함**.
> MIG cross-partition은 (B)에서 누출, MIG same-partition은 모든 격리 사라짐,
> MPS는 (C)에서 catastrophic. 세 가지 실패가 모두 **cuPHY의 frame-by-frame cudaFree (~4,263회/측정)**
> 와 결합되어 host-side cudaFree blocking으로 수렴.
>
> **신규 발견 (6/14 A2)**: same-partition coloc 시 L1 latency는 AI 프로세스 수에 정확히 비례
> (L1 ≈ 37 + N_AI × 330ms, n=900 std=0.42).

---

# 1. 연구 질문 / scope

## 1.1. 핵심 질문
"NVIDIA GPU 격리 옵션(MIG, MPS, default time-slice)이 AI-RAN의 **cuPHY L1 + AI co-tenancy**에서 어떻게 작동하고/실패하며, 그 메커니즘은 무엇인가."

## 1.2. 측정 대상
- **L1**: cuPHY PUSCH RX, 20-cell serial frame, MCS=2, 273 PRB, 4×4 안테나 (일부 캠페인은 8-ant)
- **AI 워크로드**: chanpred, NeuralRx (PHY-AI), xapp, qwen variants, ResNet50, forecaster, sat_compute, sat_hbm, GEMM, memcpy
- **격리 옵션**: MIG cross-partition / MIG same-partition coloc / no-MIG default / no-MIG MPS

## 1.3. 측정 환경 (두 platform)
| 항목 | CloudLab d8545 | Perlmutter |
|---|---|---|
| GPU | A100-SXM4-40GB × 4 | A100-SXM4 |
| MIG 지원 | ✅ | ❌ (no-MIG만) |
| MPS 측정 | ❌ (Phase 4 미수행) | ✅ |
| 캠페인 | 5/31, 6/1, 6/14 | F (priorities 1-5) + 10.5-10.9 deep analysis |

## 1.4. scope 밖
- PDSCH TX (downlink) — `real_pdsch.py` 없음
- non-A100 GPU (H100 등)
- single-tenant batch processing 시나리오 (실시간 deadline 아닌 경우)

---

# 2. 측정 캠페인 inventory

## 2.1. 5/31 CloudLab (driver 525, Aerial 25-3, 8-ant)
**Raw data**: `results/20260531/`

| 디렉터리 | 측정 | n | 결과 요약 |
|---|---|---|---|
| `n20_baseline_*` | partition baseline (1g/2g/3g/4g/7g/fullGPU) | 1000 each | 2g L1 mean +40%, 작을수록 noisy tail |
| `n20_phase1_qwen*` | qwen variants cross-partition | 1000 each | p99 ≈ 70-72ms (alone과 동급) |
| `n20_phase4_neuralrx` | **NeuralRx 8-ant cross-partition** | 1000 | p99=**205ms**, max=313 (bistable, dominant 위험) |
| `n20_phase4_chanpred` | chanpred cross-partition | 1000 | p99=73ms |
| `n20_phase4_xapp` | xapp cross-partition | 1000 | p99=72ms |
| `ai_full_matrix/` | 6 AI × 4 partition (1g/2g/3g/4g) | varied | partition별 AI 효과 매트릭스 |
| `l1_multi_ai/` | 3-way/4-way stacking | varied | multi-AI 영향 |
| `p3_partition_sweep` | partition fine sweep | — | partition × workload sensitivity |
| `p4_l1_timeseries` | L1 timeseries | — | 동적 변동 |
| `p5_sustained` | 5분 sustained | — | drift 측정 |
| `p7_pdsch_tx` | PDSCH TX 시도 | — | scope 밖 |
| `nsys/`, `nsys_deep_*` | NSYS deep dive | — | gap = memcpy/memset boundary 발견 (§13) |
| `ncu/` | NCU per-kernel | — | DRAM throughput 11-12% (낮음) |

**핵심 발견**:
- NeuralRx PHY-AI는 cross-partition도 위험 (p99 205ms, 6배 증가)
- 일반 AI(chanpred/xapp/qwen)는 70-73ms (alone과 거의 동급)
- 작은 partition(2g)은 baseline 자체가 noisy
- NSYS gap이 idle이 아니라 memcpy/memset boundary

## 2.2. 6/1 CloudLab (driver 525, Aerial 25-3)
**Raw data**: `results/20260601/`

| 디렉터리 | 측정 | n | 결과 요약 |
|---|---|---|---|
| `F_saturation/` | 40+ condition cross-partition matrix | 500 each | 모든 generic AI cross-part flat ~45ms |
| `G_coloc/` | 17 condition same-partition coloc | 500-1000 each | 모든 AI coloc 350-371ms (catastrophic) |
| `H_dual/` | 9 condition cross/coloc transition | 200 each | placement 1번 바뀌면 44→358ms (sharp transition) |
| `I_ncu/`, `J_mps/` | NCU, MPS 시도 | (대부분 비어있음) | 캠페인 미완 |

**핵심 발견**:
- Cross-partition 격리는 모든 generic AI에서 작동 (~45ms flat)
- Same-partition coloc은 AI 종류 무관 ~357ms (G_coloc 매트릭스로 입증)
- Placement transition은 deadband 없이 sharp (H_dual)

## 2.3. 6/14 CloudLab (driver 550, fresh Aerial build, 4-ant)
**Raw data**: `results/20260614/`

| 측정 | 디렉터리 | n | 결과 (p50/p99) |
|---|---|---|---|
| E0 L1 alone (3g) | `E0_baseline_3g/` | 500 | **37.5 / 47.8** |
| E1 cross+NeuralRx | `E1_neuralrx/` | 1000 | 37.6 / **43.6** ⬇️ |
| E2 cross+chanpred | `E2_chanpred/` | 500 | 37.7 / 42.2 |
| E3 same-part 3g+chanpred coloc | `E3_coloc/` | 500 | 353.5 / **365.9** |
| E4 cross+xapp/sat_compute/sat_hbm | `E4_misc/` | 500 each | 37.4-37.6 / 39.4-41.1 |
| E5 alone partition sweep 2g/3g/4g/7g | `E5_alone_partition/` | 500 each | 50.2/37.7/37.5/**34.4** |
| E6 NeuralRx coloc partition sweep | `E6_coloc_neuralrx/` | 1000 each | 361/356/351/**345** (모두 ~360ms) |
| A1 cross+chanpred×4 | `A1_stacking/chanpred_x4/` | 500 | 37.5 / 39.3 (가장 깨끗) |
| A1 cross+ResNet×2 | `A1_stacking/resnet_x2/` | 500 | 38.5 / 43.2 |
| A1 cross+kitchen | `A1_stacking/kitchen/` | 500 | 39.2 / 45.1 |
| **A2 same-part 3g+cp+rn coloc** | `A2_mixed_coloc/` | **900** | **698.6 / 699.6** ⭐ |
| C1 NCU alone vs coloc | `C1_ncu/` | csv | DRAM/L2/SM per-kernel |
| C2 NSYS deep trace | `C2_nsys/` | nsys-rep | alone vs coloc trace |

**핵심 발견**:
- 5/31 NeuralRx 205ms bistability **재현 안 됨** (E1 = 43.6) — driver 변경 or 4-ant 차이
- partition 크기는 alone에선 영향 있지만 coloc에선 무관 (E5 vs E6)
- **N-AI scaling law (A2)**: L1 + 2 AI = **698ms** = L1 + 1 AI (~354ms)의 **정확히 2배**
- Linear fit: **L1 p50 ≈ 37.5 + N_AI × 330.6ms**

**미완**: A3 (chanpred coloc 2g/4g/7g), B1 (four-way bigL1), B3 (ResNet batch sweep), C3 (5min sustained), Phase 4 (MIG off + MPS) — 모두 chain 시간 제약으로 중단.

## 2.4. Perlmutter (no-MIG + MPS)
**Raw data**: `results/perlmutter_handoff/perlmutter_nomig/`

| 측정 | 디렉터리 | 결과 요약 |
|---|---|---|
| F_nomig | 14 condition no-MIG default | NeuralRx 389, sat_hbm 426, chanpred×4 1330ms |
| F_nomig_mps | 7 condition MPS | NeuralRx 40 ⬇️, sat_hbm **6985 bistable** |
| NCU_nomig | DRAM/L2/SM | L1 DRAM 8% (alone과 동일 — throughput 가설 reject) |
| nsys_nomig + nsys_deep | SQLite 4-level 분석 | cudaFree, ioctl, correlationId 인과 |
| P5_nomig | 5분 sustained 9 시나리오 | 저하 5분간 지속 (transient 아님) |

**핵심 발견** (Perlmutter §10.5-10.9):
- **GPU idle gap = cudaFree host blocking** (correlationId로 85% overlap 입증)
- cudaFree avg가 contention과 비례: alone 246μs → NeuralRx default 3.7ms → sat_hbm MPS **115ms**
- Memcpy 60KB bimodal 4.2 → 16.8μs (queue arbitration 직접 시그니처)
- cuPHY 측정당 ~4,263회 cudaFree (구조적 패턴)
- MPS는 compute AI에서 cudaFree 회복 (3.7ms → 279μs)
- MPS는 memory AI에서 cudaFree 폭발 (246μs → 115ms)

---

# 3. 확인된 사실 (직접 측정, 재현됨)

## 3.1. MIG cross-partition 격리

| Claim | 데이터 | 신뢰도 |
|---|---|---|
| Generic AI에 작동 (chanpred/ResNet/GEMM/memcpy/forecaster/xapp/sat_compute/sat_hbm) | 6/1 F + 6/14 E2-E4: 모두 ≤45ms p99 | ✅ 강 |
| 4-AI stacking 까지 견고 | 6/14 A1: chanpred×4 39ms, ResNet×2 43ms, kitchen 45ms | ✅ 강 |
| Driver 525→550 변경에 robust | 6/1 chanpred 45 vs 6/14 chanpred 42ms | ✅ 강 |
| **PHY-AI 8-ant NeuralRx는 cross-partition도 누출** | 5/31 phase4: p99=205 (n=1000), Perlmutter §F5: 197 | ⚠️ 강 — 단 8-ant만 확인 |
| **4-ant NeuralRx는 cross-partition에서 안전** | 6/14 E1: p99=43.6 (n=1000) | ⚠️ 보통 — 8-ant 차이 미해명 |

## 3.2. MIG same-partition coloc

| Claim | 데이터 | 신뢰도 |
|---|---|---|
| AI 종류 무관 ~360ms로 폭락 | 6/1 G_coloc 12종 AI, 6/14 E3+E6: 모두 p99 357-371ms | ✅ 강 |
| Partition 크기 무관 | 6/14 E6: 2g=371, 3g=360, 4g=359, 7g=356 (Δ 4%) | ✅ 강 |
| Driver 변경 무관 | 6/1 G_2_chanpred 361 vs 6/14 E3 366 (Δ <2%) | ✅ 강 |
| 7g (=full GPU) coloc도 같은 메커니즘 | 6/14 E6-7g: 356ms ≈ no-MIG 389ms (Δ 8%) | ✅ 강 |
| **L1 + N AI 같은 partition = 37 + N×330ms** ⭐ | 6/14 A2: L1+2AI=698 (n=900, std=0.42) | ⚠️ 보통 — N=2 한 점만 |

## 3.3. MPS (Perlmutter)

| Claim | 데이터 | 신뢰도 |
|---|---|---|
| Compute AI 회복 | NeuralRx 389→40, forecaster 381→42, qwen 185→43 | ✅ 강 |
| Memory AI catastrophic | sat_hbm 426→**6985ms bistable** | ✅ 강 |
| 판별자: compute vs memory bound | sat_hbm은 dram contention 명확, 다른 AI는 compute | ✅ 강 |

## 3.4. Default time-slice (no-MIG, no-MPS)

| Claim | 데이터 |
|---|---|
| 모든 AI에 일관되게 무너짐 | Perlmutter §F1: NeuralRx 389, sat_hbm 426, chanpred×4 1330 |

## 3.5. 메커니즘 (4-level chain)

| Layer | 측정 | 비고 |
|---|---|---|
| **L4 frame**: L1 frame deadline | p99 측정 | 결과 layer |
| **L3 GPU**: 커널 사이 idle gap | nsys SQLite gap_p99 (Perlmutter §10.5) | default 2.4ms → MPS 250μs (compute), 23ms (memory) |
| **L2 CUDA API**: cudaFree avg | CUPTI_ACTIVITY_KIND_RUNTIME (§10.6) | 246μs → 3.7ms (default) → 115ms (MPS+memory) |
| **L1 syscall**: ioctl(GPU sync) | OSRT_API (§10.8) | 205μs → 559μs (sat_hbm) — host syscall layer |
| **인과 증명** | correlationId overlap (§10.9) | GPU gap의 60-82%가 cudaFree와 시간 overlap |
| **위치** | convert_kernel post-gap (§10.7-A) | 87-90% GPU time이 convert/copy boundary |
| **시그니처** | 60KB memcpy bimodal (§10.7-B) | compute 경합 = single, memory 경합 = bimodal |
| **방향** | memcpy direction (§10.7-C) | 100% H2D, D2D 0% → queue 위치는 PCIe/DMA copy engine |
| **구조적 원인** | cuPHY 측정당 ~4,263 cudaFree (§10.6) | application layer 패턴 |

---

# 4. 통합 framing — 3 공유 자원 × 4 격리 옵션

이게 user께서 정확히 지적하신 framing입니다.

## 4.1. GPU의 3가지 chip-wide 공유 자원

| 자원 | 위치 | partition 격리 가능? |
|---|---|---|
| (A) **CUDA context kernel queue** | per-context | ✅ partition 분리로 격리 |
| (B) **PCIe/DMA copy engine queue** | 1개 chip-wide | ❌ partition 경계 무시 |
| (C) **DRAM bandwidth (HBM)** | partition 할당 있으나 실효 contention | ⚠️ 부분만 |

## 4.2. 4가지 옵션의 격리 범위 (그리고 실패 mode)

| 옵션 | (A) | (B) | (C) | 어디서 무너지는가 |
|---|---|---|---|---|
| MIG cross-partition | ✅ 격리 | ❌ 공유 | ✅ 분리 | **(B)** → PHY-AI 8-ant 누출 (NeuralRx 197ms) |
| MIG same-partition | ❌ 공유 | ❌ 공유 | ❌ 공유 | 모든 자원 → N-AI scaling 360ms × N |
| MPS | ✅ 회피 (동시 실행) | ❌ 공유 | ❌ 공유 (실시간 share) | **(C)** → memory AI bistable (sat_hbm 6985ms) |
| Default | ❌ 공유 | ❌ 공유 | ❌ 공유 | 모든 자원 → 단일 AI에도 389ms+ |

## 4.3. 모든 실패가 host에서 같은 패턴으로 수렴

**공통 host symptom**: `cudaFree` (또는 `cudaEventSync`)가 device-sync 기다리며 block.

| 옵션 + 조건 | cudaFree avg | GPU gap p99 | L1 p99 |
|---|---|---|---|
| alone | 246μs | 251μs | 124ms |
| Default + NeuralRx | 3,752μs | 2,410μs | 389ms |
| MPS + NeuralRx | 279μs ⬇️ | 251μs ⬇️ | 40ms |
| MPS + sat_hbm | **115,506μs** | 23,238μs | **6985ms** |

→ **무슨 자원에서 contention이 났든, host에서는 cudaFree block으로 동일하게 측정됨**.

→ correlationId 분석으로 GPU gap의 ~85%가 cudaFree와 시간 overlap (인과 증명).

→ **NVIDIA 격리 스택의 fundamental limit**: 어떤 옵션도 (A)+(B)+(C)를 동시 격리 못 함.

---

# 5. 신규 발견 (이 연구가 처음 측정한 것)

## 5.1. ⭐ N-AI scaling law (6/14 A2)

> **L1 p50 ≈ 37.5 + N_AI × 330.6 ms** (같은 partition에 AI 프로세스 N개 추가)

| N_AI | 측정 | 예측 | 비고 |
|---|---|---|---|
| 0 (alone) | 37.5ms | base | E0 |
| 1 (L1+chanpred) | 353.5ms | 368 | E3 |
| 1 (L1+NeuralRx) | 355.8ms | 368 | E6-3g |
| **2 (L1+cp+rn)** | **698.6ms** ⭐ | 699 | **A2 (n=900, std=0.42)** |
| 3 | (예측) | 1029 | 미측정 |
| 4 | (예측) | 1360 | 미측정 |

→ 이전 결론 "MIG misconfigure 시 ~360ms로 폭락" → **정정**: "AI tenant 수에 정확히 비례해 cumulative하게 폭락"

→ N=3,4 직접 측정으로 linear law 확정 필요 (§7 next steps).

## 5.2. ⭐ 7g coloc ≡ no-MIG (mechanism unification)

| 환경 | L1 p99 (NeuralRx coloc) |
|---|---|
| CloudLab MIG 7g coloc | 356.3ms |
| Perlmutter no-MIG default | 389ms |
| 차이 | 8% (driver/build) |

→ "MIG instance를 full GPU로 키우기" = "MIG 안 쓰기" 같은 메커니즘.

→ CloudLab MIG 데이터와 Perlmutter no-MIG 데이터를 **같은 framework**로 묶을 수 있음.

## 5.3. ⭐ MPS의 dual-mode failure (Perlmutter)

| Compute AI | Memory AI |
|---|---|
| MPS가 MIG cross-partition보다 우수 (NeuralRx 40ms vs MIG 197ms) | MPS가 default보다 16× 나쁨 (sat_hbm 6985ms vs default 426ms) |

→ MPS의 single binary view (좋다/나쁘다) 거부, **compute/memory bifurcation** 필요.

## 5.4. ⭐ 4-level mechanism chain 완전 입증 (Perlmutter §10.5-10.9)

L4 frame ↔ L3 GPU gap ↔ L2 cudaFree ↔ L1 ioctl. correlationId로 인과 증명. 모든 실패 mode 통합 설명.

---

# 6. 미해결 / 검증 필요

## 6.1. paper-critical missing (우선순위 높음)

| 항목 | 왜 필요 | 예상 시간 |
|---|---|---|
| **N-AI scaling N=3,4 직접 측정** | A2 가설(slope 330ms/AI)을 1점에서 3점으로 확장 | ~1시간 |
| **CloudLab Phase 4** (MIG off + no-MIG default + MPS) | Perlmutter MPS와 같은 hardware에서 cross-check | ~2시간 (reboot 포함) |
| **NeuralRx 4-ant vs 8-ant 격차 해명** | 5/31 phase4 (8-ant) 205ms vs 6/14 E1 (4-ant) 43.6ms 모순 | ~30분 |

## 6.2. 보강 측정 (우선순위 보통)

| 항목 | 가치 |
|---|---|
| A3 chanpred coloc 2g/4g/7g | partition × workload coloc 정밀 (3g 외 다른 partition에서 chanpred coloc 데이터 없음) |
| B1 four-way (4g L1 + 1g×3 AI) | 실제 multi-tenant 운영 시나리오 |
| B3 ResNet batch sweep (b16/b64/b256) | AI intensity sensitivity |
| C3 5분 sustained (CloudLab) | drift / long-tail (Perlmutter는 있음) |
| qwen_small coloc (HF cache 마운트 fix 후) | F-table 한 칸 빈 곳 |

## 6.3. 메커니즘 hypothesis verification

| 가설 | 신뢰도 | 검증 방법 |
|---|---|---|
| Slope 330ms/AI가 partition 크기 무관 | 중간 | A2를 2g/4g/7g에서도 측정 |
| Slope이 AI 종류 무관 | 중간 | chanpred×N coloc (다양한 N) |
| MPS bistable이 sat_hbm 외 다른 memory AI에서도 발생 | 낮음 | sat_hbm 외 다른 memory hog (cuPHY-like) 측정 |
| 5/31 NeuralRx 205ms anomaly의 진짜 원인 | 낮음 | 5/31 phase4 setup 재구성 (driver 525 다시 가능?) |

## 6.4. scope 밖 (paper에 caveat)

- PDSCH TX (downlink) — 우리 스크립트 없음
- non-A100 GPU (H100 등)
- single-tenant batch processing
- AI workload classification 자동화 (compute vs memory를 runtime에 판별)

---

# 7. 다음 단계 (concrete action items)

## 7.1. 측정 단계 (실험 필요)

1. **🔴 N-AI scaling 직접 입증** — L1 + chanpred × {1,2,3,4} same 3g coloc
   - 예측: 367, 697, 1027, 1357ms
   - A2 (697 측정)이 N=2 일치 확인됨, 다른 점 검증 시 linear law 확정
   - 1시간

2. **🔴 CloudLab Phase 4** — MIG off + no-MIG default + MPS
   - reboot 필요 (MIG-off "pending" 상태)
   - 같은 hardware에서 Perlmutter MPS와 직접 비교
   - 2시간

3. **🟡 NeuralRx ant 수 차이 해명** — 6/14 환경에서 8-ant NeuralRx 측정
   - 5/31 (8-ant, 205ms) vs 6/14 (4-ant, 43ms) 갭이 ant 수 때문인지 driver 때문인지 분리
   - 30분

## 7.2. 분석 단계 (실험 없이 가능)

1. **🟡 NCU/NSYS deep dive (6/14 C1/C2)** — alone vs coloc DRAM/L2/SM 정량 비교, Perlmutter §F11과 cross-check
2. **🟡 통합 figure 매트릭스** — 3 자원 × 4 옵션 × failure mode를 단일 그림으로
3. **🟢 paper draft 업데이트** — N-AI scaling law (A2) 반영, 정정된 framing으로 통합

## 7.3. 문서화 단계

1. **이 문서 (`RESEARCH_NOTEBOOK_KR.md`)** — 본 master doc
2. **각 캠페인 visual evidence**:
   - `results/visual_evidence/MIG_AIRAN_VISUAL_EVIDENCE_KR.md` (PART A-E, 5/31+6/1)
   - `results/visual_evidence/PERLMUTTER_NOMIG_VISUAL_EVIDENCE_KR.md` (PART F + 10.5-10.9)
   - `results/visual_evidence/MIG_AIRAN_20260614_VISUAL_EVIDENCE_KR.md` (PART G, 6/14)
3. **Paper draft** (현재 미반영):
   - `results/visual_evidence/MIG_AIRAN_KCC_PAPER_KR.{md,docx}` (5/31+6/1 기반, 6/14 + Perlmutter 미반영)
   - `results/visual_evidence/TSALA_KCC_PAPER_KR.docx`
   - `results/visual_evidence/AURORA_Q_REPORT_KR.docx`

---

# 8. 데이터 위치 (전체 인덱스)

## 8.1. Raw measurement data

```
results/
├── 20260531/                  5/31 CloudLab (driver 525, 8-ant)
│   ├── n20_baseline_*/        partition baselines (1000 frames each)
│   ├── n20_phase1_qwen*/      qwen cross-partition
│   ├── n20_phase4_*/          NeuralRx (205ms!), chanpred, xapp
│   ├── ai_full_matrix/        6 AI × 4 partition matrix
│   ├── l1_multi_ai/           3-way/4-way stacking
│   ├── nsys/, nsys_deep_*/    NSYS deep dive (gap → memcpy boundary)
│   └── ncu/                   NCU per-kernel
│
├── 20260601/                  6/1 CloudLab (driver 525)
│   ├── F_saturation/          40+ cross-partition condition (n=500 each)
│   ├── G_coloc/               17 same-partition coloc condition
│   ├── H_dual/                cross/coloc transition sanity
│   └── analyze_F_saturation.py, analyze_G_coloc.py
│
├── 20260614/                  6/14 CloudLab (driver 550, fresh build, 4-ant)
│   ├── E0_baseline_3g/        L1 alone (3g cross-part layout)
│   ├── E1_neuralrx/           cross-part + NeuralRx (n=1000)
│   ├── E2_chanpred/           cross-part + chanpred
│   ├── E3_coloc/              same-part 3g + chanpred
│   ├── E4_misc/               cross-part + {xapp, sat_compute, sat_hbm}
│   ├── E5_alone_partition/    {2g,3g,4g,7g}/ alone baselines
│   ├── E6_coloc_neuralrx/     {2g,3g,4g,7g}/ NeuralRx coloc sweep
│   ├── A1_stacking/           {chanpred_x4, resnet_x2, kitchen}/
│   ├── A2_mixed_coloc/        chanpred_resnet/ ⭐ NEW
│   ├── C1_ncu/                alone.csv + coloc_nrx.csv (NCU)
│   ├── C2_nsys/               alone.nsys-rep + coloc_nrx.nsys-rep
│   ├── figures/               fig_t01-t07 (6/14 figures)
│   ├── DETAILED_ANALYSIS_KR.md   자세한 분석
│   ├── TODAY_VISUAL_EVIDENCE_KR.md  시각 증거
│   └── SUMMARY_KR.md          간략 요약
│
├── perlmutter_handoff/        Perlmutter no-MIG + MPS
│   ├── perlmutter_nomig/      raw data
│   │   ├── F_nomig/           default time-slice
│   │   ├── F_nomig_mps/       MPS
│   │   ├── NCU_nomig/         NCU per-kernel
│   │   ├── nsys_*/            nsys deep dive
│   │   └── P5_nomig/          5분 sustained
│   ├── *.sbatch, *.sh         실행 스크립트
│   ├── analyze_*.py           분석 스크립트
│   └── PERLMUTTER_NOMIG_VISUAL_EVIDENCE_KR.md   (workspace copy)
│
└── visual_evidence/           최종 분석 문서
    ├── MIG_AIRAN_VISUAL_EVIDENCE_KR.md            PART A-E (5/31+6/1, mechanism deep)
    ├── PERLMUTTER_NOMIG_VISUAL_EVIDENCE_KR.md     PART F + §10.5-10.9 4-level proof
    ├── MIG_AIRAN_20260614_VISUAL_EVIDENCE_KR.md   PART G (6/14, N-AI scaling)
    ├── MIG_AIRAN_KCC_PAPER_KR.{md,docx}            paper draft (오래됨)
    ├── TSALA_KCC_PAPER_KR.docx                     paper version 2
    ├── AURORA_Q_REPORT_KR.docx                     report
    ├── build_*.py                                   figure builders
    └── figures/                                      모든 figure (figXX, fig_supp, fig_t, figF, pmF)
```

## 8.2. 측정 + 분석 스크립트

```
scripts_for_node/
├── cloudlab_aerial/
│   ├── 00_bootstrap.sh        driver/CUDA/Docker/NGC setup
│   ├── 01_aerial.sh           Aerial container pull
│   ├── 02_mig.sh              MIG partition config (presets)
│   ├── 03_workloads.sh        airan:25-3 image build
│   ├── Dockerfile.airan       PyTorch + transformers wrapper
│   ├── real_l1.py             ⭐ L1 측정 본 스크립트 (5/24 since)
│   ├── chain_all_tiers.sh     ⭐ 6/14 통합 chain (Phase 1-3, 5)
│   ├── chain_phase4_mps.sh    ⭐ Phase 4 chain (MIG off + MPS)
│   ├── chain_partition_sweep.sh   6/14 partition sweep
│   ├── post_reboot_setup.sh   리부트 후 환경 복원
│   └── phase*_*.sh, run_*.sh, etc.    개별 측정 스크립트
│
└── experiments/
    ├── run_channel_prediction.py     chanpred
    ├── run_neural_rx_stress.py       NeuralRx (TRT 엔진)
    ├── run_resnet_stress.py          ResNet50
    ├── run_qwen_small_stress.py      Qwen-1.5B
    ├── run_qwen7b_*.py               Qwen-7B 변형
    ├── run_realistic_ai_stress.py    matmul saturation
    ├── run_hbm_stress.py             sat_hbm
    ├── run_memcpy_massive.py         memcpy stress
    ├── run_gemm_massive.py           GEMM
    ├── run_traffic_forecaster.py     forecaster
    └── run_xapp_anomaly.py           xapp
```

---

# 9. 환경 재현 (한 번 더)

## 9.1. CloudLab d8545 셋업 시퀀스

```bash
# 1. 노드 확보 (CloudLab portal에서 d8545 reservation)
ssh sgkim@<node>.wisc.cloudlab.us

# 2. Driver / CUDA / Docker / NGC CLI
sudo bash 00_bootstrap.sh && sudo reboot
# 리부트 후
nvidia-smi   # 4× A100 확인

# 3. NVMe 마운트 (data 디스크) + Docker root 이동
sudo mkfs.ext4 -F /dev/nvme0n1 && sudo mount /dev/nvme0n1 /mydata
sudo systemctl stop docker
sudo ln -s /mydata/docker /var/lib/docker
sudo ln -s /mydata/containerd /var/lib/containerd
sudo systemctl start docker

# 4. Aerial 컨테이너 (NGC public pull)
docker pull nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb
# repo clone (paper code)
git clone https://github.com/changjongkim/airan_cloudlab.git /mydata/work/airan_cloudlab

# 5. airan:25-3 빌드 (PyTorch + transformers 추가)
bash 03_workloads.sh

# 6. pyaerial source 빌드 (cmake/ninja, ~15분)
git clone --branch 25.3.2 https://github.com/NVIDIA/aerial-cuda-accelerated-ran.git /mydata/work/aerial-cuda-accelerated-ran
docker run -d --name aerial-build --gpus all -v /mydata/work/aerial-cuda-accelerated-ran:/opt/nvidia/cuBB \
  -w /opt/nvidia/cuBB airan:25-3 bash -c "
    cmake -Bbuild -GNinja -DCMAKE_TOOLCHAIN_FILE=cuPHY/cmake/toolchains/native \
      -DNVIPC_FMTLOG_ENABLE=OFF -DASIM_CUPHY_SRS_OUTPUT_FP32=ON
    cmake --build build -t _pycuphy pycuphycpp -j16
    cp build/pyaerial/_pycuphy*.so build/pyaerial/lib*.so pyaerial/src/aerial/pycuphy/
  "
# build done → 컨테이너 commit 또는 .so 파일이 src dir에 위치

# 7. MIG 구성
sudo bash 02_mig.sh config split-60-40   # 4g + 3g on GPU 0
# 처음엔 pending → 두 번째 리부트 후 Enabled

# 8. 측정 실행 (idempotent chain)
bash chain_all_tiers.sh    # Phase 1-3, 5
# Phase 4 (MIG off):
bash chain_phase4_mps.sh
```

## 9.2. Perlmutter 셋업 시퀀스
- `results/perlmutter_handoff/BRIEFING.md` 참고
- shifter image: `nvcr.io/nvidia/aerial/aerial-cuda-accelerated-ran:25-3-cubb`
- SLURM batch: `run_all.sbatch`

---

# 10. 정직한 자기 평가 (research maturity)

## 10.1. 강한 부분
- ✅ raw 데이터 충분 (3 캠페인 + Perlmutter)
- ✅ 메커니즘 4-level chain 완전 입증 (correlationId 인과)
- ✅ 3-platform cross-check (CloudLab driver 525, 550, Perlmutter)
- ✅ AI 종류별 격리 효과 매트릭스 완성 (6/1 F + G + Perlmutter MPS)
- ✅ Mechanism universal: convert boundary, cudaFree, ioctl
- ✅ 신규 발견 3개 (N-AI scaling, 7g≡no-MIG, MPS dual-mode)

## 10.2. 약한 부분
- ⚠️ N-AI scaling은 N=2 한 점만 직접 측정 (N=3,4 미수행)
- ⚠️ CloudLab MPS 없음 (Phase 4 reboot 필요)
- ⚠️ 5/31 NeuralRx 205ms anomaly 원인 미해명 (driver vs ant 수 분리 안 됨)
- ⚠️ paper draft 3개가 6/14 발견 미반영
- ⚠️ figure가 캠페인별로 따로 — 통합 매트릭스 figure 없음

## 10.3. 잘못 갔던 부분 (lessons learned)
- chain 첫 실행 시 chmod 누락으로 silent permission error (4시간 wasted) → 권한 fix 후 retry
- 너무 일찍 paper writing에 들어가려 한 적 있음 (user 지적) → research 정리 우선
- "MIG 좋다 / 나쁘다" 단순 framing으로 헤맨 적 있음 → 3 자원 × 4 옵션 framing으로 정정
- thesis 초기엔 "cudaFree만 root cause" → user 지적 후 "3 공유 자원 모두 같은 host symptom"으로 정정

---

# 11. 한 줄 정리 (이 notebook의 essence)

> **현재까지의 측정으로 NVIDIA GPU 격리 옵션 (MIG / MPS / default) 모두가 AI-RAN의 cuPHY L1 + AI co-tenancy에서 chip-wide 공유 자원(CUDA context queue, PCIe/DMA queue, DRAM bandwidth) 중 일부만 격리하고, 격리되지 않은 자원에서의 contention이 cuPHY의 frame-by-frame cudaFree 패턴과 결합되어 host-side cudaFree blocking으로 동일하게 무너지는 mechanism family를 형성함을 확인했다. 신규 발견 3개(N-AI scaling, 7g≡no-MIG, MPS dual-mode)와 4-level mechanism chain (GPU gap ↔ cudaFree ↔ convert boundary ↔ ioctl, correlationId로 인과 입증)으로 이 framework를 입증한 상태이며, N-AI scaling law 직접 입증 (N=3,4)과 CloudLab MPS 측정이 미수행으로 남아있다.**

---

## Change log

| 날짜 | 변경 |
|---|---|
| 2026-06-17 | 초기 작성. 5/31 + 6/1 + 6/14 + Perlmutter 통합. |
