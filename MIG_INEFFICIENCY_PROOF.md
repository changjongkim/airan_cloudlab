# MIG 비효율 증명 — 각 metric이 어떤 주장을 증명하는가

> 5/24 5h reservation의 모든 데이터로 publishable한 강한 evidence를 만드는 매핑.

## 0. 증명하려는 주장 — 4개

| # | Claim | 기존 evidence (v1+v2) | 강화 후 (5/24) |
|---|---|---|---|
| C1 | **MIG mean isolation은 marginal** (1.24×) | N=1 측정, Qwen만 | N=20 × 7 workload (Qwen+AI-RAN+control) |
| C2 | **Asymmetric MIG split은 bimodal leakage** | N=4 | N=20 + phase 가설 분리 |
| C3 | **MIG는 HBM-bandwidth 격리 못 함** (소프트웨어 marketing vs hardware reality) | 결과만, 직접 증거 없음 | **dmon ↔ latency correlation 직접 측정** |
| C4 | **5G TTI SLA 보장 못 함** (TTI deadline 위반) | mean+p99 시사 | **deadline miss rate 직접 측정** |

## 1. 각 metric → 어떤 claim 증명

### Metric M1: **per-iter latency array** (`raw_ms` in JSON)
- 출처: `real_l1.py` (이미 저장 중)
- 분석: `analyze_run.py` → CDF + Q-Q plot
- 증명:
  - **C4 직접 증명** — TTI miss rate = (raw_ms > 1.0).sum() / len(raw_ms) × 100%
  - bimodal 시 Q-Q plot에 step 두 개 명확히 보임 → C2 강화
- publishable 표현:
  > "Even under symmetric MIG (split-50-50), 100% of TTIs exceed the 1ms deadline.
  > Under asymmetric (split-60-40 HIGH mode), TTIs exceed deadline by 5×."

### Metric M2: **bimodality score** (gap / intra-cluster spread)
- 출처: `analyze_run.py` `bimodality_score()`
- 분석: score > 2 → BIMODAL, 1-2 → AMBIGUOUS, <1 → UNIMODAL
- 증명:
  - **C2 통계적 confirm** — N=20에서 score 안정적이면 bimodal 실재
  - phase 1 A1a (prefill, score 작아야 H1 confirm) vs A1b (decode, score 작아야 H1 confirm)
  - phase 4 (AI-RAN workloads, score 큰지 작은지로 H1 vs H2/H3 분리)
- publishable 표현:
  > "Bimodality score > 30 for split-60-40 with Qwen-7B mixed prefill/decode (N=20).
  > Score drops below 2 for AI-RAN workloads (Neural RX, channel prediction LSTM),
  > confirming the bimodal phenomenon is workload-phase-dependent (H1)."

### Metric M3: **HBM BW correlation** (Pearson r)
- 출처: `nvidia-smi dmon -s mu` (DMON=1 옵션, dmon_sync.sh)
- 분석: `analyze_run.py` `plot_dmon_correlation()` — run window별 avg HBM BW vs L1 mean
- 증명:
  - **C3 직접 증명** — Pearson r > 0.5이면 "HBM BW utilization과 L1 latency가 양의 상관" = MIG 격리 실패의 직접 증거
  - 동시에 SM util correlation, Power correlation도 측정 → 다른 mechanism 후보 분리
- publishable 표현:
  > "Across N=20 runs of split-60-40 + Qwen-7B, HBM bandwidth utilization and L1
  > latency show Pearson r = X. This is direct hardware-level evidence that MIG
  > partitioning fails to isolate HBM bandwidth, contradicting NVIDIA's MIG
  > whitepaper claim of 'dedicated and isolated memory bandwidth'."

### Metric M4: **SLA miss rate at multiple thresholds**
- 출처: `analyze_run.py` `plot_sla_miss()`
- 분석: per-run % iters > {1ms, 1.5ms, 2ms, 5ms}, + relative threshold (10% over median)
- 증명:
  - **C4 정량 증명** — 5G NR slot deadline (1ms) 위반율
  - URLLC slice 요건 (99.9999% reliability) 위반 즉시 확인
- publishable 표현:
  > "Under MIG split-60-40 HIGH mode, 100% of TTIs exceed the 1ms 5G NR slot deadline,
  > and 47% exceed 2ms. URLLC reliability targets (99.9999%) require <0.0001% miss
  > rate — MIG cannot meet this even in symmetric configuration."

### Metric M5: **bimodal cluster fraction** (HIGH/LOW %)
- 출처: `analyze_run.py` `bimodal` summary
- 증명:
  - phase 1 A0 (Qwen mixed): ~50% HIGH 기대
  - phase 1 A1a (Qwen prefill): ~100% HIGH 기대 (H1)
  - phase 1 A1b (Qwen decode): ~0% HIGH 기대 (H1)
  - phase 4 AR1-3 (AI-RAN): ~0% HIGH 기대 (H1 confirm) or ~50% (H1 reject)

### Metric M6: **Partition cap decomposition** (D1)
- 출처: phase 3 `D1a - D1b` 자동 계산 (phase3_extras.sh 끝의 python heredoc)
- 증명:
  - D1a (split-40-60 + Qwen) - D1b (split-40-60 alone) = **순수 AI leakage**
  - D1b - A baseline (L1 alone full GPU) = **순수 partition cap**
- publishable 표현:
  > "For split-40-60: total observed latency 66.55ms decomposes into 13.13ms partition cap (2g.10gb vs full GPU baseline) + 7.28ms AI cross-partition leakage. The partition cap dominates, suggesting that 'MIG isolation works' claims are confounded by partition size effects."

## 2. 분석 자동화 매트릭스

| Phase | Metric M1 (CDF) | M2 (bimodality) | M3 (dmon corr) | M4 (SLA miss) | M5 (cluster frac) | M6 (decomp) |
|---|---|---|---|---|---|---|
| Phase 1 (Qwen) | ✓ | ✓ ⭐ | ✓ ⭐ | ✓ | ✓ ⭐ | — |
| Phase 2 (multi-AI) | ✓ | ✓ | ✓ | ✓ | ✓ | — |
| Phase 3 (D1+A) | ✓ | ✓ | — | ✓ | — | ✓ ⭐ |
| Phase 4 (AI-RAN) | ✓ | ✓ ⭐ | ✓ | ✓ | ✓ ⭐ | — |

⭐ = 그 phase에서 가장 publishable한 metric

## 3. Publication-grade evidence 4-tier

### Tier 1: 결과만 (v1+v2 기존) — **약함**
- "mean isolation은 1.24×, p99 3.6×, bimodal 발견 (N=4)"
- 한계: N 작음, mechanism 모름, 직접 증거 없음

### Tier 2: 5/24 phase 1+2+3+4 결과 — **강함 (publishable)**
- N=20 통계 + 7 workload + 4 partition config + dmon correlation + SLA miss rate
- 한계: Nsight L2 profile 없음, MPS 비교 없음, single node

### Tier 3: 5/31 30h 추가 — **complete proof**
- + Nsight Compute L2 hit-rate (3g vs 4g 직접 측정)
- + MPS 직접 비교 (CUDA_MPS_ACTIVE_THREAD_PERCENTAGE)
- + B1 model size sweep (isolation factor curve)
- + 8-hour stability run

### Tier 4: 후속 reservation — **definitive**
- + A100 80GB 비교 (다른 MIG profile)
- + 다중 node reproducibility
- + Power efficiency metrics

## 4. Tier 2 (내일) 끝나면 어떤 paper draft 가능?

**Title 후보**:
> "Hardware MIG isolation in AI-RAN: not perfect, not universal, not deadline-safe"

**Abstract draft**:
> NVIDIA MIG (Multi-Instance GPU) is widely viewed as the hardware foundation
> for AI-RAN co-location, enabling shared GPU between 5G L1 processing and
> AI workloads with claimed bandwidth isolation. We measure cuPHY PUSCH RX
> processing on A100 40GB with seven AI workloads (LLM, in-line PHY NN,
> RAN xApp) across two-, three-, and four-partition MIG configurations.
>
> Our findings contradict the claim of full isolation:
>   (i)   Mean isolation is marginal (1.24×) under realistic heavy AI workloads,
>         versus the 11× isolation observed under light AI artifacts.
>   (ii)  Asymmetric MIG splits exhibit intermittent bimodal leakage:
>         50% of measurement windows show +14% L1 latency, the other 50% match
>         baseline. This bimodality is **workload-phase-aligned**, occurring with
>         LLM (prefill burst) but not with in-line AI-RAN workloads (Neural RX,
>         channel prediction).
>   (iii) Direct HBM bandwidth measurement (Pearson r = X) confirms that MIG
>         "isolated bandwidth" claims are inaccurate — chip-level resources
>         (HBM channel arbiter, on-chip NoC) remain shared.
>   (iv)  No tested MIG configuration meets 5G NR 1ms TTI deadline reliably.
>
> We propose that hardware MIG must be augmented with software-level workload
> phase orchestration to meet AI-RAN SLA requirements.

이 abstract draft는 **Tier 2 데이터로 충분히 뒷받침 가능**. Tier 3가 있으면 더 강해짐.

## 5. Tier 2 가능 publication venues

- **NSDI / SIGCOMM / SOSP**: 강함 — systems angle
- **MobiCom / MobiSys**: 매우 강함 — 5G/wireless 직접 연관
- **NetSys / ICDCS**: 적절
- **GLOBECOM / ICC**: 5G 학회, 작업 적합
- **Workshop / arXiv preprint**: tier 2면 충분

마지막 업데이트: 2026-05-23 7:00PM
