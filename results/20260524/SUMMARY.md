# CloudLab d8545 (Wisconsin A100 x4) — 2026-05-24 Experiments

CloudLab system date: 20260523 (CDT). User local: 2026-05-24 (KST).
Reservation: 10:00-15:00 KST = 20:00 CDT 5/23 – 01:00 CDT 5/24.

## Setup

- Node: `d8545-10s10305.wisc.cloudlab.us` (4× A100-SXM4-40GB)
- OS: Ubuntu 22.04 fresh
- Driver: 550.163.01, CUDA 12.4
- Container: `airan:25-3-final` (Aerial 25-3-cubb + PyTorch 2.4.1 + transformers 4.44.2 + pyaerial 25.3.dev1)
- pyAerial source: GitHub aerial-cuda-accelerated-ran tag `25.3.2`
- L1 workload: `real_l1.py` — cuPHY PUSCH RX, 8T8R, 273 PRB, MCS 2, 20 cells (unless noted), 50 iters

## Inventory (21 datasets + 2 logs)

### Phase 1 — Bimodal mechanism (split-60-40, 3g L1 + 2g AI), N=20

| Tag | AI | Verdict | LOW | HIGH | gap | median | p99 |
|---|---|---|---|---|---|---|---|
| n20_A0_qwen_baseline | Qwen-7B full | BIMODAL | 53.21 | 56.60 | 3.39 | 55.73 | 80.27 |
| n20_A1_prefill | Qwen prefill-only | BIMODAL | 52.61 | 55.55 | 2.94 | 53.68 | 76.88 |
| n20_A1_decode | Qwen decode-only | BIMODAL | 54.29 | 59.76 | 5.47 | 56.03 | 87.53 |
| n20_A2_hbm | static HBM 16GB | BIMODAL | 52.49 | 57.07 | 4.58 | 55.45 | 77.60 |

### Baselines

| Tag | Setup | N | median | stdev | p99 | Verdict |
|---|---|---|---|---|---|---|
| n20_baseline_gpu1_fullGPU | full GPU (no MIG), no AI | 15 valid | **39.28** | 2.88 | 55.58 | BIMODAL (35.49/40.76) |
| n20_L1_alone_3g20gb | 3g.20gb MIG, no AI | 20 | **52.54** | 2.14 | 77.65 | BIMODAL (52.29/56.36) |
| n10_L1_alone_4g20gb | 4g.20gb MIG, no AI (parallel with Phase 4) | 10 | 58.29 | 4.97 | 92.35 | (noisy) |
| n10_L1_alone_4g20gb_clean | 4g.20gb MIG, no AI (clean) | 10 | **56.50** | 2.48 | 85.11 | BIMODAL (55.93/60.60) |
| n10_L1_alone_7g40gb_MIG | 7g.40gb MIG single, no AI | 10 | **36.73** | 2.39 | 51.60 | (unimodal-ish) |

### Phase 2 — Multi-partition + multi-AI, N=10

| Tag | preset (L1 + AI) | median | stdev | p99 |
|---|---|---|---|---|
| n10_M1_3way_balanced_AIRAN | 3g L1 + 2× 2g (xapp + chanpred) | 73.94 | 12.59 | 171.43 |
| n10_M2_3way_L1small_mixed | 2g L1 + 3g + 2g AI (qwen7b + qwen_small) | 132.82 | 0.00⚠ | 381.61 |
| n10_M3_3way_asym_AIRAN | 4g L1 + 1g + 2g AI | 63.50 | 2.39 | 150.36 |
| n10_M4_4way_3xApp | 4g L1 + 3× 1g (gpt2 + chanpred + xapp) | 59.64 | 2.46 | 83.47 |

### Phase 3 D1 (split-40-60 = 2g L1 + 3g AI), N=10

| Tag | preset | median | stdev | p99 | Verdict |
|---|---|---|---|---|---|
| n10_D1a_4060_qwen | 2g L1 + Qwen on 3g | 70.71 | 2.39 | 94.41 | BIMODAL (68.77/73.17) |
| n10_D1b_4060_alone | 2g L1 alone | 71.58 | 2.69 | 94.52 | BIMODAL (69.04/73.20) |

### Phase 4 — Real AI-RAN workloads (split-60-40 = 3g L1 + 2g AI), N=10

| Tag | AI | median | stdev | p99 |
|---|---|---|---|---|
| n10_AR1_6040_neuralrx | NeuralRx TensorRT | **78.71** | 5.29 | 233.70 |
| n10_AR2_6040_chanpred | LSTM channel predictor | 69.64 | 2.55 | 98.00 |
| n10_AR3_6040_xapp | xApp anomaly CNN | 73.73 | 3.46 | 108.02 |

### Cell scaling on 3g.20gb L1 alone (no AI), N=5

| cells | mean (ms) | per-cell |
|---|---|---|
| 5 | 18.29 | 3.66 |
| 10 | 36.09 | 3.61 |
| 20 | 52.54 (from n20) | **2.63** ⭐ sweet spot |
| 40 | 147.98 | 3.70 (saturated) |

---

## Key Findings (refined)

### F1. MIG mode itself is essentially free
- 7g.40gb MIG single: 36.73 ms ≈ no-MIG: 39.28 ms (within noise)
- → MIG runtime overhead is negligible

### F2. Partition cap is the real cost — NOT a linear scale
- 7g → 4g: 36.73 → 56.50 (+19, +54%) ← BIG jump
- 4g → 3g: 56.50 → 52.54 (-4!) ← non-monotonic, 3g actually better than 4g (likely noise + saturation)
- 3g → 2g: 52.54 → 71.58 (+19, +36%) ← BIG jump
- **3g.20gb is the saturation point** for 20-cell cuPHY workload

### F3. cuPHY saturates at ~20 cells on 3g.20gb
- Sub-linear scaling 5→20 cells (fixed overhead amortizes)
- **Super-linear 20→40 cells** (2x cells = 2.84x latency) — bandwidth wall
- → 20 cells is at the saturation knee

### F4. Bimodal exists in ALL configurations (incl. baseline)
- Even 7g no-MIG baseline shows BIMODAL — cuPHY pipeline intrinsic
- A2 static HBM ALSO bimodal → H1 (Qwen phase) hypothesis REJECTED
- Mechanism likely H3 (HBM scheduler/L2 directory) or cuPHY launch ordering

### F5. AI leakage is small for single AI (MIG isolation works)
- 3g L1 + Qwen: 55.73, 3g L1 alone: 52.54 → +3.20 ms (+6%)
- 2g L1 + Qwen: 70.71, 2g L1 alone: 71.58 → -0.86 ms (compute-bound, no contention)
- BUT multi-AI: 3g + 2 AI (M1) = 73.94 vs 3g + 1 AI (A0) = 55.73 → +18 ms

### F6. Real AI-RAN workloads cause MORE leakage than LLM
- A0 Qwen (LLM): 55.73 ms (baseline)
- AR1 NeuralRx (TensorRT inline): +23 ms (78.71)
- AR2 ChanPred (LSTM): +14 ms (69.64)
- AR3 xApp (CNN): +18 ms (73.73)
- TensorRT bursty inference disrupts L1 more than steady LLM inference

### F7. Best AI-RAN config still +51% over baseline
- M4 (4g L1 + 3 light AI) = 59.64 ms vs full GPU 39.28 = **+52%**
- URLLC (p99 < 1ms) impossible in ANY MIG config (all p99 > 80ms)

---

## Paper Punchline (revised)

> "MIG provides effectively free isolation when used as a single partition (7g.40gb MIG ≈ no-MIG), but the AI-RAN use case fundamentally requires partition fragmentation to co-locate AI workloads alongside L1 cuPHY. This fragmentation costs 33-82% L1 latency depending on partition size, plus 6-30% additional leakage from chip-level shared resources in multi-AI scenarios. Real-time AI-RAN workloads (TensorRT NeuralRx) cause significantly more L1 disruption than steady LLM workloads. The best feasible MIG configuration for AI-RAN co-location imposes 52% L1 latency overhead. MIG is structurally inadequate for AI-RAN URLLC requirements."

---

## AI throughput measurement attempt (INVALID — see below)

`ai_throughput/` directory contains attempt to measure AI workload throughput alone vs co-located with L1.

**Setup**:
- AI on 2g.10gb partition (qwen_small / chanpred / xapp / neuralrx)
- L1 on neighbor 3g.20gb (or no L1 for alone)
- N=5 per workload × 2 setups

**Result: INVALID** — L1 background container only ran ~10 seconds (real_l1.py 50 iters × ~52ms = 2.6s + container overhead = ~10s), while AI ran for 30 seconds. So AI was mostly running alone:

| Workload | alone (clean) | "with L1" (mostly alone) | true penalty |
|---|---|---|---|
| qwen_small | 31.4 it/s | 33-48 it/s | not measurable |
| chanpred | ~2200 pred/s | ~2200 pred/s | not measurable |
| xapp | ~1865 inf/s | ~1865 inf/s | not measurable |
| neuralrx | 1 run only | not run | n/a |

→ For valid measurement, need L1 to run in LOOP for full AI duration (next reservation).

## Limitations / Next Steps

1. **AI throughput measurement INVALID** — L1 background didn't persist for AI duration. Need wrapper that loops real_l1.py for full 30+ seconds.
2. **No dmon HBM utilization analysis** — collected but not analyzed locally yet.
3. **M2 stdev=0 anomaly** — possibly cross-contamination earlier; not fully diagnosed.
4. **4g vs 3g non-monotonic** — 3g (52.54) < 4g (56.50) at saturation, needs re-test with cleaner N.
5. **5/31 reservation TODO**:
   - Fix AI throughput methodology (persistent L1 loop)
   - MPS+bank-aware comparison
   - Multi-AI count scaling on 3g (3 AI, 4 AI)
   - Full CDF analysis from raw_ms arrays
   - dmon HBM utilization correlation with L1 latency
   - Nsight Systems profile of a sample run
