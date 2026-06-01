# AI-RAN Results Timeline Diagnosis

Date: 2026-06-01  
Scope: `/Users/changjongkim/New_research/cloudlab_results/results`

This document summarizes what the current AI-RAN/MIG experiment archive appears to contain, in chronological order, and how the interpretation evolved.

## 0. Current High-Level Diagnosis

The project has moved through four stages:

1. **2026-05-12: bring-up / smoke tests**
   - Confirmed that real cuPHY L1 runs under no-MIG and MIG layouts.
   - Early evidence showed partition-size sensitivity, but many AI logs were incomplete or only model-loading logs.

2. **2026-05-24: first systematic sweep**
   - Established the first strong story: MIG fragmentation hurts L1, 2g is risky, real AI-RAN workloads disturb L1 more than LLM-like workloads.
   - Also found intrinsic bimodality in L1 timing.
   - But several measurements were noisy, duplicated, failed, or later marked invalid, especially AI throughput.

3. **2026-05-31: large-scale expansion and profiler-based mechanism search**
   - Re-ran baselines and Tier1 with cleaner N=20.
   - Added AI throughput, AI per-op latency, NCU, NSYS, deep-dive stages A-E/Dv2, P3/P4/P5/P7, and 36 figures.
   - Initially suggested cross-partition tail leakage through copy/convert/inter-kernel gaps.
   - Later Dv2 n=10 and figure notes already started revising that story: small-N effects were likely over-interpreted.

4. **2026-06-01: decisive hypothesis test**
   - F saturation matrix tested cross-partition bandwidth/compute/launch stress directly.
   - G coloc tested same-partition L1 + NeuralRx.
   - H dual-concurrent sanity captures confirmed the split.
   - Current best conclusion: **cross-partition MIG isolation is mostly strong under saturation; catastrophic failure is same-partition L1 + NeuralRx co-location, plus standalone small-partition headroom loss.**

## 1. 2026-05-12: Bring-Up / Smoke Test Phase

### Data Present

Directories:

- `20260512_084023` through `20260512_093125`
- `v4_B2_2g10gb_alone_092434`
- `v4_B4_4g20gb_alone_092606`

Artifacts:

- `l1.log`
- `ai_0_qwen7b.log`
- `ai_0_hbm.log`
- `mig.txt`
- two later standalone JSON captures for 2g and 4g baselines

### Representative Results

| Config | Mean | p99 | Notes |
|---|---:|---:|---|
| split-50-50 | 46.1-46.6 ms | 47.9-49.3 ms | repeated stable smoke runs |
| split-40-60 | 66.6 ms | 67.8 ms | likely 2g-ish L1, slower |
| split-60-40 | 52.8 ms | 53.8 ms | intermediate |
| no-MIG clean-ish | 48.0 ms | 49.9 ms | one stable run |
| no-MIG noisy | 57.9 ms | 172.6 ms | early noisy/outlier run |
| v4 2g.10gb alone | 59.3 ms | 61.8 ms | standalone JSON |
| v4 4g.20gb alone | 52.8 ms | 54.3 ms | standalone JSON |

### Interpretation At This Point

This was not yet a paper-grade experiment. It mainly proved:

- The CloudLab/A100/cuPHY/MIG setup can run.
- L1 latency is sensitive to partition layout.
- 2g-like L1 placement is slower than larger placements.
- Some logs are incomplete; Qwen logs often show model loading only, not clean throughput/inference summaries.

### Confidence

Low-to-medium. Useful for operational bring-up and initial intuition, not final conclusions.

## 2. 2026-05-24: First Systematic N=10/N=20 Sweep

Primary summary:

- `20260524/SUMMARY.md`

### Data Present

Main groups:

- `n20_A0_qwen_baseline`
- `n20_A1_prefill`
- `n20_A1_decode`
- `n20_A2_hbm`
- `n20_L1_alone_3g20gb`
- `n20_baseline_gpu1_fullGPU`
- `n10_L1_alone_4g20gb*`
- `n10_L1_alone_7g40gb_MIG`
- `n10_M1` to `n10_M4`
- `n10_D1a/D1b`
- `n10_AR1/AR2/AR3`
- cell scaling `n5_L1_alone_3g_cells{5,10,40}`

### Main Findings Then

| Finding | Evidence |
|---|---|
| MIG single-partition overhead looked small | 7g MIG close to no-MIG in summary |
| Partition fragmentation looked costly | 3g/4g/2g baselines were much slower than 7g/no-MIG |
| 2g L1 was risky | D1 2g L1 alone and qwen both around 70 ms median / 94 ms p99 |
| Bimodality existed even without AI | baseline and HBM cases showed bimodal behavior |
| Real AI-RAN workloads were worse than LLM-like loads | NeuralRx p99 up to 233 ms summary; chanpred/xapp also high |
| Cell scaling had a knee | 20 cells looked efficient; 40 cells saturated badly |

### Important Caveats

The 5/24 data was valuable but messy:

- Some summary files contain duplicated stale lines or failed runs.
- `n10_M2` had stdev=0 anomaly and failed early runs.
- AI throughput measurement was explicitly marked invalid because L1 did not persist for the full AI measurement window.
- Several results mixed noisy early attempts with cleaner reruns.

### Interpretation After 5/24

The working story was:

> MIG is not enough for AI-RAN L1 because partition fragmentation and AI co-tenants create large L1 latency overheads. NeuralRx-like PHY-AI appears more disruptive than Qwen-like LLM inference.

That was directionally useful, but not yet mechanistically precise.

## 3. 2026-05-31: Large-Scale Expansion And Mechanism Search

Primary documents:

- `20260531/EXPERIMENTS_20260531.md`
- `20260531/NCU_ANALYSIS_20260531.md`
- `20260531/NSYS_ANALYSIS_20260531.md`
- `20260531/NSYS_DETAILED_ANALYSIS.md`
- `20260531/NSYS_DEEP_DIVE.md`
- `20260531/figures/FIGURES_README.md`
- `20260531/MIG_L1_ISOLATION_SYNTHESIS.md`

### Data Present

This is the broadest single-day dataset. It includes:

- Full GPU / 7g / 4g / 3g / 2g L1 baselines
- Tier1 phase 1-4 with Qwen variants, NeuralRx, chanpred, xapp, and multi-partition layouts
- AI throughput v2 with persistent L1 wrapper
- AI full matrix
- AI supplement: ResNet and Forecaster
- AI per-op latency matrix
- L1 multi-AI matrix M5-M12
- NCU hardware metrics
- NSYS timeline v1/v2/v3
- Deep-dive A/B/C/D/Dv2/E
- P3 partition sweep
- P4 timeseries
- P5 sustained
- P7 PDSCH TX attempt
- 36 generated figures

### Clean Tier1 Baseline Results

| Config | N | Mean | p99 |
|---|---:|---:|---:|
| Full GPU v2 | 20 | 36.38 ms | 39.18 ms |
| 7g MIG single | 20 | 37.21 ms | 38.60 ms |
| 4g alone | 20 | 39.07 ms | 40.47 ms |
| 3g alone | 20 | 40.20 ms | 41.32 ms |
| 2g alone | 20 | 51.01 ms | 52.14 ms |

Stable conclusion:

- 7g MIG is near Full GPU.
- 2g L1 is clearly worse.
- Partition fragmentation/headroom loss is real.

### Tier1 Co-Tenant Results

| Co-tenant / scenario | Mean | p99 | Initial interpretation |
|---|---:|---:|---|
| qwen_small | 52.78 ms | 67.96 ms | L1 tail increases |
| qwen7b prefill/decode/stress | 52.97-54.23 ms | 69.72-70.76 ms | LLM variants similar |
| chanpred | 53.97 ms | 71.06 ms | LSTM-like workload disrupts L1 |
| xapp | 52.93 ms | 68.85 ms | xApp also disrupts L1 |
| NeuralRx | 60.43 ms | 196.68 ms | extreme outlier |
| M2 2g L1 small | 65.98 ms | 84.45 ms | 2g L1 risky |

### NCU/NSYS Mechanism Search

NCU:

- Per-kernel hardware metrics changed little with AI.
- This suggested L1 kernel execution itself was mostly isolated.

NSYS:

- Inter-kernel gaps and memcpy operations showed tail events.
- Vulnerable transitions included `cupy_copy -> convert_kernel` and `convert_kernel -> cuPHY` stages.

Initial 5/31 interpretation:

> The leakage might be cross-partition runtime/memory queue contention rather than direct per-kernel slowdown.

### Later 5/31 Self-Correction

`figures/FIGURES_README.md` and Dv2 results already start correcting the initial story:

- Dv2 n=10: H2D/D2D/compute/launch/chanpred p99 mostly overlaps or sits near baseline.
- Figure README says the earlier small-N deep-dive effects were false-positive-like.
- Cross-partition saturation hypothesis was already weakening before 6/1.

### Failed Or Invalid Items

- `p7_pdsch_tx`: all 45 logs failed due to `get_tb_size()` API mismatch.
- Some directories are present mostly as logs, summaries, or figures rather than clean raw JSON.
- The 20260531 synthesis is now superseded by the 20260531-20260601 deep dive.

## 4. 2026-06-01: Decisive Hypothesis Test

Primary document:

- `20260601/NSYS_20260601_ANALYSIS.md`

Generated full-data report:

- `all_deep_dive/MIG_L1_ALL_EXPERIMENTS_DEEP_DIVE.md`

### Data Present

Groups:

- `F_saturation`
- `G_coloc`
- `H_dual`
- `I_ncu`
- `J_mps`
- `analysis_F/F_summary.csv`
- `analysis_G/G_summary.csv`

### F: Cross-Partition Saturation Matrix

Question:

> If another MIG partition aggressively consumes bandwidth/compute/launch resources, does L1 p99 inflate?

F tested:

- D2D copies across sizes/streams
- H2D copies
- GEMM compute
- chanpred intensity
- ResNet intensity
- Forecaster intensity
- stacked AI workloads
- kitchen-sink stress

Baseline:

- `F_0_alone`: mean 43.50 ms, p99 53.67 ms

Result:

- 39 non-baseline conditions
- 0 positive p99 inflation cases
- Block mean p99 deltas were all negative or baseline-equivalent

Interpretation:

> The naive continuous cross-partition HBM bandwidth contention hypothesis is not supported.

This is the most important correction to the earlier story.

### G: Same-Partition NeuralRx Co-Location

Question:

> What happens if L1 cuPHY and NeuralRx are in the same MIG partition?

Result:

| Condition | Alone p99 | Coloc p99 | Delta |
|---|---:|---:|---:|
| 3g L1 + NeuralRx same partition | 56.14 ms | 265.32 ms | +372.6% |
| 4g L1 + NeuralRx same partition | 56.00 ms | 356.56 ms | +536.7% |
| 2g L1 + NeuralRx same partition | 61.14 ms | 369.60 ms | +504.5% |

External AI added outside the coloc partition:

- chanpred/resnet/forecaster/qwen/xapp/sat all add similar extra p99, roughly +34-40% relative to coloc.
- External AI type matters much less than the fact that L1+NeuralRx are colocated.

Interpretation:

> Same-partition temporal sharing is catastrophic. The main AI-RAN danger is not cross-partition saturation; it is co-locating in-line PHY-AI with cuPHY L1 inside the same partition.

### H: Dual-Concurrent Sanity Captures

H confirms F/G split:

| Condition | Mean | p99 | Interpretation |
|---|---:|---:|---|
| H baseline 3g alone | 44.09 ms | 45.56 ms | clean baseline |
| H F1 D2D max | 40.33 ms | 43.26 ms | cross-partition saturation safe |
| H F3 GEMM 4096 | 42.97 ms | 44.45 ms | cross-partition compute safe |
| H F5 stack4 chanpred | 44.07 ms | 45.53 ms | stacked external AI safe |
| H kitchen sink | 42.90 ms | 44.65 ms | combined external stress safe |
| H G1 3g coloc + chanpred | 124.87 ms | 355.84 ms | catastrophic coloc |
| H G2 2g coloc + chanpred | 139.63 ms | 368.19 ms | catastrophic coloc |

Interpretation:

> The distribution becomes bimodal under coloc: normal-looking median-ish behavior with catastrophic top-tail frames.

### I/J Status

- `I_ncu` files exist and are included in inventory, but metric-level deep parsing is still an open analysis task.
- `J_mps` directory exists, but no clear summary has been integrated yet.

## 5. Current Correct Project Story

### What We Can Say Confidently

1. **MIG single large partition overhead is small.**
   - 7g MIG behaves close to Full GPU.

2. **Partition fragmentation is real.**
   - 2g L1 standalone is consistently worse than 3g/4g.
   - Small partition headroom is a real deployment risk.

3. **Cross-partition saturation is not the main failure mode.**
   - 6/1 F and H show D2D/H2D/GEMM/AI-stack/kitchen-sink external stress does not inflate L1 p99.
   - The older cross-partition leakage interpretation should be downgraded to exploratory.

4. **Same-partition L1 + NeuralRx is catastrophic.**
   - p99 increases by +373% to +537%.
   - H shows the behavior is bimodal: normal-ish typical frames, catastrophic tail frames.

5. **Throughput metrics are insufficient.**
   - Earlier AI throughput was either invalid or too averaged to expose tail risk.
   - L1 frame p99/p999 and raw distribution shape are the right diagnostics.

### What Should Be Removed Or Softened

Avoid claiming:

- "MIG cross-partition bandwidth isolation fails generally."
- "Fixed total HBM bandwidth is the dominant observed bottleneck."
- "External AI type alone explains L1 p99 inflation."

Better claim:

> MIG provides strong cross-partition capacity/throughput isolation under tested saturation, but it does not solve AI-RAN scheduling when latency-critical L1 and in-line PHY-AI share the same partition. The dominant confirmed failure is intra-partition temporal sharing and burst occupancy, amplified by small-partition headroom loss.

## 6. Recommended Paper Structure From Current Data

### Section 1: Motivation

AI-RAN wants cuPHY L1 and AI workloads on the same A100. MIG looks attractive because it promises hardware isolation.

### Section 2: Baseline Fragmentation Cost

Use 5/31 baselines:

- Full GPU / 7g / 4g / 3g / 2g.
- Show 2g L1 standalone penalty.

Claim:

> MIG fragmentation imposes a real capacity/headroom cost even before AI co-location.

### Section 3: Cross-Partition Isolation Stress Test

Use 6/1 F and H:

- D2D, H2D, GEMM, AI stack, kitchen sink.
- Show no positive p99 inflation.

Claim:

> Contrary to the naive bandwidth-contention hypothesis, cross-partition saturation does not materially disturb L1 in this dataset.

### Section 4: Same-Partition Co-Location Failure

Use 6/1 G and H:

- L1 + NeuralRx same partition.
- 3g/4g/2g coloc p99 explosion.
- H bimodal distribution.

Claim:

> In-line PHY-AI co-location breaks L1 real-time behavior catastrophically.

### Section 5: Operational Guidance

Rules:

- Do not place L1 on 2g if SLA matters.
- Do not coloc L1 and NeuralRx in the same MIG partition.
- Cross-partition AI is safer than same-partition time-sharing, based on current tests.
- Use p99/p999 and bimodality, not throughput, as the acceptance metric.

## 7. Remaining Work

1. Parse `I_ncu` metric-level data to explain coloc failure in hardware terms.
2. Finish/parse `J_mps` if MPS comparison is needed.
3. Export H nsys to SQLite and align NeuralRx kernels with catastrophic L1 frames.
4. Generate revised final figures:
   - timeline of hypothesis evolution
   - partition baseline
   - F negative saturation matrix
   - G coloc p99 explosion
   - H raw CDF/bimodal distribution
5. Audit older reports so they do not overstate cross-partition bandwidth failure.

