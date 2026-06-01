# MIG L1 Deep Dive Across All 20260531-20260601 Experiments

Generated from local artifacts by `cloudlab_results/results/analyze_all_mig_l1.py`.

## 1. Scope

- 20260531 files indexed: 2925
- 20260601 files indexed: 736
- Master tables generated in `cloudlab_results/results/all_deep_dive/`:
  - `artifact_inventory.csv`
  - `l1_json_runs.csv`
  - `l1_log_runs.csv`
  - `l1_condition_summary.csv`

## 2. Revised Verdict

The earlier 20260531-only interpretation over-weighted small-N cross-partition tail events. The full dataset changes the conclusion: MIG cross-partition saturation is mostly isolated, while same-partition co-location of cuPHY L1 with NeuralRx is catastrophically bad.

| Question | Full-data answer | Evidence |
|---|---|---|
| Does cross-partition bandwidth saturation break L1? | Mostly no | 20260601 F: 40 saturation conditions, 0 positive p99 inflation cases |
| Does small MIG partition hurt L1 alone? | Yes | 20260531/20260601 baselines: 2g has clearly worse mean/p99 than 3g/4g |
| Does same-partition PHY-AI coloc break L1? | Yes, massively | 20260601 G: p99 +373% to +537% |
| Is the problem continuous raw HBM GB/s contention? | Not supported | F D2D/H2D/GEMM/stack/kitchen conditions do not inflate p99 |
| What remains the real-time risk? | Temporal sharing inside one partition, plus static partition headroom | G/H bimodal coloc and 2g baseline penalty |

## 3. 20260531: Initial Broad Sweep

### 3.1 Tier1 baselines and phase experiments

| Condition | N | Mean ms | p99 ms |
|---|---:|---:|---:|
| n20_baseline_fullGPU_v2 | 20 | 36.377 | 39.176 |
| n20_baseline_7g_single | 20 | 37.208 | 38.596 |
| n20_baseline_4g_alone | 20 | 39.073 | 40.468 |
| n20_baseline_3g_alone | 20 | 40.203 | 41.321 |
| n20_baseline_2g_alone | 20 | 51.010 | 52.143 |
| n20_phase1_qwen_small | 20 | 52.781 | 67.958 |
| n20_phase4_neuralrx | 20 | 60.435 | 196.682 |
| n20_phase4_chanpred | 20 | 53.971 | 71.059 |
| n20_phase4_xapp | 20 | 52.926 | 68.850 |

Interpretation: these early runs established that small partitions are risky and that some AI co-tenant configurations correlate with large L1 tail latency. However, later n=10/n=5 experiments show that not all of this should be attributed to cross-partition bandwidth contention.

### 3.2 Dv2 n=10 phase decomposition

| Scenario | N | p99 mean us | p99 CI | p999 mean us | max mean us |
|---|---:|---:|---:|---:|---:|
| Dv2_0_alone | 10 | 950.4 | 920.4-980.3 | 1005.9 | 2690.8 |
| Dv2_1_H2D_8MB | 10 | 952.3 | 908.0-996.6 | 1131.4 | 14394.7 |
| Dv2_2_D2D_32MB | 10 | 891.8 | 853.3-930.3 | 941.5 | 2517.8 |
| Dv2_3_compute | 10 | 918.3 | 875.3-961.4 | 966.8 | 2398.9 |
| Dv2_4_launch | 10 | 951.7 | 921.3-982.1 | 1043.1 | 23500.4 |
| Dv2_5_chanpred | 10 | 913.0 | 871.3-954.6 | 985.2 | 2016.3 |

Interpretation: Dv2 weakens the simple bandwidth-contention story. H2D/D2D/compute/launch/chanpred all overlap or sit near the alone baseline in p99. Rare max outliers exist, but the central p99 result does not support a monotonic cross-partition saturation effect.

### 3.3 Stage A wall-clock alignment

| Scenario | L1 kernels | p99 gap us | p999 gap us | max gap ms | top1 idle share |
|---|---:|---:|---:|---:|---:|
| A1_S35_2gL1_chanpred3g | 48976 | 1564 | 1612 | 118.8 | 26.3% |
| A2_S34_4gL1_resnet2g | 48976 | 811 | 840 | 106.5 | 28.7% |
| A3_M5c_3gL1 | 48976 | 870 | 1014 | 117.4 | 28.0% |
| A4_M8a_3gL1 | 48976 | 980 | 1185 | 116.1 | 29.3% |

Interpretation: Stage A remains useful as a burst-localization study, but it should not by itself be used as final proof of cross-partition bandwidth failure because later controlled F/Dv2 sweeps do not reproduce positive p99 inflation under saturation.

## 4. 20260601 F: Cross-Partition Saturation Matrix

Baseline F_0_alone: N=10, mean=43.50 ms, p99=53.67 ms.

| Block | Conditions | Mean p99 delta | Worst p99 delta | Worst condition |
|---|---:|---:|---:|---|
| B_D2D | 12 | -17.2% | -15.0% | F_B_D2D_sz64_str1 |
| C_H2D | 6 | -13.5% | -12.2% | F_C_H2D_sz256_str4 |
| D_GEMM | 4 | -16.3% | -12.1% | F_D_GEMM_d2048 |
| E_chanpred | 5 | -15.5% | -9.6% | F_E_chanpred_b256 |
| E_forecaster | 3 | -16.7% | -16.3% | F_E_forecaster_d1024 |
| E_resnet | 3 | -17.4% | -17.0% | F_E_resnet_b64 |
| F_stack_chanpred | 3 | -15.4% | -14.4% | F_F_stack_chanpred_x8 |
| F_stack_resnet | 2 | -16.8% | -16.1% | F_F_stack_resnet_x4 |
| G_kitchen | 1 | -18.6% | -18.6% | F_G_kitchen_all |

Positive p99 inflation count: 0 / 39.

Interpretation: this is the strongest evidence against the naive fixed-HBM-bandwidth contention hypothesis. Even aggressive D2D, H2D, GEMM, workload stacking, and kitchen-sink stressors in other MIG partitions did not increase L1 p99 above baseline.

## 5. 20260601 G: Same-Partition NeuralRx Co-Location

| Condition | N | Mean ms | p99 ms | Relevant baseline | p99 delta |
|---|---:|---:|---:|---|---:|
| G_1a_3g_coloc | 10 | 245.15 | 265.32 | G_0a_3g_alone | +372.6% |
| G_1b_4g_coloc | 5 | 355.20 | 356.56 | G_0b_4g_alone | +536.7% |
| G_1c_2g_coloc | 5 | 363.65 | 369.60 | G_0c_2g_alone | +504.5% |
| G_2_3gColoc_chanpred | 5 | 354.67 | 361.13 | G_1a_3g_coloc | +36.1% |
| G_2_3gColoc_forecaster | 5 | 349.84 | 356.85 | G_1a_3g_coloc | +34.5% |
| G_2_3gColoc_qwen_small | 5 | 351.25 | 359.28 | G_1a_3g_coloc | +35.4% |
| G_2_3gColoc_resnet | 5 | 353.34 | 360.85 | G_1a_3g_coloc | +36.0% |
| G_2_3gColoc_sat_compute | 5 | 351.56 | 359.55 | G_1a_3g_coloc | +35.5% |
| G_2_3gColoc_sat_hbm | 5 | 349.55 | 356.60 | G_1a_3g_coloc | +34.4% |
| G_2_3gColoc_xapp | 5 | 350.16 | 357.92 | G_1a_3g_coloc | +34.9% |
| G_3_3gColoc_het_chanpred_resnet | 5 | 350.02 | 371.20 | G_1a_3g_coloc | +39.9% |
| G_4_3gColoc_homo_2chanpred | 5 | 356.10 | 370.83 | G_1a_3g_coloc | +39.8% |
| G_5_4gColoc_chanpred | 5 | 355.05 | 357.09 | G_0b_4g_alone | +537.7% |
| G_6_2gColoc_chanpred_3g | 5 | 364.10 | 369.60 | G_0c_2g_alone | +504.5% |

Interpretation: G is the decisive dataset. Same-partition L1+NeuralRx co-location causes catastrophic p99 inflation. External AI type matters much less once coloc is active; the coloc condition itself dominates.

## 6. 20260601 H: Dual-Concurrent Sanity Captures

| Condition | N | Mean ms | p99 ms | Max ms |
|---|---:|---:|---:|---:|
| H_F1_D2D_1024MB_str8_l1 | 1 | 40.327 | 43.263 | 53.866 |
| H_F2_H2D_256MB_str4_l1 | 1 | 42.704 | 44.240 | 44.897 |
| H_F3_GEMM_4096_l1 | 1 | 42.968 | 44.448 | 45.174 |
| H_F4_chanpred_b1024_l1 | 1 | 44.185 | 45.760 | 47.169 |
| H_F5_stack4_chanpred_l1 | 1 | 44.074 | 45.528 | 46.182 |
| H_F_kitchen_l1 | 1 | 42.898 | 44.649 | 45.599 |
| H_G1_3gColoc_chanpred_l1 | 1 | 124.870 | 355.836 | 359.215 |
| H_G2_2gColoc_chanpred_3g_l1 | 1 | 139.631 | 368.188 | 370.760 |
| H_baseline_3g_alone_l1 | 1 | 44.090 | 45.562 | 46.780 |

Interpretation: H agrees with F/G: cross-partition saturation captures remain near baseline, while G coloc cases show bimodal behavior where median-like behavior can look normal but p99/max become catastrophic.

## 7. What Should Be Changed In The Paper Story

1. Do not claim that cross-partition MIG bandwidth isolation generally fails. The full F/Dv2 data do not support that.
2. Keep the partition-fragmentation claim: small L1 partitions have worse standalone headroom.
3. Shift the main failure mode to same-partition temporal sharing: L1 + NeuralRx coloc creates massive bimodal tails.
4. Treat early 5/31 cross-partition p99 spikes as exploratory observations requiring statistical qualification.
5. Reframe bandwidth carefully: raw cross-partition HBM bandwidth is not the main culprit; in-partition SM/memory time-sharing and burst occupancy are.

## 8. Open Analysis Gaps

- Export H dual nsys traces to SQLite and align NeuralRx kernels against L1 catastrophic frames.
- Re-run or finish I_ncu/J_mps if needed; current I files exist but need metric-level parsing and J has no obvious summaries.
- Build figures for the revised final story: F negative matrix, G coloc p99 explosion, H bimodal raw distribution, partition baseline headroom.
- Audit old `MIG_L1_ISOLATION_SYNTHESIS.md` language so it does not overclaim cross-partition bandwidth failure.

