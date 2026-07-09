# 20260708 CloudLab session — MPS + Time-slicing experiments

## Node
- `d8545-10s10501.wisc.cloudlab.us` (CloudLab AIRANSLICING)
- 4× A100 SXM4 40GB, AMD EPYC 7413, Ubuntu 22.04
- Driver 550.163.01, CUDA 12.4 host / 12.9 container

## Environment build
- cuPHY: aerial-cuda-accelerated-ran repo at tag `25.3.2`
- Container image: `airan:25-3-final` (pyaerial built with x86-64 toolchain due to AMD Milan compat)
- SDK build log: `logs/sdk_build.log` (first attempt, native toolchain — SIGILL), `logs/sdk_build_x86.log` (final, x86-64 toolchain)

## Experiments in this directory

| Directory | Description | Trials | Status |
|---|---|---|---|
| `mps_ts_v2/` | First MPS vs TS at cells=20 with NRx (v2 script) | 3 | ✅ Complete |
| `mps_only/` | MPS mode L1 alone + L1+NRx coloc, cells=20 | 3 | ✅ Complete |
| `mps_hbm/` | MPS mode L1 + HBM stress (mechanism verify) | 1 | ✅ Complete |
| `mps_verify/` | MPS single-run with NRx monitoring | 1 | ✅ Complete |
| `matrix_v2/` | Full matrix: cells(4,10,20,40,60) × workloads(alone,nrx,chanpred,hbm,resnet) × modes(TS,MPS) × 3 trials | 3 each | 🔄 Running |
| `analysis/` | Per-call cudaFree distribution analysis (Phase 1) | — | ✅ Complete |
| `scripts/` | Shell + Python scripts used in this session | — | ✅ Complete |
| `logs/` | stdout / stderr logs | — | ✅ Complete |

## Key findings from Phase 1 (mps_only + mps_hbm)

Per-call cudaFree distribution (30s NSYS window, real_l1.py 20 cells × 100 iters):

| Condition | n | p50 | Fast <1ms | Slow 1-10ms | Cat >10ms | Mechanism |
|---|---|---|---|---|---|---|
| TS_alone | 7,263 | 143µs | 100% | 0% | 0% | no sync |
| **TS + NRx** | 4,847 | **4,725µs** | 4.7% | **95.3%** | 0% | cross-process sync |
| MPS_alone | 7,263 | 146µs | 100% | 0% | 0% | no sync |
| **MPS + NRx** | 7,263 | **167µs** | **99.93%** | 0.07% | 0% | shared context → no sync |
| **MPS + HBM** | 372 | **58,336µs** | 1.9% | 0% | **98.1%** | HBM bandwidth contention |

## MIG comparison (previous CloudLab session — see 20260701/)

| Condition | cudaFree total | vs baseline |
|---|---|---|
| MIG 4g L1 alone | ~1,500 ms | 1.0× |
| MIG 4g + NRx same-partition | **18,076 ms** | **12.0×** |
| Time-slicing + NRx (today) | 18,342 ms | 11.7× |
| MPS + NRx (today) | 1,822 ms | **1.2×** (surprising!) |
| MPS + HBM stress (today) | 22,645 ms | 15.0× |

## Next steps
- Wait for matrix_v2 completion (~150 files)
- Full cell × workload × mode analysis
- Hypothesis formalization (A: persistent pool, B: CUDA Graph re-frame, C: cudaMemPool + reuse, D: shared context mechanism)
