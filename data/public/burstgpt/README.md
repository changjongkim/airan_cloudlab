# BurstGPT input used for SoftWall workload preparation

Source: <https://github.com/HPMLL/BurstGPT>  
Paper: Yuxin Wang et al., “BurstGPT: A Real-World Workload Dataset to Optimize
LLM Serving Systems,” KDD 2025.  
License: CC-BY-4.0; the upstream `LICENSE` and `UPSTREAM_README.md` are preserved
in this directory.

## Preserved files

| File | SHA-256 | Meaning |
|---|---|---|
| `BurstGPT_1.csv` | `46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a` | Upstream 1,429,737-row trace downloaded from the repository's `main` branch on 2026-09-23 |
| `LICENSE` | `9e5f1b3c610b9c2da5c313bf81d577a7d1acec686bdb0384edefa6df0f90cd94` | Upstream CC-BY-4.0 license |
| `UPSTREAM_README.md` | `9d9061ef8548e96a854df7c317b4cdf286745c1a6da2f711f288f5e24aa09ef3` | Upstream documentation at download time |
| `softwall_burst60_prefill_trace.json` | `da760e696d78ec4d71c3d595770b74fa1fab54eb18ba0d9ab50d56388f7bb18d` | Deterministically derived SoftWall request trace |

## Derived trace rule

`build_burstgpt_prefill_trace.py` keeps requests with positive request and
response token counts, selects the earliest densest fixed 60-second window,
and maps input lengths upward to `{16,32,64,128,256,512}`. Inputs above 512
are capped and explicitly counted. The current result contains 1,136 requests,
288,591 offered input-token value, and 56 capped requests. Arrival timestamps
come from BurstGPT. The fixed 1,000 ms SLO is a synthetic experiment parameter
because BurstGPT does not publish a request deadline.

The derived trace is workload preparation, not a completed SoftWall baseline
experiment. Freeze a separate experiment protocol before changing time scale,
SLO, token mapping, or the selected source window.
