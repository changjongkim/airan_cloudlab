# cuPHY cudaFree Mitigation Shims

LD_PRELOAD shims for testing cudaFree mitigation strategies in NVIDIA cuPHY-based AI-RAN pipelines.

## Background

Empirical finding (20260622 + 20260701 experiments): NVIDIA cuPHY L1 calls `cudaFree()` ~130 times per 5G cell, and each call becomes an implicit synchronization wait when co-located with concurrent AI kernels in the same MIG partition. Result: cudaFree contention scales as `(cells × 130 × per-call-us × 10x coloc-penalty)`.

These shims explore three families of mitigation:

## Shims

### Existing (diagnostic — do NOT fix the problem)
- **e9_sync_shim.c** — `cudaDeviceSynchronize()` before every `cudaFree()`. Makes the sync explicit for measurement.
- **e2_defer_shim.c** — replaces `cudaFree()` with no-op. Cannot free (memory leak) — diagnostic only.
- **proof_56_shim.c** — deferred queue + `atexit()` drain. Same effect as e2 but with actual free at process exit.

### New (Option A + B — actual mitigation attempts)
- **cudaFreeAsync_shim.c** *(Option A)* — replaces `cudaFree()` with `cudaFreeAsync()` on a shim-owned stream. Simple but may fall back to sync if pointer wasn't from an async pool.
- **cudaMemPool_shim.c** *(Option B)* — replaces both `cudaMalloc()` and `cudaFree()` with stream-ordered pool variants. Complete stream-ordered memory management.

## Build

```bash
gcc -shared -fPIC -O2 cudaFreeAsync_shim.c -o cudaFreeAsync.so -ldl -lpthread
gcc -shared -fPIC -O2 cudaMemPool_shim.c   -o cudaMemPool.so   -ldl -lpthread
```

## Use

```bash
# Option A
LD_PRELOAD=/path/to/cudaFreeAsync.so \
  CUFREE_ASYNC_LOG=1 \
  python3 real_l1.py ...

# Option B
LD_PRELOAD=/path/to/cudaMemPool.so \
  CUPOOL_LOG=1 \
  CUPOOL_RELEASE_THRESHOLD_MB=2048 \
  python3 real_l1.py ...
```

## Expected outcomes

### Scenario A: Async fully fixes coloc penalty
```
NRx coloc without shim: 18334 ms cudaFree (cells=40)
NRx coloc with shim:    ~1500 ms cudaFree
→ ~10× penalty → 1× (contention eliminated)
```

### Scenario B: Sync wait shifts to another API (Finding 4 — conservation)
```
without shim: cudaFree 18334 ms
with shim:    cudaFree 0 ms, cudaMemcpyAsync +18000 ms
→ Async API insufficient; GPU work queue depth is the true bottleneck
```

### Scenario C: Partial improvement
Some conditions improve, others don't. Reveals which allocations are actually sync-critical.

## Chain 8 conditions (planned)

Per cell size ∈ {4, 10, 40, 60} on 3g+2g layout:
- `A_alone` — baseline (no shim, no AI)
- `A_nrx_coloc_noshim` — NRx coloc, no shim (matches Chain 6/7 baseline)
- `A_nrx_coloc_freeasync` — NRx coloc + Option A shim
- `A_nrx_coloc_pool` — NRx coloc + Option B shim

= 4 conditions × 4 cells = 16 conditions total. Expected runtime ~30-45 min.

## Structure

```
cuPHY_mitigation_shims/
├── README.md              (this file)
├── shims/
│   ├── e9_sync_shim.c              — diagnostic (add explicit sync)
│   ├── e2_defer_shim.c             — diagnostic (skip cudaFree)
│   ├── proof_56_shim.c             — diagnostic (defer + drain)
│   ├── cudaFreeAsync_shim.c        — Option A (simple async free)
│   └── cudaMemPool_shim.c          — Option B (full pool)
├── scripts/
│   └── run_chain8.sh               — chain 8 driver
├── docs/
│   └── (diagrams, references)
└── results/
    └── (chain 8 output, rsync'd from remote)
```

## Related datasets

- `../cloudlab_results/results/20260622/` — Chain 1, 2, 4 + PROOF 5/6 (mechanism identification)
- `../cloudlab_results/results/20260701/` — Chain 5, 6, 7 (partition sweep + cell scaling)
