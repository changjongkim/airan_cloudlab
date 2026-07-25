"""
persistent_kernel.py — Approach 3 (proper implementation)

Compares two structural patterns processing the SAME synthetic memory-bound
workload (elementwise buffer transform, mimics per-frame cuPHY L1 pattern):

  BASELINE: N iterations of (cudaMalloc + kernel_launch + cudaFree)
            → many CUDA API calls per second, hits cudaFree sync path in coloc

  MEGAKERNEL: ONE kernel launch that does all N iterations internally
              → zero cudaMalloc/cudaFree/cudaLaunchKernel in hot path
              → sync path cannot be triggered because there are no per-iter
                API calls to synchronize on

nsys will capture:
  - baseline: cudaMalloc(count=N), cudaFree(count=N), cudaLaunchKernel(count=N)
    In coloc: cudaFree waits become the sync path (as in Chain 9 baseline).
  - megakernel: cudaLaunchKernel(count=1), 0 malloc/free in hot path
    In coloc: nothing to synchronize on.

Design choice: report per-iteration latency = total_time / N for megakernel
(rather than polling from host, which introduces its own coherence bugs).
This gives directly comparable per-iter numbers.

Usage:
  python3 persistent_kernel.py <label> <mode> <iterations>
    mode = 'baseline' | 'megakernel'
"""
import os, sys, json, time, datetime
import numpy as np
import cupy as cp

LABEL = sys.argv[1] if len(sys.argv) > 1 else "persist_test"
MODE = sys.argv[2] if len(sys.argv) > 2 else "baseline"
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
WARMUP = 10

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

BUFFER_SIZE = int(os.environ.get("BUFFER_SIZE", "4194304"))  # 4M floats = 16MB
BLOCKS = int(os.environ.get("BLOCKS", "108"))
THREADS = int(os.environ.get("THREADS", "256"))

# ------------------------------------------------------------------------------
# Kernels
# ------------------------------------------------------------------------------
kernel_src = r"""
// Per-iteration worker: one iteration of buffer transform (baseline)
extern "C" __global__ void worker(float* buf, int n) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int i = tid; i < n; i += stride) {
        buf[i] = buf[i] * 1.0001f + 0.001f;
    }
}

// Megakernel: runs N iterations of same work in a single launch
// No API-level synchronization needed between iterations
extern "C" __global__ void megaworker(float* buf, int n, int iterations) {
    int stride = gridDim.x * blockDim.x;
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    for (int iter = 0; iter < iterations; iter++) {
        for (int i = tid; i < n; i += stride) {
            buf[i] = buf[i] * 1.0001f + 0.001f;
        }
        // __syncthreads() would sync within block only; not needed for elementwise
    }
}
"""

module = cp.RawModule(code=kernel_src)
worker_kernel = module.get_function("worker")
mega_kernel = module.get_function("megaworker")

print(f"[persist] mode={MODE} iters={ITERATIONS} buf={BUFFER_SIZE} blocks={BLOCKS} threads={THREADS}", flush=True)

# ------------------------------------------------------------------------------
# BASELINE: cudaMalloc + kernel launch + cudaFree per iteration
# ------------------------------------------------------------------------------
def run_baseline():
    print("[persist:baseline] warmup...", flush=True)
    pool = cp.get_default_memory_pool()
    for _ in range(WARMUP):
        buf = cp.zeros(BUFFER_SIZE, dtype=cp.float32)
        worker_kernel((BLOCKS,), (THREADS,), (buf, BUFFER_SIZE))
        cp.cuda.runtime.deviceSynchronize()
        del buf
        pool.free_all_blocks()  # force real cudaFree

    print("[persist:baseline] measuring...", flush=True)
    lat = []
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        buf = cp.zeros(BUFFER_SIZE, dtype=cp.float32)  # cudaMalloc
        worker_kernel((BLOCKS,), (THREADS,), (buf, BUFFER_SIZE))  # cudaLaunchKernel
        cp.cuda.runtime.deviceSynchronize()  # ensure completion
        del buf
        pool.free_all_blocks()  # cudaFree
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000)
    return lat

# ------------------------------------------------------------------------------
# MEGAKERNEL: allocate ONCE, run N iterations in a single kernel launch
# ------------------------------------------------------------------------------
def run_megakernel():
    print("[persist:megakernel] setup...", flush=True)
    # Allocate ONCE outside hot path (this is the whole point of persistent pool)
    buf = cp.zeros(BUFFER_SIZE, dtype=cp.float32)

    print("[persist:megakernel] warmup...", flush=True)
    # Warmup: single small mega-kernel launch
    mega_kernel((BLOCKS,), (THREADS,), (buf, BUFFER_SIZE, WARMUP))
    cp.cuda.runtime.deviceSynchronize()

    print("[persist:megakernel] measuring...", flush=True)
    # Single mega-kernel launch that does ALL N iterations internally
    t0 = time.perf_counter()
    mega_kernel((BLOCKS,), (THREADS,), (buf, BUFFER_SIZE, ITERATIONS))
    cp.cuda.runtime.deviceSynchronize()  # only one sync at end
    t1 = time.perf_counter()

    total_ms = (t1 - t0) * 1000
    per_iter = total_ms / ITERATIONS
    # Report per-iter (equal for all iters — kernel-side coordination avoided)
    return [per_iter] * ITERATIONS


if MODE == "baseline":
    latencies = run_baseline()
elif MODE == "megakernel":
    latencies = run_megakernel()
else:
    print(f"unknown mode: {MODE}"); sys.exit(1)

arr = np.array(latencies)
result = {
    "label": LABEL,
    "mode": MODE,
    "iterations": ITERATIONS,
    "buffer_size": BUFFER_SIZE,
    "blocks": BLOCKS,
    "threads": THREADS,
    "mean_ms": float(arr.mean()),
    "p50_ms": float(np.percentile(arr, 50)),
    "p95_ms": float(np.percentile(arr, 95)),
    "p99_ms": float(np.percentile(arr, 99)),
    "min_ms": float(arr.min()),
    "max_ms": float(arr.max()),
    "total_wall_ms": float(arr.sum()),
    "raw_ms": [float(x) for x in latencies],
}

ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = os.path.join(RESULTS_DIR, f"persist_{MODE}_{LABEL}_{ts}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)

print(f"[persist:{MODE}] {LABEL}: mean={arr.mean():.3f}ms p95={np.percentile(arr,95):.3f}ms "
      f"total_wall={arr.sum():.0f}ms", flush=True)
print(f"[persist:{MODE}] saved: {out_file}", flush=True)
