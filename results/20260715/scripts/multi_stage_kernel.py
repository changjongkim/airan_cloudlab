"""
multi_stage_kernel.py — Approach B: cuPHY-like multi-stage megakernel benchmark.

Improves on persistent_kernel.py by simulating cuPHY's actual multi-stage
structure with different memory access patterns per stage.

6 stages per iteration (mimicking PUSCH RX pipeline):
  Stage 1 (ChEst-like)  : matrix operations on H (channel matrix)
  Stage 2 (NI-like)     : reduction over residual
  Stage 3 (MMSE-like)   : outer product + inversion pattern
  Stage 4 (Convert)     : layout transformation (large memory moves)
  Stage 5 (LDPC-like)   : bit-level iteration on LLRs
  Stage 6 (CRC)         : final small reduction

Two modes:
  BASELINE: N iterations of (cudaMalloc-allocate all buffers, launch 6 kernels, cudaFree)
  MEGAKERNEL: 1 kernel with all 6 stages inline in a grid-stride loop, N iters inside

This is a much better proxy for real cuPHY L1 than the single-stage elementwise
transform in persistent_kernel.py.

Buffer sizes chosen to match PUSCH RX typical scale:
  H_est   : 3276 subc × 12 sym × 4 ant × complex64 = 630KB
  LLRs    : 3276 × 12 × 8 (mod_order max) × float32 = 1.2MB
  Total per-iter working set ~2MB × 4 (double-buffered) = 8MB

Usage:
  python3 multi_stage_kernel.py <label> <mode> <iterations>
    mode = 'baseline' | 'megakernel'
"""
import os, sys, json, time, datetime
import numpy as np
import cupy as cp

LABEL = sys.argv[1] if len(sys.argv) > 1 else "multistage_test"
MODE = sys.argv[2] if len(sys.argv) > 2 else "baseline"
ITERATIONS = int(sys.argv[3]) if len(sys.argv) > 3 else 100
WARMUP = 10

RESULTS_DIR = os.environ.get("RESULTS_DIR", "./results")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Sizes chosen to match PUSCH RX typical (273 PRB × 12 subc = 3276)
SUBC = 3276
SYMS = 12
ANT = 4
MOD = 8
H_SIZE = SUBC * SYMS * ANT * 2  # complex64 = 2 floats
LLR_SIZE = SUBC * SYMS * MOD    # float32

BLOCKS = 108
THREADS = 256

# ------------------------------------------------------------------------------
# Kernels
# ------------------------------------------------------------------------------
src = r"""
// Stage 1 — Channel estimation-like: complex outer product per subcarrier
extern "C" __global__ void chest_stage(float* H_out, float* rx, float* dmrs, int subc, int sym, int ant) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int total = subc * sym * ant * 2;  // complex = 2 floats
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < total; i += stride) {
        // Simple correlate: H[i] = rx[i] * conj(dmrs[i])
        float r = rx[i];
        float d = dmrs[i];
        H_out[i] = r * d + 0.001f;
    }
}

// Stage 2 — Noise/Interference estimation: reduction over per-subcarrier residuals
extern "C" __global__ void ni_stage(float* noise_var, float* H, float* rx, int subc, int sym, int ant) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = subc * ant * 2;
    for (int i = tid; i < total; i += stride) {
        float acc = 0.f;
        for (int s = 0; s < sym; s++) {
            float diff = H[i * sym + s] - rx[i * sym + s];
            acc += diff * diff;
        }
        noise_var[i] = acc / sym;
    }
}

// Stage 3 — MMSE-like: element-wise combine H, noise, and rx
extern "C" __global__ void mmse_stage(float* llr_out, float* H, float* noise_var, float* rx, int subc, int sym, int mod) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int total = subc * sym * mod;
    for (int i = tid; i < total; i += stride) {
        int subc_i = (i / mod) % subc;
        int nv_idx = subc_i * 2;  // simplified — one noise sample per subc pair
        float h = H[subc_i];
        float nv = noise_var[nv_idx % (subc*2)];
        float r = rx[i % (subc*sym*2)];
        // MMSE: (H^H * r) / (H^H*H + noise_var)
        llr_out[i] = (h * r) / (h * h + nv + 1e-6f);
    }
}

// Stage 4 — Convert / layout transformation (memcpy-like heavy work)
extern "C" __global__ void convert_stage(float* out_buf, float* in_buf, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < size; i += stride) {
        // Simulate layout change: permute + scale
        int j = ((i * 7) ^ 0xa5) % size;
        out_buf[i] = in_buf[j] * 1.001f;
    }
}

// Stage 5 — LDPC-like: bit-wise iteration on LLRs
extern "C" __global__ void ldpc_stage(float* llr_inout, int size, int iters) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    for (int i = tid; i < size; i += stride) {
        float v = llr_inout[i];
        for (int k = 0; k < iters; k++) {
            v = tanhf(v * 0.5f) * 0.9f;  // dummy iterative refinement
        }
        llr_inout[i] = v;
    }
}

// Stage 6 — CRC-like reduction
extern "C" __global__ void crc_stage(int* crc_out, float* llr, int size) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int acc = 0;
    for (int i = tid; i < size; i += stride) {
        acc ^= __float_as_int(llr[i]);
    }
    if (tid == 0) atomicAdd(crc_out, acc);
}

// -----------------------------------------------------------------------
// MEGAKERNEL — all 6 stages inline, N iterations in a single grid-stride
// -----------------------------------------------------------------------
extern "C" __global__ void megaworker(
    float* H, float* rx, float* dmrs, float* noise_var, float* llrs,
    float* convert_buf, int* crc_out,
    int subc, int sym, int ant, int mod, int iterations, int ldpc_iters
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;
    int H_size = subc * sym * ant * 2;
    int NV_size = subc * ant * 2;
    int LLR_size = subc * sym * mod;

    for (int iter = 0; iter < iterations; iter++) {
        // Stage 1: Channel estimation-like
        for (int i = tid; i < H_size; i += stride) {
            H[i] = rx[i] * dmrs[i] + 0.001f;
        }
        __syncthreads();

        // Stage 2: NI estimation-like (accumulation)
        for (int i = tid; i < NV_size; i += stride) {
            float acc = 0.f;
            for (int s = 0; s < sym; s++) {
                float diff = H[i * sym + s] - rx[i * sym + s];
                acc += diff * diff;
            }
            noise_var[i] = acc / sym;
        }
        __syncthreads();

        // Stage 3: MMSE
        for (int i = tid; i < LLR_size; i += stride) {
            int subc_i = (i / mod) % subc;
            int nv_idx = (subc_i * 2) % NV_size;
            float h = H[subc_i];
            float nv = noise_var[nv_idx];
            float r = rx[i % H_size];
            llrs[i] = (h * r) / (h * h + nv + 1e-6f);
        }
        __syncthreads();

        // Stage 4: Convert
        for (int i = tid; i < LLR_size; i += stride) {
            int j = ((i * 7) ^ 0xa5) % LLR_size;
            convert_buf[i] = llrs[j] * 1.001f;
        }
        __syncthreads();

        // Stage 5: LDPC
        for (int i = tid; i < LLR_size; i += stride) {
            float v = convert_buf[i];
            for (int k = 0; k < ldpc_iters; k++) {
                v = tanhf(v * 0.5f) * 0.9f;
            }
            llrs[i] = v;
        }
        __syncthreads();

        // Stage 6: CRC
        int acc = 0;
        for (int i = tid; i < LLR_size; i += stride) {
            acc ^= __float_as_int(llrs[i]);
        }
        if (tid == 0) atomicAdd(crc_out, acc);
        __syncthreads();
    }
}
"""

module = cp.RawModule(code=src)
chest_kern = module.get_function("chest_stage")
ni_kern    = module.get_function("ni_stage")
mmse_kern  = module.get_function("mmse_stage")
convert_kern = module.get_function("convert_stage")
ldpc_kern  = module.get_function("ldpc_stage")
crc_kern   = module.get_function("crc_stage")
mega_kern  = module.get_function("megaworker")

print(f"[multi] mode={MODE} iters={ITERATIONS} H={H_SIZE*4/1024:.0f}KB LLR={LLR_SIZE*4/1024:.0f}KB", flush=True)

def alloc_buffers():
    H  = cp.zeros(H_SIZE, dtype=cp.float32)
    rx = cp.ones(H_SIZE, dtype=cp.float32) * 0.5
    dmrs = cp.ones(H_SIZE, dtype=cp.float32) * 0.3
    NV_SIZE = SUBC * ANT * 2
    noise_var = cp.zeros(NV_SIZE, dtype=cp.float32)
    llrs = cp.zeros(LLR_SIZE, dtype=cp.float32)
    conv = cp.zeros(LLR_SIZE, dtype=cp.float32)
    crc  = cp.zeros(1, dtype=cp.int32)
    return H, rx, dmrs, noise_var, llrs, conv, crc

def run_baseline():
    print("[multi:baseline] warmup...", flush=True)
    pool = cp.get_default_memory_pool()
    for _ in range(WARMUP):
        H, rx, dmrs, noise_var, llrs, conv, crc = alloc_buffers()
        chest_kern((BLOCKS,), (THREADS,), (H, rx, dmrs, SUBC, SYMS, ANT))
        ni_kern((BLOCKS,), (THREADS,), (noise_var, H, rx, SUBC, SYMS, ANT))
        mmse_kern((BLOCKS,), (THREADS,), (llrs, H, noise_var, rx, SUBC, SYMS, MOD))
        convert_kern((BLOCKS,), (THREADS,), (conv, llrs, LLR_SIZE))
        ldpc_kern((BLOCKS,), (THREADS,), (conv, LLR_SIZE, 3))
        crc_kern((BLOCKS,), (THREADS,), (crc, conv, LLR_SIZE))
        cp.cuda.runtime.deviceSynchronize()
        del H, rx, dmrs, noise_var, llrs, conv, crc
        pool.free_all_blocks()

    print("[multi:baseline] measuring...", flush=True)
    lat = []
    for i in range(ITERATIONS):
        t0 = time.perf_counter()
        H, rx, dmrs, noise_var, llrs, conv, crc = alloc_buffers()
        chest_kern((BLOCKS,), (THREADS,), (H, rx, dmrs, SUBC, SYMS, ANT))
        ni_kern((BLOCKS,), (THREADS,), (noise_var, H, rx, SUBC, SYMS, ANT))
        mmse_kern((BLOCKS,), (THREADS,), (llrs, H, noise_var, rx, SUBC, SYMS, MOD))
        convert_kern((BLOCKS,), (THREADS,), (conv, llrs, LLR_SIZE))
        ldpc_kern((BLOCKS,), (THREADS,), (conv, LLR_SIZE, 3))
        crc_kern((BLOCKS,), (THREADS,), (crc, conv, LLR_SIZE))
        cp.cuda.runtime.deviceSynchronize()
        del H, rx, dmrs, noise_var, llrs, conv, crc
        pool.free_all_blocks()
        t1 = time.perf_counter()
        lat.append((t1 - t0) * 1000)
    return lat

def run_megakernel():
    print("[multi:megakernel] setup...", flush=True)
    H, rx, dmrs, noise_var, llrs, conv, crc = alloc_buffers()

    print("[multi:megakernel] warmup...", flush=True)
    mega_kern((BLOCKS,), (THREADS,),
              (H, rx, dmrs, noise_var, llrs, conv, crc,
               SUBC, SYMS, ANT, MOD, WARMUP, 3))
    cp.cuda.runtime.deviceSynchronize()

    print("[multi:megakernel] measuring...", flush=True)
    t0 = time.perf_counter()
    mega_kern((BLOCKS,), (THREADS,),
              (H, rx, dmrs, noise_var, llrs, conv, crc,
               SUBC, SYMS, ANT, MOD, ITERATIONS, 3))
    cp.cuda.runtime.deviceSynchronize()
    t1 = time.perf_counter()
    total_ms = (t1 - t0) * 1000
    return [total_ms / ITERATIONS] * ITERATIONS

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
    "H_size": H_SIZE, "LLR_size": LLR_SIZE,
    "subc": SUBC, "syms": SYMS, "ant": ANT, "mod": MOD,
    "blocks": BLOCKS, "threads": THREADS,
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
out_file = os.path.join(RESULTS_DIR, f"multistage_{MODE}_{LABEL}_{ts}.json")
with open(out_file, "w") as f:
    json.dump(result, f, indent=2)
print(f"[multi:{MODE}] {LABEL}: mean={arr.mean():.3f}ms total_wall={arr.sum():.0f}ms", flush=True)
