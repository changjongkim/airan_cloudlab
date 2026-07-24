"""HBM saturation stress — CUDA triad kernel (STREAM benchmark style).

CLI: python3 run_hbm_stress.py <label> <duration_s> [--gb 8]

The 20260708 "MPS + HBM stress" reference workload. Ground-truth catastrophic
MPS breakdown case. Use as control alongside realistic (LLM/VLM/ASR) workloads
to show the saturation gradient.

Design:
- Three fp32 arrays, size GB each: a, b, c
- Triad kernel: a[i] = b[i] + scalar * c[i]
- Continuous loop for duration_s
- Per iteration reads 2×GB + writes 1×GB = 3×GB HBM traffic
- On A100 4g slice (~890 GB/s), 8 GB triad = ~300+ GB/s sustained (~35% peak)
- With scaled-up size, easily saturates
"""
import argparse, time
import cupy as cp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--gb", type=float, default=8.0, help="array size in GB")
    args = ap.parse_args()

    n = int(args.gb * (1024**3) / 4)  # fp32 elements
    print(f"[{time.strftime('%H:%M:%S')}] allocating 3 × {args.gb} GB arrays (n={n:,})", flush=True)
    a = cp.zeros(n, dtype=cp.float32)
    b = cp.random.rand(n, dtype=cp.float32)
    c = cp.random.rand(n, dtype=cp.float32)
    cp.cuda.Stream.null.synchronize()
    print(f"[{time.strftime('%H:%M:%S')}] triad loop starting", flush=True)

    start = time.time()
    n_iters = 0
    scalar = cp.float32(3.14)
    while time.time() - start < args.duration_s:
        cp.add(b, scalar * c, out=a)
        cp.cuda.Stream.null.synchronize()
        n_iters += 1
        if n_iters % 100 == 0:
            elapsed = time.time() - start
            gb_moved = n_iters * args.gb * 3
            gbps = gb_moved / elapsed
            print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters}  elapsed={elapsed:.1f}s  HBM_BW={gbps:.1f} GB/s", flush=True)

    elapsed = time.time() - start
    gbps = n_iters * args.gb * 3 / elapsed
    print(f"[{time.strftime('%H:%M:%S')}] DONE  iters={n_iters}  HBM_BW={gbps:.1f} GB/s  label={args.label}", flush=True)

if __name__ == "__main__":
    main()
