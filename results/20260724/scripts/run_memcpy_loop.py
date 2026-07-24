"""Small memcpy loop — high launch rate + moderate HBM traffic.

CLI: python3 run_memcpy_loop.py <label> <duration_s>

Design (small-kernel memory pattern):
- 128KB source buffer, 128KB destination buffer
- cudaMemcpyAsync in tight loop
- Each launch is TINY but launches per second is EXTREME
- Simulates a control-plane message copy loop or small tensor shuffling

Roofline:
- Per copy: 128KB read + 128KB write = 256KB HBM traffic
- Compute: ZERO
- AI = 0 → PURE memory-bound
- Expected: 50K memcpy/s → 12.5 GB/s HBM (low BW), 50K launches/s (HIGH launch rate)

Purpose: isolate "high launch rate + memory access" from "few large kernels":
- Should show sync explosion when MPS off
- Should saturate MPS server queue (potential residual sync)
"""
import argparse, time
import cupy as cp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--copy_kb", type=int, default=128)
    args = ap.parse_args()

    n = args.copy_kb * 1024 // 4  # fp32 elements
    print(f"[{time.strftime('%H:%M:%S')}] allocating 2× {args.copy_kb} KB buffers", flush=True)
    src = cp.random.rand(n, dtype=cp.float32)
    dst = cp.empty_like(src)
    cp.cuda.Stream.null.synchronize()
    print(f"[{time.strftime('%H:%M:%S')}] entering high-launch-rate memcpy loop", flush=True)

    start = time.time()
    n_iters = 0
    while time.time() - start < args.duration_s:
        # Tight loop of 100 memcpys per sync — high launch rate
        for _ in range(100):
            cp.copyto(dst, src)
        cp.cuda.Stream.null.synchronize()
        n_iters += 100
        if n_iters % 10000 == 0:
            elapsed = time.time() - start
            rate = n_iters / elapsed
            bytes_moved = n_iters * args.copy_kb * 1024 * 2  # read+write
            bw = bytes_moved / elapsed / 1e9
            print(f"[{time.strftime('%H:%M:%S')}] copies={n_iters} rate={rate:.0f}/s BW={bw:.1f} GB/s elapsed={elapsed:.1f}s", flush=True)

    elapsed = time.time() - start
    rate = n_iters / elapsed
    print(f"[{time.strftime('%H:%M:%S')}] DONE copies={n_iters} rate={rate:.0f}/s label={args.label}", flush=True)

if __name__ == "__main__":
    main()
