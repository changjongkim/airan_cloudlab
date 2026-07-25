"""Embedding table random-access lookup (DLRM/recsys memory pattern).

CLI: python3 run_embed_lookup.py <label> <duration_s>

Design (real recommender system inference):
- Large embedding table: 32M rows × 128 dim × 2 bytes (fp16) = 8 GB
- Batch of 4096 random indices per lookup
- Simulates DLRM/Wide&Deep pattern
- Random access → cache-hostile → PURE memory-bound

Roofline:
- Read 4096 × 128 × 2 = 1 MB per lookup, essentially random from 8 GB table
- Compute: negligible (few adds)
- AI ≈ 0.001 FLOP/byte → EXTREMELY memory-bound (worse than STREAM)
- Kernels: many small gather + reduce (10+ per lookup batch)
- Expected 500 iter/s × 10 kernels = 5K kernels/s
"""
import argparse, time
import cupy as cp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--table_rows", type=int, default=32_000_000)  # 32M rows
    ap.add_argument("--dim", type=int, default=128)
    ap.add_argument("--batch", type=int, default=4096)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] allocating embedding table {args.table_rows/1e6:.0f}M×{args.dim} fp16 = {args.table_rows*args.dim*2/1e9:.1f}GB", flush=True)
    # cupy.random only supports fp32/fp64 → allocate empty + fill row-by-row via cp.arange
    table = cp.empty((args.table_rows, args.dim), dtype=cp.float16)
    # Initialize with cheap deterministic pattern (avoids fp32→fp16 32GB spike)
    x = cp.arange(args.dim, dtype=cp.float16)
    table[:] = x
    cp.cuda.Stream.null.synchronize()
    print(f"[{time.strftime('%H:%M:%S')}] table ready, entering random-access loop", flush=True)

    # Pool of pre-generated random index tensors (avoid CPU→GPU per iter)
    rngs = [cp.random.randint(0, args.table_rows, size=args.batch, dtype=cp.int32) for _ in range(16)]

    start = time.time()
    n_iters = 0
    total_bytes = 0
    idx = 0
    while time.time() - start < args.duration_s:
        # Gather 4096 random rows → sum → scalar (mimics DLRM interaction)
        rows = table[rngs[idx % len(rngs)]]           # random gather
        out  = rows.sum(axis=0)                       # reduce
        _    = out.sum()                              # final reduce
        cp.cuda.Stream.null.synchronize()
        n_iters += 1
        idx += 1
        total_bytes += args.batch * args.dim * 2
        if n_iters % 200 == 0:
            elapsed = time.time() - start
            bw = total_bytes / elapsed / 1e9
            print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters} elapsed={elapsed:.1f}s effective_BW={bw:.1f} GB/s (of 8GB table random access)", flush=True)

    elapsed = time.time() - start
    print(f"[{time.strftime('%H:%M:%S')}] DONE iters={n_iters} bytes_read={total_bytes/1e9:.1f}GB label={args.label}", flush=True)

if __name__ == "__main__":
    main()
