"""
Massive GEMM stress — generic tensor-core / SM compute contention generator.

Usage:  python3 run_gemm_massive.py <gpu_id> <duration_sec>
Env:    DIM   (square matrix dimension, default 4096)

Mirrors the F_D / F_G "GEMM" conditions: sustained large fp16 matmuls on the
tensor cores to saturate SM/compute while the L1 pipeline runs concurrently.
"""
import os
import sys
import time

import torch

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
dim = int(os.environ.get("DIM", "4096"))

dev = f"cuda:{gpu_id}"
torch.cuda.set_device(gpu_id)
print(f"[gemm] dim={dim} dtype=fp16 dur={duration}s", flush=True)

a = torch.randn(dim, dim, dtype=torch.float16, device=dev)
b = torch.randn(dim, dim, dtype=torch.float16, device=dev)
c = torch.empty(dim, dim, dtype=torch.float16, device=dev)

t_end = time.time() + duration
iters = 0
while time.time() < t_end:
    for _ in range(50):
        torch.matmul(a, b, out=c)
    iters += 50
    torch.cuda.synchronize()
print(f"[gemm] done, {iters} matmuls", flush=True)
