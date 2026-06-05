"""
Massive memcpy stress — generic memory-bandwidth contention generator.

Usage:  python3 run_memcpy_massive.py <gpu_id> <duration_sec>
Env:    SIZE_MB   (per-buffer size, default 256)
        N_STREAMS (concurrent CUDA streams, default 4)
        KIND      (D2D | H2D | D2H, default D2D)

Mirrors the F_B / F_C / F_G "memcpy massive" conditions from the CloudLab runs:
sustained large copies on multiple streams to saturate HBM / PCIe bandwidth
while the L1 pipeline runs concurrently (no-MIG time-slice contention).
"""
import os
import sys
import time

import torch

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
size_mb = int(os.environ.get("SIZE_MB", "256"))
n_streams = int(os.environ.get("N_STREAMS", "4"))
kind = os.environ.get("KIND", "D2D").upper()

dev = f"cuda:{gpu_id}"
torch.cuda.set_device(gpu_id)
n_elems = size_mb * 1024 * 1024 // 4  # float32

print(f"[memcpy] kind={kind} size={size_mb}MB streams={n_streams} dur={duration}s", flush=True)

streams = [torch.cuda.Stream(device=gpu_id) for _ in range(n_streams)]

if kind == "D2D":
    src = [torch.empty(n_elems, dtype=torch.float32, device=dev) for _ in range(n_streams)]
    dst = [torch.empty(n_elems, dtype=torch.float32, device=dev) for _ in range(n_streams)]
elif kind == "H2D":
    src = [torch.empty(n_elems, dtype=torch.float32, pin_memory=True) for _ in range(n_streams)]
    dst = [torch.empty(n_elems, dtype=torch.float32, device=dev) for _ in range(n_streams)]
elif kind == "D2H":
    src = [torch.empty(n_elems, dtype=torch.float32, device=dev) for _ in range(n_streams)]
    dst = [torch.empty(n_elems, dtype=torch.float32, pin_memory=True) for _ in range(n_streams)]
else:
    print(f"[memcpy] unknown KIND={kind}", flush=True)
    sys.exit(1)

t_end = time.time() + duration
copies = 0
while time.time() < t_end:
    for i, s in enumerate(streams):
        with torch.cuda.stream(s):
            dst[i].copy_(src[i], non_blocking=True)
    copies += n_streams
    if copies % (n_streams * 200) == 0:
        torch.cuda.synchronize()
torch.cuda.synchronize()
print(f"[memcpy] done, {copies} copies", flush=True)
