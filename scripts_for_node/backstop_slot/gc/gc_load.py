"""A stand-in for AI on the GPU: PyTorch matrix products for SECONDS, in the primary context or in a green context.

usage: gc_load.py SPEC SECONDS      (SPEC: 0 primary, N a green context on the first N SMs)
"""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import greenctx as g
spec, seconds = sys.argv[1], float(sys.argv[2])
dev, primary, res = g.init(0)
got = int(res.u.sm.smCount)
if spec != "0":
    _g, ctx, got, other = g.green(dev, res, int(spec)); g.use(ctx)
import torch
a = torch.randn(4096, 4096, device="cuda", dtype=torch.float16)
for _ in range(10): b = a @ a
torch.cuda.synchronize()
print(f"LOAD READY on {got} SMs", flush=True)
start = time.perf_counter(); n = 0
while time.perf_counter() - start < seconds:
    for _ in range(20): b = a @ a
    torch.cuda.synchronize(); n += 20
print(f"RESULT load on {got} SMs ({spec}): {n / (time.perf_counter() - start):.0f} products/s", flush=True)
