"""Qwen2.5-7B inference stress workload — CloudLab / airan:25-3-final container.

Runs continuous forward passes to sustain HBM bandwidth pressure while L1+NRx
executes on another MIG partition. Reports iters/sec periodically so the
orchestrator can detect stalls.

Usage: qwen7b_stress.py <duration_sec>
"""
import os
import sys
import time

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/mydata/hf_cache")

import torch
from transformers import AutoModelForCausalLM

duration = int(sys.argv[1]) if len(sys.argv) > 1 else 300
model_name = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B")

torch.cuda.set_device(0)
print(f"[Qwen] loading {model_name}...", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.float16,
    device_map="cuda:0",
)
model.eval()

dummy = torch.randint(0, 32000, (1, 512), device="cuda:0")
with torch.no_grad():
    for _ in range(3):
        model(dummy)
torch.cuda.synchronize()

free, total = torch.cuda.mem_get_info(0)
print(f"[Qwen] HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB "
      f"duration={duration}s", flush=True)

c = 0
last = time.time()
start = last
with torch.no_grad():
    while time.time() - start < duration:
        model(dummy)
        c += 1
        now = time.time()
        if now - last >= 10.0:
            print(f"[Qwen] progress: {c} iters, {c/(now-start):.2f} it/s", flush=True)
            last = now
elapsed = time.time() - start
print(f"[Qwen] done: {c} iters, {c/elapsed:.2f} it/s", flush=True)
