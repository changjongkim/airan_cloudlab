"""
Qwen-7B PREFILL-ONLY stress — HBM-burst phase isolated.

Every iteration runs a FULL forward pass on a fresh 512-token batch.
This is equivalent to the prefill phase of LLM serving: large GEMM, heavy
HBM weight reads, no KV cache reuse. Used to test phase-alignment hypothesis
in MIG bimodal-leakage analysis (SENSITIVITY_EXPERIMENTS §A1).

Usage: python3 run_qwen7b_prefill.py <gpu_id> <duration_sec>
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME_LOCAL", "/mydata/hf_cache"))

import torch
from transformers import AutoModelForCausalLM

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
ctx_len = int(os.environ.get("CONTEXT_LEN", "512"))

torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}"

print(f"[Qwen-7B PREFILL] Loading model, ctx={ctx_len}, duration={duration}s", flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map=device,
)
model.eval()

# Random token batch
batch = torch.randint(0, 32000, (1, ctx_len), device=device)

# Warmup
with torch.no_grad():
    for _ in range(3):
        model(batch)
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
print(f"[Qwen-7B PREFILL] HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB, running", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        # Fresh batch every iter → no KV cache reuse → pure prefill burst
        batch = torch.randint(0, 32000, (1, ctx_len), device=device)
        model(batch, use_cache=False)
        c += 1
torch.cuda.synchronize(gpu_id)
elapsed = time.time() - start
print(f"[Qwen-7B PREFILL] done: {c} iters, {c/elapsed:.2f} it/s, "
      f"avg {elapsed*1000/c:.1f} ms/iter", flush=True)
