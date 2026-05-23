"""
Qwen2.5-1.8B small LLM stress — fits in 2g.10gb MIG instance.

For multi-partition co-location experiments where the full Qwen-7B (14GB)
exceeds the partition cap. Same prefill/decode dynamics, smaller model
weight (~3.6GB fp16).

Usage: python3 run_qwen_small_stress.py <gpu_id> <duration_sec>

Env:
  MODEL_NAME        — HF model id (default Qwen/Qwen2.5-1.5B)
  CONTEXT_LEN       — prefill context length (default 512)
  PHASE             — "mixed" (default), "prefill", "decode"
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME_LOCAL", "/mydata/hf_cache"))

import torch
from transformers import AutoModelForCausalLM

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
model_name = os.environ.get("MODEL_NAME", "Qwen/Qwen2.5-1.5B")
ctx_len = int(os.environ.get("CONTEXT_LEN", "512"))
phase = os.environ.get("PHASE", "mixed")

torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}"

print(f"[Qwen-small] model={model_name}, ctx={ctx_len}, phase={phase}, dur={duration}s",
      flush=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name, torch_dtype=torch.float16, device_map=device,
)
model.eval()

batch = torch.randint(0, 32000, (1, ctx_len), device=device)
with torch.no_grad():
    out = model(batch, use_cache=True)
past_kv = out.past_key_values
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
print(f"[Qwen-small] HBM={(total-free)/1e9:.2f}/{total/1e9:.1f}GB", flush=True)

c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        if phase == "prefill":
            batch = torch.randint(0, 32000, (1, ctx_len), device=device)
            model(batch, use_cache=False)
        elif phase == "decode":
            next_tok = torch.randint(0, 32000, (1, 1), device=device)
            out = model(next_tok, past_key_values=past_kv, use_cache=True)
            past_kv = out.past_key_values
            if c % 200 == 199:
                # reset to keep KV bounded
                out = model(batch, use_cache=True)
                past_kv = out.past_key_values
        else:  # mixed
            if c % 50 == 0:
                # periodic prefill (every ~50 iters)
                out = model(batch, use_cache=True)
                past_kv = out.past_key_values
            else:
                next_tok = torch.randint(0, 32000, (1, 1), device=device)
                out = model(next_tok, past_key_values=past_kv, use_cache=True)
                past_kv = out.past_key_values
        c += 1
torch.cuda.synchronize(gpu_id)
elapsed = time.time() - start
print(f"[Qwen-small] done: {c} iters, {c/elapsed:.1f} it/s", flush=True)
