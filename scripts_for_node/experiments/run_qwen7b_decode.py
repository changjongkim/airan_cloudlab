"""
Qwen-7B DECODE-ONLY stress — HBM-idle phase isolated.

Pre-fills KV cache ONCE with full context, then loops doing 1-token incremental
decode using past_key_values. Each step is a tiny GEMM (1 token × hidden_dim)
that mostly hits L2 cache; HBM is largely idle except for weight + KV read.

Used to test phase-alignment hypothesis in MIG bimodal-leakage analysis
(SENSITIVITY_EXPERIMENTS §A1). If H1 (phase alignment) is correct, L1
co-located with decode-only AI should stay in LOW mode (no leakage).

Usage: python3 run_qwen7b_decode.py <gpu_id> <duration_sec>
"""
import os
import sys
import time

os.environ.setdefault("HF_HOME", os.environ.get("HF_HOME_LOCAL", "/mydata/hf_cache"))

import torch
from transformers import AutoModelForCausalLM

gpu_id = int(sys.argv[1]) if len(sys.argv) > 1 else 0
duration = int(sys.argv[2]) if len(sys.argv) > 2 else 120
init_ctx_len = int(os.environ.get("INIT_CONTEXT_LEN", "512"))

torch.cuda.set_device(gpu_id)
device = f"cuda:{gpu_id}"

print(f"[Qwen-7B DECODE] Loading model, init_ctx={init_ctx_len}, duration={duration}s",
      flush=True)
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B",
    torch_dtype=torch.float16,
    device_map=device,
)
model.eval()

# One-time prefill to populate KV cache
print("[Qwen-7B DECODE] Prefilling KV cache (once)...", flush=True)
ctx_tokens = torch.randint(0, 32000, (1, init_ctx_len), device=device)
with torch.no_grad():
    out = model(ctx_tokens, use_cache=True)
past_kv = out.past_key_values

# Warmup the decode path
next_tok = torch.randint(0, 32000, (1, 1), device=device)
with torch.no_grad():
    for _ in range(3):
        out = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
torch.cuda.synchronize(gpu_id)

free, total = torch.cuda.mem_get_info(gpu_id)
print(f"[Qwen-7B DECODE] HBM={(total-free)/1e9:.1f}/{total/1e9:.1f}GB, running", flush=True)

# Decode loop: 1 token per iter, never re-prefill. KV cache grows.
# To avoid unbounded growth eating HBM, periodically reset KV cache.
KV_RESET_EVERY = int(os.environ.get("KV_RESET_EVERY", "200"))
c = 0
start = time.time()
with torch.no_grad():
    while time.time() - start < duration:
        next_tok = torch.randint(0, 32000, (1, 1), device=device)
        out = model(next_tok, past_key_values=past_kv, use_cache=True)
        past_kv = out.past_key_values
        c += 1
        if c % KV_RESET_EVERY == 0:
            # Reset KV: re-prefill (brief burst) to keep memory bounded
            out = model(ctx_tokens, use_cache=True)
            past_kv = out.past_key_values
torch.cuda.synchronize(gpu_id)
elapsed = time.time() - start
print(f"[Qwen-7B DECODE] done: {c} iters, {c/elapsed:.1f} it/s, "
      f"avg {elapsed*1000/c:.2f} ms/iter", flush=True)
