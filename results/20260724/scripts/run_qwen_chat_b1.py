"""Qwen-3B batch=1 latency-critical chat decode — TRULY memory-bound.

CLI: python3 run_qwen_chat_b1.py <label> <duration_s>

Design (real single-user edge chatbot):
- vLLM with max_num_seqs=1 (single request at a time)
- enforce_eager=True to get MANY kernel launches (no CUDA graph fusion)
- Real prompts from ultrachat_200k
- max_tokens=256

Roofline:
- Per token: 3.09B params × 2 bytes = 6.18 GB HBM weight read
- Compute: 6.18 GFLOPs
- Arithmetic Intensity = 1 FLOP/byte ≪ 200 ridge → GUARANTEED memory-bound
- Kernels/token: ~200 (36 layers × ~6 kernels)
- Expected: ~100 tok/s → ~20K kernels/s (high launch rate)
- HBM: 6.18 GB × 100 = 618 GB/s (~40% A100 peak, ~70% 4g slice)
"""
import argparse, time, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--gpu_mem_util", type=float, default=0.50)
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} batch=1 eager=True", flush=True)
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=1024,
        max_num_seqs=1,
        enforce_eager=True,       # NO CUDA graphs → many kernel launches
    )
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    prompts_pool = []
    try:
        from datasets import load_dataset
        ds = load_dataset("HuggingFaceH4/ultrachat_200k", split="test_gen", streaming=True)
        for i, ex in enumerate(ds):
            if i >= 256: break
            if "prompt" in ex:
                prompts_pool.append(ex["prompt"])
            elif "messages" in ex and ex["messages"]:
                m = next((x["content"] for x in ex["messages"] if x.get("role") == "user"), None)
                if m: prompts_pool.append(m)
        print(f"[{time.strftime('%H:%M:%S')}] loaded {len(prompts_pool)} real prompts", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] real prompts load fail ({e}); fallback", flush=True)

    if len(prompts_pool) < 4:
        prompts_pool = [
            "Explain 5G NR PUSCH channel estimation in 100 words.",
            "How does MMSE equalization handle interference?",
            "Describe the LDPC min-sum decoder briefly.",
            "What is a neural receiver in 5G?",
        ]

    sp = SamplingParams(temperature=0.7, max_tokens=args.max_tokens, top_p=0.9)
    rng = random.Random(42)

    # warmup
    _ = llm.generate([rng.choice(prompts_pool)], sp)
    print(f"[{time.strftime('%H:%M:%S')}] warmup done, single-user chat loop", flush=True)

    start = time.time()
    n_queries = 0
    total_toks = 0
    while time.time() - start < args.duration_s:
        # ONE request at a time (batch=1)
        outs = llm.generate([rng.choice(prompts_pool)], sp)
        n_queries += 1
        total_toks += len(outs[0].outputs[0].token_ids)
        elapsed = time.time() - start
        if n_queries % 5 == 0:
            print(f"[{time.strftime('%H:%M:%S')}] q={n_queries} toks={total_toks} tok/s={total_toks/elapsed:.0f} q/s={n_queries/elapsed:.1f} elapsed={elapsed:.1f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE queries={n_queries} tok/s={total_toks/args.duration_s:.0f} q/s={n_queries/args.duration_s:.2f} label={args.label}", flush=True)

if __name__ == "__main__":
    main()
