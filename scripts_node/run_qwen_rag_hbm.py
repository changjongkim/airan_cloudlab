"""Qwen-3B RAG with continuous batching + REAL prompts from HF dataset.

CLI: python3 run_qwen_rag_hbm.py <label> <duration_s> [options]

Real-world simulation:
- Prompts drawn from HuggingFace dataset (default: ShareGPT-style ultrachat_200k)
- Variable prompt lengths (128–2048 tokens) → variable KV cache pressure
- n=64 concurrent requests (vLLM continuous batching) — production LLM serving pattern

HBM bandwidth: with real variable-length prompts + n=64 concurrency,
Qwen-3B params (~6GB fp16) + KV cache scales with total tokens → sustained 300-600 GB/s
"""
import argparse, time, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--rag", action="store_true")
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--n_concurrent", type=int, default=64)
    ap.add_argument("--max_tokens", type=int, default=512)
    ap.add_argument("--gpu_mem_util", type=float, default=0.60)
    ap.add_argument("--dataset", default="HuggingFaceH4/ultrachat_200k")
    ap.add_argument("--dataset_split", default="test_gen")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams

    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} (gpu_mem_util={args.gpu_mem_util})", flush=True)
    llm = LLM(
        model=args.model,
        dtype="float16",
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=2048,
        max_num_seqs=args.n_concurrent,
        enforce_eager=False,
    )
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    # Load real prompts from HuggingFace dataset (fallback to hardcoded if fails)
    prompts_pool = []
    try:
        from datasets import load_dataset
        print(f"[{time.strftime('%H:%M:%S')}] loading dataset {args.dataset}", flush=True)
        ds = load_dataset(args.dataset, split=args.dataset_split, streaming=True)
        for i, ex in enumerate(ds):
            if i >= 512: break
            # ultrachat schema: prompt or messages
            if "prompt" in ex:
                prompts_pool.append(ex["prompt"])
            elif "messages" in ex and ex["messages"]:
                first_user = next((m["content"] for m in ex["messages"] if m.get("role") == "user"), None)
                if first_user: prompts_pool.append(first_user)
        print(f"[{time.strftime('%H:%M:%S')}] loaded {len(prompts_pool)} real prompts", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] dataset load failed ({e}); using fallback prompts", flush=True)

    if len(prompts_pool) < 8:
        prompts_pool = [
            "Explain the OFDM waveform used in 5G NR uplink.",
            "How does the LDPC decoder converge?",
            "Describe MMSE equalization for MU-MIMO.",
            "What are the tradeoffs of Neural Receiver vs classical MMSE?",
            "Summarize O-RAN Near-RT RIC architecture.",
            "How does DMRS pilot channel estimation work?",
            "Explain HARQ retransmission and throughput impact.",
            "Beam management in 5G NR mmWave — measurement procedure?",
        ]

    rag_prefix = (
        "Retrieved context: 5G NR PHY uses OFDM with configurable numerology. "
        "PUSCH DMRS pilots enable LMMSE channel estimation. TensorRT-based neural "
        "receivers can improve BLER at high SNR. LDPC decoding uses layered min-sum.\n\nQuestion: "
    )

    rng = random.Random(42)

    def sample_prompts(n):
        chosen = rng.sample(prompts_pool, min(n, len(prompts_pool))) if len(prompts_pool) >= n else \
                 [rng.choice(prompts_pool) for _ in range(n)]
        return [rag_prefix + p if args.rag else p for p in chosen]

    sp = SamplingParams(temperature=0.8, max_tokens=args.max_tokens, top_p=0.9)

    _ = llm.generate(sample_prompts(8), sp)  # warmup
    print(f"[{time.strftime('%H:%M:%S')}] warmup done, main loop", flush=True)

    start = time.time()
    n_iters = 0
    total_toks = 0
    while time.time() - start < args.duration_s:
        outs = llm.generate(sample_prompts(args.n_concurrent * 2), sp)
        n_iters += 1
        total_toks += sum(len(o.outputs[0].token_ids) for o in outs)
        elapsed = time.time() - start
        print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters} toks={total_toks} tok/s={total_toks/elapsed:.0f} elapsed={elapsed:.1f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE iters={n_iters} total_toks={total_toks} tok_per_s={total_toks/args.duration_s:.0f} label={args.label}", flush=True)

if __name__ == "__main__":
    main()
