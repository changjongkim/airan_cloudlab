"""BERT-large batch=1 continuous inference — realistic latency-critical NLP + memory-bound.

CLI: python3 run_bert_b1.py <label> <duration_s>

Design (real production BERT deployment):
- BERT-large-uncased (340M params, ~680MB fp16)
- batch=1 (single request at a time — latency-critical serving)
- max_length=384 (typical NLU task length)
- Real inputs from GLUE-like data (or fallback random text)

Roofline:
- 340M params × 2 bytes = 680MB weight read per fwd
- Compute: ~110 GFLOPs (24 layers × attention + ffn)
- AI ≈ 160 FLOPs/byte → borderline (compute-bound at batch>=1 usually)
- BUT at batch=1 with short seq: weight bandwidth dominates → memory-bound
- Kernels per fwd: ~500 (24 layers × 20 kernels each)
- ~30 fwd/s → 15K kernels/s (high launch rate)
"""
import argparse, time, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--model", default="bert-large-uncased")
    ap.add_argument("--max_length", type=int, default=384)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda"; dtype = torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} batch=1", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    # Realistic input texts (from prompts pool if available)
    texts = []
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        texts = [ex["sentence"] for ex in list(ds)[:32]]
        print(f"[{time.strftime('%H:%M:%S')}] loaded {len(texts)} real SST-2 sentences", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] real texts fail ({e}); fallback", flush=True)

    if len(texts) < 4:
        texts = [
            "The 5G physical layer uses OFDM with configurable numerology.",
            "MMSE channel equalization is a standard technique for MIMO uplink.",
            "Neural receivers can improve BLER performance at high SNR.",
            "The LDPC decoder converges through layered belief propagation.",
        ]

    rng = random.Random(42)

    # Warmup
    with torch.inference_mode():
        inp = tok(rng.choice(texts), return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True).to(device)
        _ = model(**inp)
        torch.cuda.synchronize()
        print(f"[{time.strftime('%H:%M:%S')}] warmup done, batch=1 continuous loop", flush=True)

        start = time.time()
        n_iters = 0
        while time.time() - start < args.duration_s:
            inp = tok(rng.choice(texts), return_tensors="pt", padding="max_length", max_length=args.max_length, truncation=True).to(device)
            _ = model(**inp)
            torch.cuda.synchronize()
            n_iters += 1
            if n_iters % 50 == 0:
                elapsed = time.time() - start
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters} elapsed={elapsed:.1f}s fwd_per_s={n_iters/elapsed:.1f}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE iters={n_iters} fwd_per_s={n_iters/args.duration_s:.1f} label={args.label}", flush=True)

if __name__ == "__main__":
    main()
