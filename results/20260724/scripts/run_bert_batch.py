"""BERT-large batch-parameterized continuous inference.

CLI: python3 run_bert_batch.py <label> <duration_s> --batch N
"""
import argparse, time, random

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--model", default="bert-large-uncased")
    ap.add_argument("--max_length", type=int, default=384)
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel

    device = "cuda"; dtype = torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} batch={args.batch}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, torch_dtype=dtype).to(device).eval()
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    texts_pool = []
    try:
        from datasets import load_dataset
        ds = load_dataset("glue", "sst2", split="validation")
        texts_pool = [ex["sentence"] for ex in list(ds)[:128]]
        print(f"[{time.strftime('%H:%M:%S')}] loaded {len(texts_pool)} SST-2 texts", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] real texts fail ({e}); fallback", flush=True)
    if len(texts_pool) < args.batch:
        texts_pool = [
            "The 5G physical layer uses OFDM with configurable numerology.",
            "MMSE channel equalization is a standard technique for MIMO.",
            "Neural receivers improve BLER at high SNR.",
            "LDPC decoding uses layered belief propagation.",
        ] * max(1, args.batch)

    rng = random.Random(42)

    def sample_batch():
        return [rng.choice(texts_pool) for _ in range(args.batch)]

    # warmup
    with torch.inference_mode():
        inp = tok(sample_batch(), return_tensors="pt", padding="max_length",
                  max_length=args.max_length, truncation=True).to(device)
        _ = model(**inp); torch.cuda.synchronize()
        print(f"[{time.strftime('%H:%M:%S')}] warmup done, batch={args.batch} loop", flush=True)

        start = time.time()
        n_iters = 0
        while time.time() - start < args.duration_s:
            inp = tok(sample_batch(), return_tensors="pt", padding="max_length",
                      max_length=args.max_length, truncation=True).to(device)
            _ = model(**inp); torch.cuda.synchronize()
            n_iters += 1
            if n_iters % 20 == 0:
                elapsed = time.time() - start
                fwd_per_s = n_iters / elapsed
                samples_per_s = fwd_per_s * args.batch
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters} fwd/s={fwd_per_s:.1f} samples/s={samples_per_s:.0f}", flush=True)

    fwd_per_s = n_iters / args.duration_s
    samples_per_s = fwd_per_s * args.batch
    print(f"[{time.strftime('%H:%M:%S')}] DONE iters={n_iters} fwd/s={fwd_per_s:.1f} samples/s={samples_per_s:.0f} label={args.label}", flush=True)

if __name__ == "__main__":
    main()
