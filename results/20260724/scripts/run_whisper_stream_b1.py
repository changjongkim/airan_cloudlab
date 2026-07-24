"""Whisper batch=1 short 5s audio streaming — realistic ASR + memory-bound decode.

CLI: python3 run_whisper_stream_b1.py <label> <duration_s>

Design (real streaming ASR pattern):
- batch=1 (single stream)
- 5s audio chunks (not full 30s) → shorter encoder seq
- Autoregressive decoder is memory-bound
- Many small kernels per token

Roofline:
- Encoder: seq≈250 tokens (5s @ 16kHz), attention QK^T is compute at seq^2·d
- Decoder: AI ≈ 1 FLOP/byte (memory-bound), 10-20 tokens per iter
- Many launches: ~2000 kernels per (encode + decode) iteration
- ~5 iters/s under alone → 10K kernels/s
- HBM: whisper-large-v3 3GB weights streaming per iter → 15 GB/s (moderate)
"""
import argparse, time, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--audio_len_s", type=int, default=5)
    ap.add_argument("--max_tokens", type=int, default=64)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    model_name = "openai/whisper-large-v3"
    device = "cuda"; dtype = torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_name} batch=1 audio={args.audio_len_s}s", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    # Load real LibriSpeech + cut into short chunks
    audios = None
    try:
        from datasets import load_dataset
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        raw = list(ds.take(32))
        def cut(a, sec):
            n = int(16000 * sec)
            a = np.asarray(a, dtype=np.float32)
            if len(a) >= n: return a[:n]
            return np.pad(a, (0, n - len(a)))
        audios = [cut(s["audio"]["array"], args.audio_len_s) for s in raw]
        print(f"[{time.strftime('%H:%M:%S')}] loaded {len(audios)} real audio chunks", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] real audio fail ({e}); synth", flush=True)
        rng = np.random.default_rng(42)
        audios = [rng.standard_normal(16000 * args.audio_len_s).astype(np.float32) * 0.05 for _ in range(8)]

    # Whisper requires input mel of length 3000 (=30s audio). Pad short clips with zeros.
    def pad_to_30s(a):
        target = 16000 * 30
        if len(a) >= target: return a[:target]
        return np.pad(a, (0, target - len(a)))
    input_features_list = []
    for a in audios[:16]:
        padded = pad_to_30s(a)
        inp = processor(padded, sampling_rate=16000, return_tensors="pt")
        input_features_list.append(inp.input_features.to(device, dtype=dtype))

    idx = 0
    start = time.time()
    n_iters = 0
    with torch.inference_mode():
        _ = model.generate(input_features_list[0], max_new_tokens=16, do_sample=False)
        torch.cuda.synchronize()
        print(f"[{time.strftime('%H:%M:%S')}] warmup done, streaming loop", flush=True)

        while time.time() - start < args.duration_s:
            feat = input_features_list[idx % len(input_features_list)]
            _ = model.generate(feat, max_new_tokens=args.max_tokens, do_sample=False)
            torch.cuda.synchronize()
            n_iters += 1
            idx += 1
            if n_iters % 5 == 0:
                elapsed = time.time() - start
                aud_s = args.audio_len_s * n_iters
                rtf = aud_s / elapsed
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters} audio_s={aud_s} rtf={rtf:.1f}x elapsed={elapsed:.1f}s", flush=True)

    aud_s = args.audio_len_s * n_iters
    print(f"[{time.strftime('%H:%M:%S')}] DONE iters={n_iters} audio_s={aud_s} rtf={aud_s/args.duration_s:.1f}x label={args.label}", flush=True)

if __name__ == "__main__":
    main()
