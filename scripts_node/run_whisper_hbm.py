"""Whisper batch=4 with REAL LibriSpeech audio (or synth fallback).

CLI: python3 run_whisper_hbm.py <label> <duration_s>
"""
import argparse, time, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--max_tokens", type=int, default=256)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    model_name = "openai/whisper-large-v3"
    device = "cuda"; dtype = torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_name} batch={args.batch}", flush=True)
    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_name, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    # Try to load real LibriSpeech audio; fall back to synth on failure
    audios = None
    try:
        from datasets import load_dataset
        print(f"[{time.strftime('%H:%M:%S')}] loading LibriSpeech test-clean (streaming)", flush=True)
        ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
        samples = list(ds.take(args.batch * 4))  # pool of real audio
        # pad/truncate to 30s @ 16kHz = 480,000 samples
        def to_30s(a):
            a = np.asarray(a, dtype=np.float32)
            if len(a) >= 480000: return a[:480000]
            return np.pad(a, (0, 480000 - len(a)))
        audios = [to_30s(s["audio"]["array"]) for s in samples[:args.batch]]
        print(f"[{time.strftime('%H:%M:%S')}] loaded {len(audios)} real audio clips", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] real audio load failed ({e}); using synth", flush=True)
        rng = np.random.default_rng(42)
        audios = [rng.standard_normal(16000 * 30).astype(np.float32) * 0.05 for _ in range(args.batch)]

    inputs = processor(audios, sampling_rate=16000, return_tensors="pt", padding=True)
    input_features = inputs.input_features.to(device, dtype=dtype)

    start = time.time()
    n_iters = 0
    with torch.inference_mode():
        _ = model.generate(input_features, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        print(f"[{time.strftime('%H:%M:%S')}] warmup done", flush=True)

        while time.time() - start < args.duration_s:
            _ = model.generate(input_features, max_new_tokens=args.max_tokens, do_sample=False)
            torch.cuda.synchronize()
            n_iters += 1
            if n_iters % 2 == 0:
                elapsed = time.time() - start
                # each iter = batch × 30s audio → transcribe rate
                audio_sec_per_wall = args.batch * 30 * n_iters / elapsed
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters} elapsed={elapsed:.1f}s audio_sec_per_wall={audio_sec_per_wall:.0f}", flush=True)

    audio_sec_transcribed = args.batch * 30 * n_iters
    print(f"[{time.strftime('%H:%M:%S')}] DONE iters={n_iters} audio_transcribed_s={audio_sec_transcribed} real_time_factor={audio_sec_transcribed/args.duration_s:.1f}x label={args.label}", flush=True)

if __name__ == "__main__":
    main()
