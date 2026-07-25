"""Whisper-large-v3 streaming ASR loop — memory-bound co-tenant.

CLI: python3 run_whisper.py <label> <duration_s>
- 30초 오디오 청크를 반복 transcribe
- attention이 O(N^2)라 long-sequence에서 HBM BW로 saturate
"""
import argparse, os, sys, time, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

    model_name = "openai/whisper-large-v3"
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_name}", flush=True)

    device = "cuda"
    dtype = torch.float16

    processor = AutoProcessor.from_pretrained(model_name)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    # 30초 랜덤 오디오 (16 kHz, mono) — deterministic seed
    rng = np.random.default_rng(42)
    audio = rng.standard_normal(16000 * 30).astype(np.float32) * 0.05
    inputs = processor(audio, sampling_rate=16000, return_tensors="pt")
    input_features = inputs.input_features.to(device, dtype=dtype)

    start = time.time()
    n_iters = 0
    with torch.inference_mode():
        # 워밍업
        _ = model.generate(input_features, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        print(f"[{time.strftime('%H:%M:%S')}] warmup done, entering loop", flush=True)

        while time.time() - start < args.duration_s:
            _ = model.generate(input_features, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize()
            n_iters += 1
            if n_iters % 5 == 0:
                elapsed = time.time() - start
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters}  elapsed={elapsed:.1f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE  iters={n_iters}  label={args.label}", flush=True)

if __name__ == "__main__":
    main()
