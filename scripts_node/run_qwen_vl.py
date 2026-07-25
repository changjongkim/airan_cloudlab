"""Qwen2-VL-7B-Instruct image + text loop — memory-bound (VLM) co-tenant.

CLI: python3 run_qwen_vl.py <label> <duration_s>
- 512x512 랜덤 이미지 + prompt를 반복 inference
- ViT encoder + LLM decoder 둘 다 HBM BW bound
"""
import argparse, os, time, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image

    model_name = "Qwen/Qwen2-VL-7B-Instruct"
    print(f"[{time.strftime('%H:%M:%S')}] loading {model_name}", flush=True)

    device = "cuda"
    dtype = torch.float16

    processor = AutoProcessor.from_pretrained(model_name)
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name, torch_dtype=dtype, low_cpu_mem_usage=True
    ).to(device)
    model.eval()
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    rng = np.random.default_rng(0)
    img_np = (rng.random((512, 512, 3)) * 255).astype(np.uint8)
    img = Image.fromarray(img_np)

    prompt = "Describe this traffic scene and identify any potential hazards for autonomous driving."
    messages = [
        {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": prompt}]}
    ]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[img], padding=True, return_tensors="pt").to(device, dtype=dtype)

    start = time.time()
    n_iters = 0
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        torch.cuda.synchronize()
        print(f"[{time.strftime('%H:%M:%S')}] warmup done", flush=True)

        while time.time() - start < args.duration_s:
            _ = model.generate(**inputs, max_new_tokens=128, do_sample=False)
            torch.cuda.synchronize()
            n_iters += 1
            if n_iters % 3 == 0:
                elapsed = time.time() - start
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters}  elapsed={elapsed:.1f}s", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] DONE  iters={n_iters}  label={args.label}", flush=True)

if __name__ == "__main__":
    main()
