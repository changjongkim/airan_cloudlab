"""Qwen2-VL-2B batch=2 continuous with REAL COCO images (or synth fallback).

CLI: python3 run_qwen_vl_hbm.py <label> <duration_s>
"""
import argparse, time, numpy as np

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("label")
    ap.add_argument("duration_s", type=int)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--max_tokens", type=int, default=256)
    ap.add_argument("--model", default="Qwen/Qwen2-VL-2B-Instruct")
    ap.add_argument("--img_res", type=int, default=560)
    args = ap.parse_args()

    import torch
    from transformers import AutoProcessor, Qwen2VLForConditionalGeneration
    from PIL import Image

    device = "cuda"; dtype = torch.float16
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.model} batch={args.batch}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model)
    model = Qwen2VLForConditionalGeneration.from_pretrained(args.model, torch_dtype=dtype, low_cpu_mem_usage=True).to(device).eval()
    print(f"[{time.strftime('%H:%M:%S')}] model ready", flush=True)

    # Try real COCO validation images (small subset); fall back to synth
    imgs = None
    try:
        from datasets import load_dataset
        print(f"[{time.strftime('%H:%M:%S')}] loading COCO images (streaming)", flush=True)
        ds = load_dataset("HuggingFaceM4/COCO", split="validation", streaming=True)
        raw_imgs = []
        for i, ex in enumerate(ds):
            if i >= args.batch: break
            raw_imgs.append(ex["image"].convert("RGB").resize((args.img_res, args.img_res)))
        imgs = raw_imgs
        print(f"[{time.strftime('%H:%M:%S')}] loaded {len(imgs)} real COCO images", flush=True)
    except Exception as e:
        print(f"[{time.strftime('%H:%M:%S')}] real COCO load failed ({e}); using synth", flush=True)
        rng = np.random.default_rng(0)
        imgs = [Image.fromarray((rng.random((args.img_res, args.img_res, 3)) * 255).astype(np.uint8)) for _ in range(args.batch)]

    prompts = ["Describe this scene in detail and identify all objects."] * args.batch

    messages = [[{"role": "user", "content": [{"type": "image", "image": im}, {"type": "text", "text": p}]}]
                for im, p in zip(imgs, prompts)]
    texts = [processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in messages]
    inputs = processor(text=texts, images=imgs, padding=True, return_tensors="pt").to(device, dtype=dtype)

    start = time.time()
    n_iters = 0
    with torch.inference_mode():
        _ = model.generate(**inputs, max_new_tokens=32, do_sample=False)
        torch.cuda.synchronize()
        print(f"[{time.strftime('%H:%M:%S')}] warmup done", flush=True)

        while time.time() - start < args.duration_s:
            _ = model.generate(**inputs, max_new_tokens=args.max_tokens, do_sample=False)
            torch.cuda.synchronize()
            n_iters += 1
            if n_iters % 2 == 0:
                elapsed = time.time() - start
                imgs_per_s = args.batch * n_iters / elapsed
                print(f"[{time.strftime('%H:%M:%S')}] iter={n_iters} elapsed={elapsed:.1f}s imgs_per_s={imgs_per_s:.2f}", flush=True)

    imgs_processed = args.batch * n_iters
    print(f"[{time.strftime('%H:%M:%S')}] DONE iters={n_iters} images_processed={imgs_processed} img_per_s={imgs_processed/args.duration_s:.2f} label={args.label}", flush=True)

if __name__ == "__main__":
    main()
