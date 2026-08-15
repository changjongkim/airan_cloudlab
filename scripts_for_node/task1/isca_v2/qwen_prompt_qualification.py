#!/usr/bin/env python3
"""Qwen prefill/decode service characterization with provenance-recorded text."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from pathlib import Path

os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("HF_HOME", "/mydata/hf_cache")

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


def load_prompt_tokens(
    tokenizer, dataset_path: Path, target_tokens: int, seed: int
) -> tuple[torch.Tensor, dict]:
    raw = dataset_path.read_bytes()
    records = json.loads(raw)
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(records))
    token_ids: list[int] = []
    used_indices: list[int] = []
    for index in order:
        item = records[int(index)]
        text = f"Instruction:\n{item['instruction'].strip()}"
        if item.get("input", "").strip():
            text += f"\nInput:\n{item['input'].strip()}"
        text += "\nResponse:\n"
        encoded = tokenizer.encode(text, add_special_tokens=False)
        if not encoded:
            continue
        token_ids.extend(encoded)
        used_indices.append(int(index))
        if len(token_ids) >= target_tokens:
            break
    if len(token_ids) < target_tokens:
        raise RuntimeError(
            f"dataset produced only {len(token_ids)} of {target_tokens} tokens"
        )
    tokens = torch.tensor(
        [token_ids[:target_tokens]], dtype=torch.long, device="cuda:0"
    )
    provenance = {
        "path": str(dataset_path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "format": "Stanford Alpaca instruction records",
        "record_count": len(records),
        "selected_record_indices": used_indices,
        "constructed_prompt": True,
        "production_trace": False,
        "target_tokens": target_tokens,
    }
    return tokens, provenance


def timed_call(function) -> tuple[object, float, float]:
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    wall_start = time.perf_counter_ns()
    begin.record()
    value = function()
    end.record()
    end.synchronize()
    wall_ms = (time.perf_counter_ns() - wall_start) / 1e6
    return value, float(begin.elapsed_time(end)), wall_ms


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--duration", type=float, default=90.0)
    parser.add_argument(
        "--duty-cycle", type=float, choices=(0.5, 0.9), required=True
    )
    parser.add_argument("--mode", choices=("prefill", "decode"), required=True)
    parser.add_argument("--target-tokens", type=int, default=512)
    parser.add_argument("--decode-steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--model", default="Qwen/Qwen2.5-7B")
    args = parser.parse_args()
    if args.trial <= 0 or args.duration <= 0 or args.target_tokens <= 0:
        parser.error("trial, duration, and target tokens must be positive")

    torch.cuda.set_device(0)
    torch.manual_seed(0xD47A + args.trial)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        local_files_only=True,
    ).eval()
    tokens, provenance = load_prompt_tokens(
        tokenizer, Path(args.dataset), args.target_tokens, 0xA17 + args.trial
    )

    past_key_values = None
    next_token = None
    decode_step = 0

    def prefill(use_cache: bool):
        return model(tokens, use_cache=use_cache)

    def reset_decode() -> float:
        nonlocal past_key_values, next_token, decode_step
        output, gpu_ms, _ = timed_call(lambda: prefill(True))
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1:, :].argmax(dim=-1)
        decode_step = 0
        return gpu_ms

    def decode_once():
        nonlocal past_key_values, next_token, decode_step
        output = model(
            next_token, past_key_values=past_key_values, use_cache=True
        )
        past_key_values = output.past_key_values
        next_token = output.logits[:, -1:, :].argmax(dim=-1)
        decode_step += 1
        return output

    calibration_ms: list[float] = []
    reset_gpu_ms: list[float] = []
    with torch.inference_mode():
        if args.mode == "decode":
            reset_gpu_ms.append(reset_decode())
            for _ in range(args.warmup):
                _, gpu_ms, _ = timed_call(decode_once)
                calibration_ms.append(gpu_ms)
        else:
            for _ in range(args.warmup):
                _, gpu_ms, _ = timed_call(lambda: prefill(False))
                calibration_ms.append(gpu_ms)
    if not calibration_ms:
        raise RuntimeError("warmup must produce at least one calibration sample")
    interval_ns = round(np.median(calibration_ms) * 1e6 / args.duty_cycle)

    records = []
    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns + round(args.duration * 1e9)
    target_ns = start_ns
    with torch.inference_mode():
        while time.perf_counter_ns() < deadline_ns:
            wait_until(target_ns)
            arrival_ns = target_ns
            actual_start_ns = time.perf_counter_ns()
            if args.mode == "decode" and decode_step >= args.decode_steps:
                reset_gpu_ms.append(reset_decode())
            if args.mode == "decode":
                _, gpu_ms, wall_ms = timed_call(decode_once)
                step = decode_step
            else:
                _, gpu_ms, wall_ms = timed_call(lambda: prefill(False))
                step = None
            records.append(
                {
                    "arrival_ns": arrival_ns,
                    "actual_start_ns": actual_start_ns,
                    "scheduler_lateness_us": (actual_start_ns - arrival_ns) / 1e3,
                    "gpu_ms": gpu_ms,
                    "wall_ms": wall_ms,
                    "decode_step": step,
                }
            )
            target_ns += interval_ns
    end_ns = time.perf_counter_ns()

    gpu_values = [item["gpu_ms"] for item in records]
    wall_values = [item["wall_ms"] for item in records]
    result = {
        "schema": "qwen-text-qualification-v1",
        "model": args.model,
        "mode": args.mode,
        "trial": args.trial,
        "target_duty_cycle": args.duty_cycle,
        "duration_target_s": args.duration,
        "duration_actual_s": (end_ns - start_ns) / 1e9,
        "target_interval_ms": interval_ns / 1e6,
        "calibration_gpu_ms": stats(calibration_ms),
        "service_gpu_ms": stats(gpu_values),
        "service_wall_ms": stats(wall_values),
        "reset_prefill_gpu_ms": stats(reset_gpu_ms) if reset_gpu_ms else {"n": 0},
        "units_per_s": len(records) / ((end_ns - start_ns) / 1e9),
        "provenance": provenance,
        "records": records,
        "pass": bool(records) and all(np.isfinite(gpu_values)),
    }
    if not result["pass"]:
        raise RuntimeError("Qwen qualification produced invalid measurements")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8")
    temporary.replace(output)
    print(
        f"[QWEN-QUAL] PASS mode={args.mode} duty={args.duty_cycle} "
        f"n={len(records)} gpu_p99={result['service_gpu_ms']['p99']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
