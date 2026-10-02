#!/usr/bin/env python3
"""Check the unit-split Qwen prefill against Hugging Face and time each unit."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch

from qwen_pieces import QwenPieces


def dist(values):
    a = np.asarray(values, dtype=np.float64)
    return {"n": int(a.size), "p50": float(np.percentile(a, 50)),
            "p99": float(np.percentile(a, 99)), "p999": float(np.percentile(a, 99.9)),
            "max": float(a.max())}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--chunk", type=int, default=64)
    parser.add_argument("--repeats", type=int, default=400)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    runner = QwenPieces(args.model, 0, chunk=args.chunk)
    generator = torch.Generator(device="cpu").manual_seed(7)
    checks = []
    for length in (17, 100, 256, 500):
        ids = torch.randint(0, 32000, (length,), generator=generator)
        runner.load_prompt(ids.to(runner.device))
        for name, chunk in runner.units_for(length):
            runner.launch(name, chunk)
        runner.stream.synchronize()
        reference = runner.reference_logits(ids)
        ours = runner.logits.float()
        checks.append({
            "prompt_len": length,
            "argmax_match": bool(int(torch.argmax(reference)) == int(runner.token)),
            "max_abs_logit_diff": float((reference - ours).abs().max()),
            "reference_logit_scale": float(reference.abs().max()),
        })

    times: dict[str, list[float]] = {}
    ids = torch.randint(0, 32000, (512,), generator=generator).to(runner.device)
    runner.load_prompt(ids)
    begin = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    for _ in range(args.repeats):
        for name, chunk in runner.units_for(512):
            key = "layer" if name.startswith("layer") else name
            begin.record(runner.stream)
            runner.launch(name, chunk)
            end.record(runner.stream)
            end.synchronize()
            times.setdefault(key, []).append(begin.elapsed_time(end))
    whole = []
    for length in (64, 128, 256, 512):
        runner.load_prompt(ids[:length])
        samples = []
        for _ in range(30):
            begin.record(runner.stream)
            for name, chunk in runner.units_for(length):
                runner.launch(name, chunk)
            end.record(runner.stream)
            end.synchronize()
            samples.append(begin.elapsed_time(end))
        with torch.inference_mode():
            hf = []
            for _ in range(10):
                torch.cuda.synchronize()
                t = time.perf_counter()
                runner.hf.model(ids[:length][None], use_cache=False)
                torch.cuda.synchronize()
                hf.append((time.perf_counter() - t) * 1e3)
        whole.append({"prompt_len": length, "units": len(runner.units_for(length)),
                      "units_back_to_back_ms": dist(samples), "hf_eager_ms": dist(hf)})
    result = {
        "schema": "backstop-slot-qwen-units-v1",
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "gpu": torch.cuda.get_device_name(0), "model": args.model, "chunk": args.chunk,
        "torch": torch.__version__, "checks": checks,
        "unit_gpu_ms": {k: dist(v) for k, v in times.items()},
        "whole_prefill": whole,
    }
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("checks", "unit_gpu_ms", "whole_prefill")}, indent=1))


if __name__ == "__main__":
    main()
