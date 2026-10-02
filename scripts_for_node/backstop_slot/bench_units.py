#!/usr/bin/env python3
"""Check mixed-chunk Qwen prefill against Hugging Face and time units per chunk."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from qwen_units import QwenUnits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunks", default="128,512")
    parser.add_argument("--repeats", type=int, default=100)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = tuple(int(c) for c in args.chunks.split(","))
    runner = QwenUnits("Qwen/Qwen2.5-1.5B", 0, chunks=chunks)
    rng = random.Random(3)
    gen = torch.Generator(device="cpu").manual_seed(5)
    checks = []
    for length in (90, 300, 700, 1000):
        ids = torch.randint(0, 32000, (length,), generator=gen)
        runner.load_prompt(ids.to(runner.device))
        done, used, last = 0, [], None
        while done < length:
            room = [c for c in chunks if done + c <= runner.max_ctx]
            c = rng.choice(room) if length - done > min(chunks) else min(chunks)
            used.append(c)
            for name in runner.chunk_units(c):
                runner.launch(c, name, done)
            last = (c, done)
            done += c
        runner.launch(last[0], "head", last[1])
        runner.stream.synchronize()
        ref = runner.reference_logits(ids)
        checks.append({"prompt_len": length, "chunks": used,
                       "argmax_match": int(torch.argmax(ref)) == int(runner.token),
                       "max_abs_logit_diff": float((ref - runner.logits.float()).abs().max())})
    times = {}
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for c in chunks:
        runner._prompt_len = c          # head reads the last token of this chunk
        for name, key in (("prep", "prep"), ("layer5", "layer"), ("head", "head")):
            samples = []
            for _ in range(args.repeats):
                begin.record(runner.stream)
                runner.launch(c, name, 0)
                end.record(runner.stream)
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            a = np.asarray(samples)
            times[f"{c}:{key}"] = {"p50": float(np.percentile(a, 50)), "p999": float(np.percentile(a, 99.9)),
                                   "max": float(a.max())}
    result = {"checks": checks, "unit_gpu_ms": times}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
