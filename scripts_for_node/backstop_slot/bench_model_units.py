#!/usr/bin/env python3
"""Isolated GPU time of the prefill units of one model, per chunk size."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from qwen_units import QwenUnits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--chunks", default="128,512,1024")
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    chunks = tuple(int(c) for c in args.chunks.split(","))
    runner = QwenUnits(args.model, 0, chunks=chunks)
    ids = torch.randint(0, 32000, (1000,), generator=torch.Generator().manual_seed(5))
    runner.load_prompt(ids.to(runner.device))
    for c in chunks:
        for name in runner.chunk_units(c):
            runner.launch(c, name, 0)
    runner.stream.synchronize()
    times = {}
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for c in chunks:
        runner._prompt_len = c
        times[str(c)] = {}
        for name, key in (("prep", "prep"), (f"layer{runner.layers // 2}", "layer"), ("head", "head")):
            samples = []
            for _ in range(args.repeats):
                begin.record(runner.stream)
                runner.launch(c, name, 0)
                end.record(runner.stream)
                end.synchronize()
                samples.append(begin.elapsed_time(end))
            a = np.asarray(samples)
            times[str(c)][key] = {"p50": float(np.percentile(a, 50)), "p999": float(np.percentile(a, 99.9))}
    result = {"model": args.model, "layers": runner.layers, "unit_gpu_ms": times}
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result))


if __name__ == "__main__":
    main()
