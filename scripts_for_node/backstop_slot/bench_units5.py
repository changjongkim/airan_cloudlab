#!/usr/bin/env python3
"""GPU time of every AI unit alone: prefill and decode units of an LLM, units of an encoder.

usage: bench_units5.py --model NAME --kind llm|encoder [--chunks 128,512,1024] [--batches 1,4,16]
                       [--decode-groups 4] --output FILE
The result gives, per unit, the median and the 99.9th percentile of its GPU time and the bound
used by the AI worker (99.9th percentile plus 10%, at least 0.05 ms).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from model_units import ChatUnits, EncoderUnits


def timed(stream, launch, repeats: int) -> dict:
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(repeats):
        begin.record(stream)
        launch()
        end.record(stream)
        end.synchronize()
        samples.append(begin.elapsed_time(end))
    a = np.asarray(samples[repeats // 10:])
    p999 = float(np.percentile(a, 99.9))
    return {"p50": float(np.percentile(a, 50)), "p999": p999, "bound": max(0.05, round(1.1 * p999, 3))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--kind", choices=("llm", "encoder"), required=True)
    parser.add_argument("--chunks", default="128,512,1024")
    parser.add_argument("--batches", default="1,4,16")
    parser.add_argument("--decode-groups", type=int, default=4)
    parser.add_argument("--max-ctx", type=int, default=1152)
    parser.add_argument("--repeats", type=int, default=300)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    out = {"model": args.model, "kind": args.kind, "units_ms": {}}
    if args.kind == "llm":
        chunks = tuple(int(c) for c in args.chunks.split(","))
        runner = ChatUnits(args.model, 0, chunks=chunks, max_ctx=args.max_ctx, decode_groups=args.decode_groups)
        ids = torch.randint(0, 32000, (1024,), generator=torch.Generator().manual_seed(5))
        runner.load_prompt(ids.to(runner.device))
        out["layers"], out["decode_groups"] = runner.layers, args.decode_groups
        for c in chunks:
            runner._prompt_len = c
            out["units_ms"][str(c)] = {
                "prep": timed(runner.stream, lambda: runner.launch(c, "prep", 0), args.repeats),
                "layer": timed(runner.stream, lambda: runner.launch(c, f"layer{runner.layers // 2}", 0), args.repeats),
                "head": timed(runner.stream, lambda: runner.launch(c, "head", 0), args.repeats),
            }
        out["units_ms"]["decode"] = {
            f"dec{g}": timed(runner.stream, (lambda gg: (lambda: runner.launch_decode(gg, 600)))(g), args.repeats)
            for g in range(args.decode_groups)}
        out["memory_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    else:
        batches = tuple(int(b) for b in args.batches.split(","))
        runner = EncoderUnits(args.model, 0, batches=batches)
        generator = torch.Generator().manual_seed(5)
        out["layers"] = runner.layers
        for b in batches:
            runner.load_inputs(b, generator)
            out["units_ms"][str(b)] = {
                "embed": timed(runner.stream, lambda: runner.launch(b, "embed"), args.repeats),
                "layer": timed(runner.stream, lambda: runner.launch(b, f"layer{runner.layers // 2}"), args.repeats),
                "pool": timed(runner.stream, lambda: runner.launch(b, "pool"), args.repeats),
            }
        out["memory_gib"] = torch.cuda.max_memory_allocated() / 2 ** 30
    args.output.write_text(json.dumps(out, indent=1))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
