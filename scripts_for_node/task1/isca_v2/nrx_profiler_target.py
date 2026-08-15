#!/usr/bin/env python3
"""Small NVTX-scoped direct-NRx target for Nsight Systems/Compute."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cupy as cp

from nrx_trt_direct import DirectNrx


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--iterations", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations positive and warmup non-negative")
    runtime = DirectNrx(args.engine)
    runtime.capture_graph()
    for _ in range(args.warmup):
        runtime.launch(use_graph=True)
    runtime.stream.synchronize()
    cp.cuda.nvtx.RangePush("nrx_inference")
    start = cp.cuda.Event()
    end = cp.cuda.Event()
    with runtime.stream:
        start.record()
        for _ in range(args.iterations):
            runtime.launch(use_graph=True)
        end.record()
    end.synchronize()
    cp.cuda.nvtx.RangePop()
    result = {
        "iterations": args.iterations,
        "total_gpu_ms": float(cp.cuda.get_elapsed_time(start, end)),
        "outputs_finite": all(
            bool(cp.all(cp.isfinite(value)).item())
            for value in runtime.outputs.values()
        ),
    }
    if not result["outputs_finite"]:
        raise RuntimeError("non-finite NRx output")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"[NRX-PROFILER] PASS {result}", flush=True)


if __name__ == "__main__":
    main()
