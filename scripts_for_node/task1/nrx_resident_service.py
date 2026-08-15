#!/usr/bin/env python3
"""Keep a warmed TensorRT NRx endpoint resident until terminated."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path

import cupy as cp

from nrx_trt_direct import DirectNrx


STOP = False


def stop(_signum, _frame):
    global STOP
    STOP = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--ready-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    process_start_ns = time.monotonic_ns()
    runtime = DirectNrx(args.engine)
    engine_ready_ns = time.monotonic_ns()
    runtime.capture_graph()
    for _ in range(args.warmup):
        runtime.launch(use_graph=True)
    runtime.stream.synchronize()
    warm_ready_ns = time.monotonic_ns()
    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    result = {
        "process_start_ns": process_start_ns,
        "engine_ready_ns": engine_ready_ns,
        "warm_ready_ns": warm_ready_ns,
        "deserialize_ms": (engine_ready_ns - process_start_ns) / 1e6,
        "warmup_ms": (warm_ready_ns - engine_ready_ns) / 1e6,
        "startup_ms": (warm_ready_ns - process_start_ns) / 1e6,
        "warmup_iterations": args.warmup,
        "device_name": name,
        "multiprocessor_count": int(properties["multiProcessorCount"]),
        "total_memory_bytes": int(properties["totalGlobalMem"]),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    Path(args.ready_file).write_text(str(warm_ready_ns), encoding="utf-8")
    print(f"[NRX-RESIDENT] ready startup_ms={result['startup_ms']:.3f}", flush=True)
    while not STOP:
        time.sleep(0.05)
    print("[NRX-RESIDENT] stopped", flush=True)


if __name__ == "__main__":
    main()
