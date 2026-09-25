#!/usr/bin/env python3
"""Open-loop full-path RAN latency measurement for the SoftWall P0 gate."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from softwall_phy import ConventionalRan, summary, wait_until


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--cells", type=int, default=1)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=30)
    parser.add_argument("--period-ms", type=float, default=10.0)
    parser.add_argument("--deadline-ms", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--ready-file")
    parser.add_argument("--start-file")
    args = parser.parse_args()
    if min(args.cells, args.iterations) <= 0 or args.warmup < 0:
        parser.error("cells/iterations must be positive and warmup non-negative")
    if args.period_ms <= 0 or args.deadline_ms <= 0:
        parser.error("period and deadline must be positive")

    ran = ConventionalRan(seed=args.seed)
    for _ in range(args.warmup):
        ran.run_cells(args.cells)
    if args.ready_file:
        Path(args.ready_file).touch()
    if args.start_file:
        while not Path(args.start_file).exists():
            time.sleep(0.001)

    records = []
    first_release_ns = time.perf_counter_ns() + 50_000_000
    for index in range(args.iterations):
        release_ns = first_release_ns + round(index * args.period_ms * 1e6)
        wait_until(release_ns)
        start_ns = time.perf_counter_ns()
        gpu_ms = ran.run_cells(args.cells)
        completed_ns = time.perf_counter_ns()
        response_ms = (completed_ns - release_ns) / 1e6
        records.append({
            "index": index,
            "release_ns": release_ns,
            "start_lateness_ms": (start_ns - release_ns) / 1e6,
            "gpu_ms": gpu_ms,
            "service_ms": (completed_ns - start_ns) / 1e6,
            "response_ms": response_ms,
            "deadline_miss": response_ms > args.deadline_ms,
        })

    result = {
        "schema": "softwall-ran-fullpath-v1",
        "label": args.label,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mps_active_thread_percentage": os.environ.get(
            "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
        ),
        "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
        "visible_sm_count": int(
            cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
        ),
        "cells": args.cells,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "period_ms": args.period_ms,
        "deadline_ms": args.deadline_ms,
        "input": "fixed-seed synthetic complex RX tensor",
        "path": "CE+NI+EQ+de-rate-match+LDPC+CRC",
        "gpu_ms": summary([x["gpu_ms"] for x in records]),
        "service_ms": summary([x["service_ms"] for x in records]),
        "response_ms": summary([x["response_ms"] for x in records]),
        "deadline_misses": sum(x["deadline_miss"] for x in records),
        "records": records,
    }
    atomic_json(Path(args.output), result)
    print(
        f"[RAN-FULL] {args.label} n={args.iterations} "
        f"mean={result['response_ms']['mean']:.3f}ms "
        f"p99={result['response_ms']['p99']:.3f}ms "
        f"max={result['response_ms']['max']:.3f}ms "
        f"miss={result['deadline_misses']}",
        flush=True,
    )
    cp.cuda.runtime.deviceSynchronize()


if __name__ == "__main__":
    main()
