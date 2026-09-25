#!/usr/bin/env python3
"""Measure an optimistic local receiver lower bound against testMAC's 4.5 ms."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import numpy as np

from dual_receiver_phy import PairedDualReceiver


def stats(values: list[float]) -> dict:
    data = np.asarray(values, dtype=np.float64)
    return {
        "count": int(data.size),
        "mean": float(data.mean()),
        "p50": float(np.percentile(data, 50)),
        "p95": float(np.percentile(data, 95)),
        "p99": float(np.percentile(data, 99)),
        "min": float(data.min()),
        "max": float(data.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20357800)
    parser.add_argument("--deadline-ms", type=float, default=4.5)
    args = parser.parse_args()

    receiver = PairedDualReceiver(args.engine, seed=args.seed)
    for _ in range(args.warmup):
        receiver.run_neural()
        receiver.run_conventional()

    records = []
    for iteration in range(args.iterations):
        begin = time.perf_counter_ns()
        neural = receiver.run_neural()
        neural_wall = (time.perf_counter_ns() - begin) / 1e6
        begin = time.perf_counter_ns()
        conventional = receiver.run_conventional()
        conventional_wall = (time.perf_counter_ns() - begin) / 1e6
        records.append({
            "iteration": iteration,
            "neural_gpu_ms": neural[0],
            "neural_wall_ms": neural_wall,
            "neural_correct": bool(neural[1]),
            "conventional_gpu_ms": conventional[0],
            "conventional_wall_ms": conventional_wall,
            "conventional_correct": bool(conventional[1]),
            "wait_then_recover_wall_ms": neural_wall + conventional_wall,
        })

    neural_wall = [x["neural_wall_ms"] for x in records]
    conventional_wall = [x["conventional_wall_ms"] for x in records]
    sequential_wall = [x["wait_then_recover_wall_ms"] for x in records]
    result = {
        "schema": "softwall-c163-local-4p5ms-diagnostic-v1",
        "analysis_role": (
            "Optimistic isolated-clean local-path diagnostic against the vendor "
            "testMAC threshold; not a production qualification or WCET result."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "deadline_ms": args.deadline_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "engine": args.engine,
        "conditions": [
            "one local process",
            "clean synthetic PUSCH",
            "no MPS co-tenant",
            "no Qwen",
            "no IPC or controller RPC"
        ],
        "neural_gpu_ms": stats([x["neural_gpu_ms"] for x in records]),
        "neural_wall_ms": stats(neural_wall),
        "conventional_gpu_ms": stats([x["conventional_gpu_ms"] for x in records]),
        "conventional_wall_ms": stats(conventional_wall),
        "wait_then_recover_wall_ms": stats(sequential_wall),
        "deadline_counts": {
            "neural_wall_le_deadline": sum(x <= args.deadline_ms for x in neural_wall),
            "conventional_wall_le_deadline": sum(x <= args.deadline_ms for x in conventional_wall),
            "wait_then_recover_wall_le_deadline": sum(x <= args.deadline_ms for x in sequential_wall),
        },
        "correctness": {
            "neural_correct": sum(x["neural_correct"] for x in records),
            "conventional_correct": sum(x["conventional_correct"] for x in records),
        },
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: result[k] for k in (
        "deadline_ms", "neural_wall_ms", "conventional_wall_ms",
        "wait_then_recover_wall_ms", "deadline_counts", "correctness"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
