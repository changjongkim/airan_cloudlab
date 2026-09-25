#!/usr/bin/env python3
"""Test a production-oriented speculative NeuralRx/conventional fast path."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import platform
import threading
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
    parser.add_argument("--seed", type=int, default=20357900)
    parser.add_argument("--deadline-ms", type=float, default=4.5)
    args = parser.parse_args()

    neural_receiver = PairedDualReceiver(args.engine, seed=args.seed)
    conventional_receiver = PairedDualReceiver(args.engine, seed=args.seed)
    if not np.array_equal(
        neural_receiver.reference_tb, conventional_receiver.reference_tb
    ):
        raise RuntimeError("receiver transport blocks differ")
    if not np.array_equal(
        np.asarray(neural_receiver.rx_slot), np.asarray(conventional_receiver.rx_slot)
    ):
        raise RuntimeError("receiver input grids differ")
    for _ in range(args.warmup):
        neural_receiver.run_neural()
        conventional_receiver.run_conventional()

    records = []
    with ThreadPoolExecutor(max_workers=2) as pool:
        for iteration in range(args.iterations):
            barrier = threading.Barrier(3)

            def neural_task():
                barrier.wait()
                begin = time.perf_counter_ns()
                result = neural_receiver.run_neural()
                return result, (time.perf_counter_ns() - begin) / 1e6

            def conventional_task():
                barrier.wait()
                begin = time.perf_counter_ns()
                result = conventional_receiver.run_conventional()
                return result, (time.perf_counter_ns() - begin) / 1e6

            neural_future = pool.submit(neural_task)
            conventional_future = pool.submit(conventional_task)
            barrier.wait()
            release = time.perf_counter_ns()
            neural, neural_wall = neural_future.result()
            conventional, conventional_wall = conventional_future.result()
            pair_wall = (time.perf_counter_ns() - release) / 1e6
            records.append({
                "iteration": iteration,
                "neural_gpu_ms": neural[0],
                "neural_thread_wall_ms": neural_wall,
                "neural_correct": bool(neural[1]),
                "conventional_gpu_ms": conventional[0],
                "conventional_thread_wall_ms": conventional_wall,
                "conventional_correct": bool(conventional[1]),
                "parallel_pair_wall_ms": pair_wall,
            })

    pair_wall = [x["parallel_pair_wall_ms"] for x in records]
    result = {
        "schema": "softwall-c163-speculative-parallel-diagnostic-v1",
        "analysis_role": (
            "Optimistic same-process speculative dual-receiver diagnostic against "
            "the vendor testMAC threshold; not a production qualification or WCET."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "deadline_ms": args.deadline_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "seed": args.seed,
        "conditions": [
            "two independent receiver objects and CUDA streams in one process",
            "clean synthetic PUSCH",
            "no MPS co-tenant",
            "no Qwen",
            "no IPC or controller RPC"
        ],
        "neural_thread_wall_ms": stats([x["neural_thread_wall_ms"] for x in records]),
        "conventional_thread_wall_ms": stats([
            x["conventional_thread_wall_ms"] for x in records
        ]),
        "parallel_pair_wall_ms": stats(pair_wall),
        "deadline_counts": {
            "parallel_pair_wall_le_deadline": sum(x <= args.deadline_ms for x in pair_wall),
            "parallel_pair_wall_gt_deadline": sum(x > args.deadline_ms for x in pair_wall),
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
        "deadline_ms", "neural_thread_wall_ms", "conventional_thread_wall_ms",
        "parallel_pair_wall_ms", "deadline_counts", "correctness"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
