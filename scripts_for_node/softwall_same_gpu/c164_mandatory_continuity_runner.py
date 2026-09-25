#!/usr/bin/env python3.11
"""Run four-cell mandatory cuPHY while optional Qwen reloads externally."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver


def wait_until_ns(target_ns: int) -> None:
    while True:
        remaining_ns = target_ns - time.perf_counter_ns()
        if remaining_ns <= 0:
            return
        if remaining_ns > 2_000_000:
            time.sleep((remaining_ns - 1_000_000) / 1e9)


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "p50": None,
                "p99": None, "max": None}
    ordered = sorted(values)
    pick = lambda fraction: ordered[round((len(ordered) - 1) * fraction)]
    return {"count": len(values), "mean": sum(values) / len(values),
            "p50": pick(.50), "p99": pick(.99), "max": ordered[-1]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--stop-file", type=Path, required=True)
    parser.add_argument("--cells", type=int, default=4)
    parser.add_argument("--period-ms", type=float, default=180.0)
    parser.add_argument("--deadline-ms", type=float, default=155.0)
    parser.add_argument("--component-bound-ms", type=float, default=25.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--minimum-iterations", type=int, default=30)
    parser.add_argument("--maximum-iterations", type=int, default=5000)
    parser.add_argument("--release-lead-ms", type=float, default=2000.0)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    if min(args.cells, args.period_ms, args.deadline_ms,
           args.component_bound_ms, args.warmup, args.minimum_iterations,
           args.maximum_iterations, args.release_lead_ms) <= 0:
        parser.error("all count and timing arguments must be positive")
    if args.minimum_iterations > args.maximum_iterations:
        parser.error("minimum iterations exceed maximum")

    cp.cuda.runtime.setDevice(0)
    receiver = PairedDualReceiver(
        args.engine, seed=args.seed, device=0, enable_local_neural=False
    )
    warmup_started_ns = time.perf_counter_ns()
    for _ in range(args.warmup):
        for _ in range(args.cells):
            if not receiver.run_conventional()[1]:
                raise RuntimeError("mandatory conventional warmup failed")
    warmup_completed_ns = time.perf_counter_ns()
    first_release_ns = (
        time.perf_counter_ns() + round(args.release_lead_ms * 1e6)
    )
    atomic_json(args.ready_file, {
        "schema": "softwall-c164-mandatory-ready-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "clock": "time.perf_counter_ns",
        "warmup_started_ns": warmup_started_ns,
        "warmup_completed_ns": warmup_completed_ns,
        "first_release_ns": first_release_ns,
        "period_ns": round(args.period_ms * 1e6),
        "ready": True,
    })

    period_ns = round(args.period_ms * 1e6)
    records = []
    error = None
    try:
        for index in range(args.maximum_iterations):
            release_ns = first_release_ns + index * period_ns
            wait_until_ns(release_ns)
            started_ns = time.perf_counter_ns()
            cells = []
            for cell in range(args.cells):
                gpu_ms, correct, crc_failures, payload_mismatches = (
                    receiver.run_conventional()
                )
                cells.append({
                    "cell": cell, "gpu_ms": gpu_ms,
                    "correct": bool(correct),
                    "crc_failures": crc_failures,
                    "payload_mismatches": payload_mismatches,
                    "component_bound_violation": (
                        gpu_ms > args.component_bound_ms
                    ),
                })
            completed_ns = time.perf_counter_ns()
            response_ms = (completed_ns - release_ns) / 1e6
            records.append({
                "index": index, "release_ns": release_ns,
                "started_ns": started_ns, "completed_ns": completed_ns,
                "start_lateness_ms": (started_ns - release_ns) / 1e6,
                "response_ms": response_ms,
                "deadline_miss": response_ms > args.deadline_ms,
                "correct": all(row["correct"] for row in cells),
                "cell_results": cells,
            })
            if (len(records) >= args.minimum_iterations
                    and args.stop_file.exists()):
                break
        else:
            raise TimeoutError("mandatory continuity runner hit maximum iterations")
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        component_ms = [cell["gpu_ms"] for row in records
                        for cell in row["cell_results"]]
        response_ms = [row["response_ms"] for row in records]
        result = {
            "schema": "softwall-c164-mandatory-continuity-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "clock": "time.perf_counter_ns",
            "cells": args.cells, "period_ms": args.period_ms,
            "deadline_ms": args.deadline_ms,
            "component_bound_ms": args.component_bound_ms,
            "warmup": args.warmup, "payload_seed": args.seed,
            "first_release_ns": first_release_ns,
            "completed_iterations": len(records),
            "correct_releases": sum(row["correct"] for row in records),
            "deadline_misses": sum(row["deadline_miss"] for row in records),
            "component_bound_violations": sum(
                cell["component_bound_violation"]
                for row in records for cell in row["cell_results"]
            ),
            "response_ms": summarize(response_ms),
            "component_gpu_ms": summarize(component_ms),
            "mps_active_thread_percentage": os.environ.get(
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            ),
            "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
            "stop_observed": args.stop_file.exists(),
            "error": error, "records": records,
        }
        atomic_json(args.output, result)


if __name__ == "__main__":
    main()
