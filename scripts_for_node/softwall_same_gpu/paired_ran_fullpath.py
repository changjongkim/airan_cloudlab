#!/usr/bin/env python3
"""Open-loop timing of a valid, payload-checked paired PUSCH Tx/Rx slot."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from paired_phy import PairedConventionalRan
from softwall_phy import summary, wait_until


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
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--period-ms", type=float, default=50.0)
    parser.add_argument("--deadline-ms", type=float, default=35.0)
    parser.add_argument("--seed", type=int, default=20260920)
    parser.add_argument("--ready-file")
    parser.add_argument("--start-file")
    args = parser.parse_args()

    ran = PairedConventionalRan(seed=args.seed)
    for _ in range(args.warmup):
        _, correct, _, _ = ran.run_cells(args.cells)
        if not correct:
            raise RuntimeError("paired-radio warmup produced an invalid transport block")
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
        gpu_ms, correct, crc_failures, payload_mismatches = ran.run_cells(args.cells)
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
            "correct_tb": correct,
            "crc_failures": crc_failures,
            "payload_mismatches": payload_mismatches,
        })
    result = {
        "schema": "softwall-paired-ran-v1",
        "label": args.label,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "cells": args.cells,
        "iterations": args.iterations,
        "period_ms": args.period_ms,
        "deadline_ms": args.deadline_ms,
        "input": "Aerial PdschTx-generated valid 273-PRB PUSCH slot",
        "path": "CE+NI+EQ+de-rate-match+LDPC+CRC+payload verification",
        "visible_sm_count": int(
            cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
        ),
        "gpu_ms": summary([x["gpu_ms"] for x in records]),
        "service_ms": summary([x["service_ms"] for x in records]),
        "response_ms": summary([x["response_ms"] for x in records]),
        "deadline_misses": sum(x["deadline_miss"] for x in records),
        "correct_transport_blocks": sum(x["correct_tb"] for x in records),
        "crc_failures": sum(x["crc_failures"] for x in records),
        "payload_mismatches": sum(x["payload_mismatches"] for x in records),
        "records": records,
    }
    atomic_json(Path(args.output), result)
    print(
        f"[PAIRED-RAN] {args.label} n={args.iterations} "
        f"p99={result['response_ms']['p99']:.3f}ms "
        f"max={result['response_ms']['max']:.3f}ms "
        f"miss={result['deadline_misses']} correct={result['correct_transport_blocks']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
