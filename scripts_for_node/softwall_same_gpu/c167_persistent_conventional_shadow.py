#!/usr/bin/env python3
"""Diagnostic parity shadow for Aerial's monolithic cuPHY PUSCH path."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from pathlib import Path

import numpy as np

from dual_receiver_phy import PairedDualReceiver


def summary(values: list[float]) -> dict:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "p50": float(np.percentile(array, 50)),
        "p99": float(np.percentile(array, 99)),
        "max": float(np.max(array)),
    }


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20359400)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iterations", type=int, default=300)
    args = parser.parse_args()
    receiver = PairedDualReceiver(
        args.engine,
        seed=args.seed,
        device=0,
        enable_local_neural=False,
        enable_native_conventional_shadow=True,
    )
    warmup = [receiver.run_native_conventional() for _ in range(args.warmup)]
    records = []
    for index in range(args.iterations):
        separable = receiver.run_conventional()
        monolithic = receiver.run_native_conventional()
        records.append({
            "iteration": index,
            "separable_gpu_ms": separable[0],
            "separable_correct": separable[1],
            "separable_crc_failures": separable[2],
            "separable_payload_mismatches": separable[3],
            "monolithic_gpu_ms": monolithic[0],
            "monolithic_correct": monolithic[1],
            "monolithic_crc_failures": monolithic[2],
            "monolithic_payload_mismatches": monolithic[3],
        })
    checks = {
        "warmup_correct": sum(row[1] for row in warmup) == args.warmup,
        "separable_correct": sum(row["separable_correct"] for row in records)
        == args.iterations,
        "monolithic_correct": sum(row["monolithic_correct"] for row in records)
        == args.iterations,
        "monolithic_crc_failures_zero": sum(
            row["monolithic_crc_failures"] for row in records
        ) == 0,
        "monolithic_payload_mismatches_zero": sum(
            row["monolithic_payload_mismatches"] for row in records
        ) == 0,
    }
    result = {
        "schema": "softwall-c167-persistent-conventional-shadow-v1",
        "analysis_role": (
            "Diagnostic parity shadow for the monolithic cuPHY PuschPipeline with "
            "readiness-time persistent HARQ buffers; phase setup still crosses the "
            "Python wrapper, so this is not N2 completion or timing qualification."
        ),
        "host": platform.node(),
        "seed": args.seed,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "checks": checks,
        "all_pass": all(checks.values()),
        "separable_gpu_ms": summary([row["separable_gpu_ms"] for row in records]),
        "monolithic_gpu_ms": summary([row["monolithic_gpu_ms"] for row in records]),
        "source_sha256": {
            "script": digest(Path(__file__)),
            "dual_receiver_phy": digest(Path(__file__).with_name("dual_receiver_phy.py")),
        },
        "qualification": "forbidden",
        "records": records,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: value for key, value in result.items() if key != "records"}, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
