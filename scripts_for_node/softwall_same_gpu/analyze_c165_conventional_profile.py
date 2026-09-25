#!/usr/bin/env python3
"""Summarize C165 conventional stage profiling without qualifying it."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summary(values: list[float]) -> dict[str, float | int]:
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
        "max": max(values),
    }


def correlation(left: list[float], right: list[float]) -> float:
    left_mean = sum(left) / len(left)
    right_mean = sum(right) / len(right)
    numerator = sum(
        (x - left_mean) * (y - right_mean) for x, y in zip(left, right)
    )
    denominator = (
        sum((x - left_mean) ** 2 for x in left)
        * sum((y - right_mean) ** 2 for y in right)
    ) ** 0.5
    return numerator / denominator if denominator else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    worker = json.loads(args.worker.read_text(encoding="utf-8"))
    rows = controller["records"]
    pair = [float(row["parallel_pair_wall_ms"]) for row in rows]
    total = [float(row["conventional_gpu_ms"]) for row in rows]
    names = tuple(rows[0]["conventional_stage_profile"]["stages_gpu_ms"])
    stages = {}
    for name in names:
        gpu = [
            float(row["conventional_stage_profile"]["stages_gpu_ms"][name])
            for row in rows
        ]
        host = [
            float(row["conventional_stage_profile"]["host_enqueue_us"][name])
            for row in rows
        ]
        stages[name] = {
            "gpu_ms": summary(gpu),
            "host_enqueue_us": summary(host),
            "pearson_with_conventional_total": correlation(gpu, total),
            "pearson_with_pair_wall": correlation(gpu, pair),
        }
    result = {
        "schema": "softwall-c165-conventional-profile-analysis-v1",
        "analysis_role": "Post-H2 mechanism diagnosis; profiling changes timing and cannot qualify the 4.5-ms path.",
        "protocol": str(args.protocol),
        "inputs": {"controller": str(args.controller), "worker": str(args.worker)},
        "iterations": len(rows),
        "correctness": controller["correctness"],
        "pair_wall_ms": summary(pair),
        "conventional_total_gpu_ms": summary(total),
        "remote_neural_gpu_ms": summary(
            [float(row["remote_neural_gpu_ms"]) for row in rows]
        ),
        "stages": stages,
        "worker_error": worker["error"],
        "qualification": "diagnostic_only",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
