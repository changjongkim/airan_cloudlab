#!/usr/bin/env python3.11
"""Summarize the diagnostic C163 same-stream stage profile without NumPy."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def percentile(values: list[float], q: float) -> float:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        raise ValueError("empty distribution")
    position = (len(ordered) - 1) * q / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def distribution(values: list[float]) -> dict:
    return {
        "n": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(values, 50),
        "p95": percentile(values, 95),
        "p99": percentile(values, 99),
        "max": max(values),
    }


def pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    numerator = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = sum((x - mx) ** 2 for x in xs)
    dy = sum((y - my) ** 2 for y in ys)
    if dx == 0 or dy == 0:
        return None
    return numerator / math.sqrt(dx * dy)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    worker = json.loads(args.worker.read_text(encoding="utf-8"))
    if worker.get("error") is not None or not worker.get("sequences_contiguous"):
        raise RuntimeError("worker did not complete a contiguous valid run")
    warmup = int(controller["warmup"])
    timed_worker = [row for row in worker["records"] if row["sequence"] > warmup]
    if len(timed_worker) != controller["iterations"]:
        raise RuntimeError("timed worker/controller count mismatch")
    controller_by_sequence = {
        int(row["sequence"]): row for row in controller["records"]
    }
    if set(controller_by_sequence) != {int(row["sequence"]) for row in timed_worker}:
        raise RuntimeError("timed worker/controller sequence mismatch")

    stage_names = list(timed_worker[0]["stage_profile"]["stages_gpu_ms"])
    host_names = list(timed_worker[0]["stage_profile"]["host_enqueue_us"])
    worker_total = [float(row["stage_profile"]["total_gpu_ms"]) for row in timed_worker]
    pair_wall = [
        float(controller_by_sequence[int(row["sequence"])]["parallel_pair_wall_ms"])
        for row in timed_worker
    ]
    deadline = float(controller["deadline_ms"])
    late = [value > deadline for value in pair_wall]

    stages = {}
    for name in stage_names:
        values = [float(row["stage_profile"]["stages_gpu_ms"][name]) for row in timed_worker]
        late_values = [value for value, is_late in zip(values, late) if is_late]
        timely_values = [value for value, is_late in zip(values, late) if not is_late]
        stages[name] = {
            "gpu_ms": distribution(values),
            "pearson_with_worker_total": pearson(values, worker_total),
            "pearson_with_pair_wall": pearson(values, pair_wall),
            "late_mean_gpu_ms": sum(late_values) / len(late_values) if late_values else None,
            "timely_mean_gpu_ms": sum(timely_values) / len(timely_values) if timely_values else None,
        }

    result = {
        "schema": "softwall-c163-stage-profile-analysis-v1",
        "analysis_role": "Mechanism diagnosis only; profiling events alter timing and do not qualify the fast path.",
        "inputs": {
            "controller": str(args.controller),
            "worker": str(args.worker),
        },
        "warmup_excluded": warmup,
        "timed_units": len(timed_worker),
        "correct_units": sum(bool(row["neural_correct"]) for row in timed_worker),
        "deadline_ms": deadline,
        "timely_units": sum(not value for value in late),
        "late_units": sum(late),
        "pair_wall_ms": distribution(pair_wall),
        "worker_total_gpu_ms": distribution(worker_total),
        "stages": stages,
        "host_enqueue_us": {
            name: distribution([
                float(row["stage_profile"]["host_enqueue_us"][name])
                for row in timed_worker
            ])
            for name in host_names
        },
        "copy": {
            "forward_gpu_us": distribution([float(row["forward_gpu_us"]) for row in timed_worker]),
            "forward_host_us": distribution([float(row["forward_host_us"]) for row in timed_worker]),
            "backward_gpu_us": distribution([float(row["backward_gpu_us"]) for row in timed_worker]),
            "backward_host_us": distribution([float(row["backward_host_us"]) for row in timed_worker]),
        },
        "interpretation": (
            "The stage with the strongest tail and pair-wall correlation localizes the "
            "remaining fast-path violation. TensorRT and P2P distributions separate "
            "model/copy service from cuPHY channel-estimation service."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
