#!/usr/bin/env python3
"""Apply the frozen C165 correctness and 4.5-ms decision rule."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    return ordered[round((len(ordered) - 1) * fraction)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--role", choices=("ablation", "development", "holdout"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    worker = json.loads(args.worker.read_text(encoding="utf-8"))
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    deadline = float(protocol["deadline_ms"])
    pair = [float(row["parallel_pair_wall_ms"]) for row in controller["records"]]
    remote = [float(row["neural_gpu_ms"]) for row in worker["records"]]
    iterations = int(controller["iterations"])
    timely = sum(value <= deadline for value in pair)
    correct = int(controller["correctness"]["neural_correct"])
    conventional_correct = int(controller["correctness"]["conventional_correct"])
    eligible = args.role in ("development", "holdout") and iterations == 1000
    all_pass = bool(
        eligible
        and len(pair) == iterations
        and len(remote) == iterations + int(controller["warmup"])
        and timely == iterations
        and correct == iterations
        and conventional_correct == iterations
        and worker["error"] is None
        and worker["sequences_contiguous"]
    )
    result = {
        "schema": "softwall-c165-ce-tail-analysis-v1",
        "analysis_role": args.role,
        "protocol": str(args.protocol),
        "inputs": {
            "controller": str(args.controller),
            "worker": str(args.worker),
            "controller_sha256": sha256(args.controller),
            "worker_sha256": sha256(args.worker),
        },
        "input_mode": worker["input_mode"],
        "slurm_job_id": controller["slurm_job_id"],
        "deadline_ms": deadline,
        "iterations": iterations,
        "timely": timely,
        "late": iterations - timely,
        "neural_correct": correct,
        "conventional_correct": conventional_correct,
        "pair_wall_ms": {
            "p50": percentile(pair, 0.50),
            "p99": percentile(pair, 0.99),
            "max": max(pair, default=0.0),
        },
        "remote_neural_gpu_ms": {
            "p50": percentile(remote, 0.50),
            "p99": percentile(remote, 0.99),
            "max": max(remote, default=0.0),
        },
        "qualification_eligible": eligible,
        "all_pass": all_pass,
        "decision": (
            "opens independent holdout"
            if args.role == "development" and all_pass
            else "qualified holdout"
            if args.role == "holdout" and all_pass
            else "diagnostic only"
            if args.role == "ablation"
            else "rejected"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2), encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
