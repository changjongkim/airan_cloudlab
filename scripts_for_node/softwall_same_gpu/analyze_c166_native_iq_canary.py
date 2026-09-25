#!/usr/bin/env python3
"""Analyze the diagnostic-only C166 native IQ bridge canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-iterations", type=int, required=True)
    args = parser.parse_args()
    controller = json.loads(args.controller.read_text())
    worker = json.loads(args.worker.read_text())
    n = args.expected_iterations
    warmup = int(controller["warmup"])
    worker_expected = n + warmup
    checks = {
        "controller_iterations": len(controller["records"]) == n,
        "worker_iterations_including_warmup": worker["completed_units"] == worker_expected,
        "controller_conventional_correct": sum(
            x["conventional_correct"] for x in controller["records"]
        ) == n,
        "controller_neural_correct": sum(
            x["neural_correct"] for x in controller["records"]
        ) == n,
        "worker_neural_correct_including_warmup": worker["correct_units"] == worker_expected,
        "worker_sequences_contiguous": worker["sequences_contiguous"],
        "worker_error_absent": worker["error"] is None,
        "native_mode": worker["input_mode"] == "native_ordered",
    }
    result = {
        "schema": "softwall-c166-native-iq-canary-v1",
        "analysis_role": (
            "Diagnostic parity canary for the C++/CUDA IQ assembly bridge; "
            "not N1 completion and not latency or production qualification."
        ),
        "iterations": n,
        "warmup": warmup,
        "worker_expected_units": worker_expected,
        "controller_conventional_mode": controller.get(
            "conventional_mode", "separable"
        ),
        "checks": checks,
        "all_pass": all(checks.values()),
        "controller_pair_wall_ms": controller["parallel_pair_wall_ms"],
        "controller_conventional_gpu_ms": controller["conventional_gpu_ms"],
        "controller_remote_neural_gpu_ms": controller["remote_neural_gpu_ms"],
        "controller_deadline_counts": controller["deadline_counts"],
        "worker_neural_gpu_ms": worker["neural_gpu_ms"],
        "source_sha256": {
            "controller_raw": digest(args.controller),
            "worker_raw": digest(args.worker),
        },
        "decision": "parity_canary_pass" if all(checks.values()) else "rejected",
        "qualification": "forbidden",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
