#!/usr/bin/env python3.11
"""Combine independent C162 development and reversed-order holdout runs."""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--holdout-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    results = [load(args.development), load(args.holdout)]
    protocols = [load(args.development_protocol), load(args.holdout_protocol)]
    case_counts = collections.Counter()
    for result in results:
        case_counts.update(result["case_counts"])
    counts = {key: sum(result["counts"][key] for result in results)
              for key in results[0]["counts"]}
    maxima = {key: max(result["maxima_ms"][key] for result in results)
              for key in results[0]["maxima_ms"]}
    gates = {
        "development_and_holdout_pass": all(result["all_pass"] for result in results),
        "two_independent_nodes": len({result["host"] for result in results}) == 2,
        "campaign_roles_fixed": [result["campaign"] for result in results]
            == ["development", "holdout"],
        "holdout_reverses_case_order": (
            protocols[0]["reverse_cases"] is False
            and protocols[1]["reverse_cases"] is True
        ),
        "independent_seeds": protocols[0]["seed_base"] != protocols[1]["seed_base"],
        "identical_frozen_sources": protocols[0]["source_sha256"]
            == protocols[1]["source_sha256"],
        "same_frozen_grid": protocols[0]["grid"]["sha256"]
            == protocols[1]["grid"]["sha256"],
        "expected_180_rounds": sum(result["iterations"] for result in results) == 180,
        "every_case_30_samples": all(value == 30 for value in case_counts.values()),
        "zero_deadline_miss": counts["deadline_misses"] == 0,
    }
    value = {
        "schema": "softwall-c162-boundary-two-node-v1",
        "status": "C162_TWO_NODE_BOUNDARY_PASS" if all(gates.values()) else "C162_TWO_NODE_BOUNDARY_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "nodes": [result["host"] for result in results],
        "jobs": [result["slurm_job_id"] for result in results],
        "total_rounds": sum(result["iterations"] for result in results),
        "case_counts": dict(case_counts), "counts": counts, "maxima_ms": maxima,
        "scope": "Finite-sample validation of the frozen warm P180/D155 qualified-node envelope. It is not WCET, cold/long-idle, arbitrary-node, or production d_MAC qualification.",
        "inputs": {
            "development": {"path": str(args.development), "sha256": sha256(args.development)},
            "holdout": {"path": str(args.holdout), "sha256": sha256(args.holdout)},
            "development_protocol": {"path": str(args.development_protocol), "sha256": sha256(args.development_protocol)},
            "holdout_protocol": {"path": str(args.holdout_protocol), "sha256": sha256(args.holdout_protocol)},
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
