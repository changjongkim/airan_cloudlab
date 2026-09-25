#!/usr/bin/env python3
"""Check fail-closed admission behavior under injected controller lateness."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = json.loads(args.input.read_text(encoding="utf-8"))
    injected = [item for item in value["records"] if item["injected_start_delay"]]
    rejected = [item for item in value["records"] if item["admission_rejections"]]
    injected_rejected = [item for item in injected if item["admission_rejections"]]
    unsafe = [
        item for item in rejected
        if (
            not item["correct"]
            or item["deadline_miss"]
            or item["nrx_commits"] != 0
            or item["conv_commits"] != 1
            or item["nrx_branch_correct"] != [None]
        )
    ]
    gates = {
        "all_injected_rejected": len(injected_rejected) == len(injected),
        "all_rejections_safe": not unsafe,
        "duplicate_commits_zero": value["duplicate_commits"] == 0,
        "bound_violations_zero": (
            value["nrx_bound_violations"] + value["conv_bound_violations"]
        ) == 0,
        "ai_violations_zero": (
            value["background_budget_violations"]
            + value["background_release_crossings"]
        ) == 0,
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema": "softwall-s2-admission-fault-v1",
        "input": args.input.name,
        "iterations": value["iterations"],
        "injected_releases": len(injected),
        "injected_rejected": len(injected_rejected),
        "total_admission_rejections": len(rejected),
        "unsafe_rejection_indices": [item["index"] for item in unsafe],
        "deadline_misses": value["deadline_misses"],
        "correct_releases": value["correct_releases"],
        "background_units": value["background_units"],
        "gates": gates,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
