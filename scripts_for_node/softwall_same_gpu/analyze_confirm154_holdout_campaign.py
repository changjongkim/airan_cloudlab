#!/usr/bin/env python3.11
"""Aggregate four frozen C154/C155 branch results."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-results", type=Path, nargs=4, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    arms = [json.loads(path.read_text(encoding="utf-8")) for path in args.arm_results]
    branches = [row["branch"] for row in arms]
    jobs = {row["job_id"] for row in arms}
    nodes = {row["node"] for row in arms}
    totals = {
        key: sum(row["summary"][key] for row in arms)
        for key in (
            "submitted_debts", "accepted_debts", "rejected_debts",
            "success_outcomes", "qwen_units", "physical_recoveries",
            "correct_home_commits", "deadline_misses",
        )
    }
    gates = {
        "four_frozen_branches": branches == protocol["branch_order"],
        "all_arm_gates_pass": all(row["all_pass"] for row in arms),
        "single_allocation_node": len(jobs) == 1 and len(nodes) == 1,
        "node_exclusion": bool(nodes)
        and next(iter(nodes)) not in protocol.get("excluded_nodes", []),
        "branch_totals": totals == {
            "submitted_debts": 18,
            "accepted_debts": 16,
            "rejected_debts": 2,
            "success_outcomes": 6,
            "qwen_units": 2,
            "physical_recoveries": 10,
            "correct_home_commits": 10,
            "deadline_misses": 0,
        },
    }
    value = {
        "schema": "softwall-confirm154-integrated-holdout-campaign-v1",
        "protocol": str(args.protocol),
        "status": (
            "ONE_NODE_CONTROLLED_HOLDOUT_PASS_SECOND_NODE_REQUIRED"
            if all(gates.values()) else "CONTROLLED_HOLDOUT_FAIL"
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
        "job_ids": sorted(jobs),
        "nodes": sorted(nodes),
        "branches": branches,
        "totals": totals,
        "arm_results": [str(path) for path in args.arm_results],
        "artifact_sha256": {
            str(args.protocol): sha256(args.protocol),
            **{str(path): sha256(path) for path in args.arm_results},
        },
        "claim_scope": protocol["scope"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

