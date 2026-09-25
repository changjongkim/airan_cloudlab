#!/usr/bin/env python3
"""Aggregate prospective C133 prepare/complete fail-stop arms with C132 commit."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arms", type=Path, nargs="+", required=True)
    parser.add_argument("--commit-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    source_checks = {
        name: {
            "expected": expected,
            "observed": sha256(Path(name)),
            "match": expected == sha256(Path(name)),
        }
        for name, expected in protocol["source_sha256"].items()
    }
    commit = json.loads(args.commit_result.read_text())
    totals = {
        "radio_records": 0,
        "radio_records_after_crash": 0,
        "ai_units_before_containment": 0,
        "deadline_misses": 0,
        "nrx_bound_violations": 0,
        "conv_bound_violations": 0,
        "ai_bound_violations": 0,
        "ai_horizon_violations": 0,
    }
    arms = []
    operations = Counter()
    for path in args.arms:
        arm = json.loads(path.read_text())
        operation = arm["crash_operation"]
        operations[operation] += 1
        controllers = []
        request_ids = []
        for home in arm["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            request_ids.extend(
                row["request_id"] for row in controller["background_records"]
            )
            totals["radio_records"] += len(controller["records"])
            totals["ai_units_before_containment"] += len(
                controller["background_records"]
            )
            totals["deadline_misses"] += controller["deadline_misses"]
            totals["nrx_bound_violations"] += controller["nrx_bound_violations"]
            totals["conv_bound_violations"] += (
                controller["conv_bound_violations"]
                + controller["conv_path_bound_violations"]
            )
            totals["ai_bound_violations"] += controller[
                "background_budget_violations"
            ]
            totals["ai_horizon_violations"] += controller[
                "background_horizon_violations"
            ]
            controllers.append({
                "home": home["home"],
                "radio_records": len(controller["records"]),
                "ai_units": len(controller["background_records"]),
                "global_queue_status": controller["global_queue_status"],
                "fallback_calendar_final": controller["fallback_calendar_final"],
                "global_broker_transaction_budget_ms": controller[
                    "global_broker_transaction_budget_ms"
                ],
                "admission_ai_guard_ms": controller["admission_ai_guard_ms"],
            })
        totals["radio_records_after_crash"] += sum(
            arm["radio_records_after_fault_detection"]
        )
        arms.append({
            "path": str(path),
            "operation": operation,
            "all_pass": arm["all_pass"],
            "gates": arm["gates"],
            "target_request_id": arm["target_request_id"],
            "actual_request_ids_unique": len(request_ids) == len(set(request_ids)),
            "controllers": controllers,
        })
    commit_hash = sha256(args.commit_result)
    gates = {
        "prospective_sources_match": all(
            row["match"] for row in source_checks.values()
        ),
        "four_independent_new_arms": len(arms) == 4,
        "two_prepare_and_two_complete_arms": operations == Counter({
            "prepare": 2, "complete": 2,
        }),
        "all_new_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "control_budget_accounted": all(
            home["global_broker_transaction_budget_ms"] == 15
            and home["admission_ai_guard_ms"] == 17
            for arm in arms for home in arm["controllers"]
        ),
        "all_clients_fail_closed": all(
            not home["global_queue_status"]["enabled"]
            for arm in arms for home in arm["controllers"]
        ),
        "no_duplicate_execution": all(
            arm["actual_request_ids_unique"] for arm in arms
        ),
        "all_safety_gates_zero": all(
            totals[key] == 0 for key in (
                "deadline_misses", "nrx_bound_violations",
                "conv_bound_violations", "ai_bound_violations",
                "ai_horizon_violations",
            )
        ),
        "frozen_c132_commit_evidence_matches": (
            commit_hash == protocol["c132_commit_result_sha256"]
            and commit["all_pass"]
            and len(commit["arms"]) == 2
        ),
    }
    combined = {
        "radio_records": totals["radio_records"] + commit["totals"]["radio_records"],
        "radio_records_after_crash": (
            totals["radio_records_after_crash"]
            + commit["totals"]["radio_records_after_crash"]
        ),
        "fault_point_arms": {
            "prepare": operations["prepare"],
            "commit": len(commit["arms"]),
            "complete": operations["complete"],
        },
    }
    value = {
        "schema": "softwall-confirm133-control-point-fault-matrix-v1",
        "protocol": str(args.protocol),
        "source_checks": source_checks,
        "new_arms": arms,
        "new_arm_totals": totals,
        "c132_commit_result": str(args.commit_result),
        "combined_c132_c133": combined,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim": (
            "With every prepare, commit, and complete RPC bounded at 5 ms and "
            "the full 15 ms control cost charged before AI admission, two "
            "prospective arms at each post-apply fault point preserve disjoint "
            "local RAN certificates while all global-AI clients fail closed."
        ),
        "scope": (
            "Volatile broker fail-stop at one post-apply control point; no "
            "restart, durable token recovery, exactly-once, partition, GPU hang, "
            "production WCET, or throughput-superiority claim."
        ),
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(args.output)
    print(json.dumps({"new_arm_totals": totals, "combined": combined,
                      "gates": gates}, indent=2))
    if not value["all_pass"]:
        raise SystemExit("C133 control-point campaign gate failed")


if __name__ == "__main__":
    main()
