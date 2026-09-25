#!/usr/bin/env python3
"""Aggregate the independently allocated C134 three-point fail-stop campaign."""

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
    expected = {row["label"]: row for row in protocol["arms"]}
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
    operations = Counter()
    arms = []
    for path in args.arms:
        arm = json.loads(path.read_text())
        label = path.name.removesuffix("_result.json")
        row = expected.get(label)
        operation = arm["crash_operation"]
        operations[operation] += 1
        artifact_checks = {}
        for name, digest in arm["artifact_sha256"].items():
            observed = sha256(Path(name))
            artifact_checks[name] = {
                "expected": digest,
                "observed": observed,
                "match": digest == observed,
            }
        arm_protocol = json.loads(Path(arm["protocol"]).read_text())
        arm_sources_match = all(
            sha256(Path({
                "/softwall/global_trace_fully_budgeted_fault_contained_controller.py":
                    "scripts_for_node/softwall_same_gpu/global_trace_fully_budgeted_fault_contained_controller.py",
                "/softwall/deadline_fault_contained_global_trace_client.py":
                    "scripts_for_node/softwall_same_gpu/deadline_fault_contained_global_trace_client.py",
                "/softwall/control_point_crash_global_trace_lease_broker.py":
                    "scripts_for_node/softwall_same_gpu/control_point_crash_global_trace_lease_broker.py",
                "/softwall/same_request_ipc_worker.py":
                    "scripts_for_node/softwall_same_gpu/same_request_ipc_worker.py",
                "/softwall/trace_qwen_worker.py":
                    "scripts_for_node/softwall_same_gpu/trace_qwen_worker.py",
                "/softwall_task1/isca_v2/dart_runtime.py":
                    "scripts_for_node/task1/isca_v2/dart_runtime.py",
                "/softwall_task1/isca_v2/cuda_ipc_channel.py":
                    "scripts_for_node/task1/isca_v2/cuda_ipc_channel.py",
            }[name])) == digest
            for name, digest in arm_protocol["source_sha256"].items()
        )
        controllers = []
        request_ids = []
        for home in arm["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            request_ids.extend(
                record["request_id"] for record in controller["background_records"]
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
                "global_broker_transaction_budget_ms":
                    controller["global_broker_transaction_budget_ms"],
                "admission_ai_guard_ms": controller["admission_ai_guard_ms"],
            })
        totals["radio_records_after_crash"] += sum(
            arm["radio_records_after_fault_detection"]
        )
        arms.append({
            "path": str(path),
            "label": label,
            "operation": operation,
            "expected_arm": row,
            "all_pass": arm["all_pass"],
            "gates": arm["gates"],
            "target_request_id": arm["target_request_id"],
            "actual_request_ids_unique": len(request_ids) == len(set(request_ids)),
            "arm_artifacts_match": all(
                item["match"] for item in artifact_checks.values()
            ),
            "arm_sources_match": arm_sources_match,
            "controllers": controllers,
        })

    gates = {
        "prospective_sources_match": all(
            row["match"] for row in source_checks.values()
        ),
        "six_expected_arms": (
            len(arms) == 6
            and {arm["label"] for arm in arms} == set(expected)
            and all(arm["expected_arm"] is not None for arm in arms)
        ),
        "two_arms_per_control_point": operations == Counter({
            "prepare": 2, "commit": 2, "complete": 2,
        }),
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "all_arm_artifacts_match": all(
            arm["arm_artifacts_match"] for arm in arms
        ),
        "all_arm_sources_match": all(arm["arm_sources_match"] for arm in arms),
        "control_budget_accounted": all(
            home["global_broker_transaction_budget_ms"] == 15
            and home["admission_ai_guard_ms"] == 17
            for arm in arms for home in arm["controllers"]
        ),
        "all_clients_fail_closed": all(
            not home["global_queue_status"]["enabled"]
            for arm in arms for home in arm["controllers"]
        ),
        "all_credits_drained": all(
            home["fallback_calendar_final"]["outstanding"] == 0
            and home["fallback_calendar_final"]["joint_leases_outstanding"] == 0
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
    }
    value = {
        "schema": "softwall-confirm134-independent-node-control-matrix-v1",
        "protocol": str(args.protocol),
        "allocation": protocol["allocation"],
        "source_checks": source_checks,
        "arms": arms,
        "totals": totals,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim": (
            "On an independent allocation and A100 node, two prospective arms "
            "at each post-apply prepare, commit, and complete broker fail-stop "
            "point preserve disjoint local RAN certificates under the full "
            "15 ms admission-charged control budget."
        ),
        "scope": (
            "Same A100 hardware family and software stack; no cross-family "
            "generalization, restart, durable reconciliation, exactly-once, "
            "production WCET, or throughput-superiority claim."
        ),
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(args.output)
    print(json.dumps({"totals": totals, "gates": gates}, indent=2))
    if not value["all_pass"]:
        raise SystemExit("C134 independent-node control matrix failed")


if __name__ == "__main__":
    main()
