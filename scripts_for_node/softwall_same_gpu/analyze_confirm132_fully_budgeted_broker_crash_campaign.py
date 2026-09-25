#!/usr/bin/env python3
"""Aggregate prospective C132 fully budgeted broker fail-stop arms."""

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
    for path in args.arms:
        arm = json.loads(path.read_text())
        controllers = []
        request_ids = []
        for home in arm["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            request_ids.extend(row["request_id"] for row in controller["background_records"])
            totals["radio_records"] += len(controller["records"])
            totals["ai_units_before_containment"] += len(controller["background_records"])
            totals["deadline_misses"] += controller["deadline_misses"]
            totals["nrx_bound_violations"] += controller["nrx_bound_violations"]
            totals["conv_bound_violations"] += (
                controller["conv_bound_violations"]
                + controller["conv_path_bound_violations"]
            )
            totals["ai_bound_violations"] += controller["background_budget_violations"]
            totals["ai_horizon_violations"] += controller["background_horizon_violations"]
            controllers.append({
                "home": home["home"],
                "radio_records": len(controller["records"]),
                "ai_units": len(controller["background_records"]),
                "global_queue_status": controller["global_queue_status"],
                "fallback_calendar_final": controller["fallback_calendar_final"],
                "global_broker_transaction_budget_ms": controller["global_broker_transaction_budget_ms"],
                "admission_ai_guard_ms": controller["admission_ai_guard_ms"],
            })
        totals["radio_records_after_crash"] += sum(
            arm["radio_records_after_fault_detection"]
        )
        arms.append({
            "path": str(path),
            "all_pass": arm["all_pass"],
            "gates": arm["gates"],
            "broker_exit_status": arm["broker_exit_status"],
            "fault_detected_ns": arm["fault_detected_ns"],
            "ambiguous_request_id": arm["ambiguous_request_id"],
            "actual_request_ids_unique": len(request_ids) == len(set(request_ids)),
            "controllers": controllers,
        })
    gates = {
        "prospective_sources_match": all(row["match"] for row in source_checks.values()),
        "two_independent_arms": len(arms) == 2,
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "expected_fail_stop": all(arm["broker_exit_status"] == 86 for arm in arms),
        "both_homes_continue_radio": all(
            arm["gates"]["both_homes_radio_after_detection"] for arm in arms
        ),
        "all_global_ai_clients_fail_closed": all(
            not home["global_queue_status"]["enabled"]
            for arm in arms for home in arm["controllers"]
        ),
        "no_duplicate_execution": all(
            arm["actual_request_ids_unique"] for arm in arms
        ),
        "control_budget_accounted_in_admission": all(
            arm["gates"]["control_budget_accounted"]
            and all(home["global_broker_transaction_budget_ms"] == 15
                    and home["admission_ai_guard_ms"] == 17
                    for home in arm["controllers"])
            for arm in arms
        ),
        "faults_record_rpc_budget": all(
            fault.get("rpc_timeout_ms") == 5
            for arm in arms for home in arm["controllers"]
            for fault in home["global_queue_status"]["faults"]
            if not fault["operation"].startswith("metadata_")
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
        "schema": "softwall-confirm132-fully-budgeted-broker-crash-campaign-v1",
        "protocol": str(args.protocol),
        "source_checks": source_checks,
        "arms": arms,
        "totals": totals,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim": (
            "With 5 ms per-RPC bounds charged as a 15 ms prepare+commit+complete admission budget, after a broker fail-stop "
            "immediately following one applied commit, "
            "all global-AI clients stop admission while both disjoint local RAN "
            "certificates continue without duplicate AI execution."
        ),
        "scope": (
            "Volatile broker fail-stop containment only; no broker restart, "
            "durable recovery, or exactly-once completion claim."
        ),
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(args.output)
    print(json.dumps({"totals": totals, "gates": gates}, indent=2))
    if not value["all_pass"]:
        raise SystemExit("C132 campaign gate failed")


if __name__ == "__main__":
    main()
