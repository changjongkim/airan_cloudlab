#!/usr/bin/env python3
"""Aggregate prospective C127 broker-fault containment arms."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arms", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    source_checks = {}
    for name, expected in protocol["source_sha256"].items():
        path = Path(name)
        observed = sha256(path)
        source_checks[name] = {
            "expected": expected,
            "observed": observed,
            "match": observed == expected,
        }

    arms = []
    totals = {
        "radio_records": 0,
        "deadline_misses": 0,
        "nrx_bound_violations": 0,
        "conv_bound_violations": 0,
        "ai_bound_violations": 0,
        "ai_horizon_violations": 0,
        "ambiguous_tokens": 0,
        "duplicate_commits": 0,
        "home0_radio_records_after_fault": 0,
        "home1_ai_units_after_fault": 0,
    }
    for path in args.arms:
        arm = json.loads(path.read_text(encoding="utf-8"))
        broker = json.loads(Path(arm["broker"]).read_text(encoding="utf-8"))
        fault_ns = broker["faults"][0]["injected_ns"]
        per_home = []
        for home in arm["homes"]:
            controller = json.loads(
                Path(home["controller"]).read_text(encoding="utf-8")
            )
            records_after_fault = sum(
                row["commit_return_ns"] > fault_ns
                for row in controller["records"]
            )
            ai_after_fault = sum(
                row["admitted_ns"] > fault_ns
                for row in controller["background_records"]
            )
            totals["radio_records"] += len(controller["records"])
            totals["deadline_misses"] += controller["deadline_misses"]
            totals["nrx_bound_violations"] += controller["nrx_bound_violations"]
            totals["conv_bound_violations"] += (
                controller["conv_bound_violations"]
                + controller["conv_path_bound_violations"]
            )
            totals["ai_bound_violations"] += controller["background_budget_violations"]
            totals["ai_horizon_violations"] += controller["background_horizon_violations"]
            if home["home"] == 0:
                totals["home0_radio_records_after_fault"] += records_after_fault
            else:
                totals["home1_ai_units_after_fault"] += ai_after_fault
            per_home.append({
                "home": home["home"],
                "radio_records": len(controller["records"]),
                "radio_records_after_fault": records_after_fault,
                "ai_units": len(controller["background_records"]),
                "ai_units_after_fault": ai_after_fault,
                "global_queue_status": controller["global_queue_status"],
                "fallback_calendar_final": controller["fallback_calendar_final"],
            })
        totals["ambiguous_tokens"] += broker["summary"]["outstanding_tokens"]
        totals["duplicate_commits"] += broker["summary"]["duplicate_commit_count"]
        arms.append({
            "path": str(path),
            "all_pass": arm["all_pass"],
            "gates": arm["gates"],
            "broker_faults": broker["faults"],
            "broker_summary": broker["summary"],
            "homes": per_home,
        })

    gates = {
        "prospective_sources_match": all(
            check["match"] for check in source_checks.values()
        ),
        "two_independent_arms": len(arms) == 2,
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "all_safety_gates_zero": all(
            totals[key] == 0 for key in (
                "deadline_misses", "nrx_bound_violations",
                "conv_bound_violations", "ai_bound_violations",
                "ai_horizon_violations", "duplicate_commits",
            )
        ),
        "one_quarantined_token_per_arm": (
            totals["ambiguous_tokens"] == len(arms)
        ),
        "faulted_home_radio_continues": (
            totals["home0_radio_records_after_fault"] > 0
            and all(
                arm["homes"][0]["radio_records_after_fault"] > 0
                for arm in arms
            )
        ),
        "other_home_ai_continues": (
            totals["home1_ai_units_after_fault"] > 0
            and all(
                arm["homes"][1]["ai_units_after_fault"] > 0
                for arm in arms
            )
        ),
    }
    result = {
        "schema": "softwall-confirm127-broker-fault-campaign-v1",
        "protocol": str(args.protocol),
        "source_checks": source_checks,
        "arms": arms,
        "totals": totals,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim": (
            "An applied global-AI commit whose reply is lost is quarantined "
            "without retry; local RAN recovery and the other home continue."
        ),
        "scope": (
            "Finite-sample two-GPU process/broker fault containment; this is "
            "not Byzantine fault tolerance or GPU/driver-hang recovery."
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps({"totals": totals, "gates": gates}, indent=2))
    if not result["all_pass"]:
        raise SystemExit("C127 campaign gate failed")


if __name__ == "__main__":
    main()
