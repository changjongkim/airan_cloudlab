#!/usr/bin/env python3
"""Validate one independent-node V16 abort post-apply fault arm."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(protocol_path, arm_path):
    protocol_path = Path(protocol_path).resolve()
    arm_path = Path(arm_path).resolve()
    protocol = json.loads(protocol_path.read_text())
    arm = json.loads(arm_path.read_text())
    root = protocol_path.parents[2]
    nodes = {
        json.loads(Path(home["controller"]).read_text())["host"]
        for home in arm["homes"]
    }
    source_mismatches = []
    for relative, digest in protocol["source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            source_mismatches.append(relative)
    ownership = [home["rpc_telemetry"]["ownership_evidence"]
                 for home in arm["homes"]]
    gates = {
        "one_expected_node": nodes == {protocol["allocation_node"]},
        "excluded_nodes_avoided": nodes.isdisjoint(
            set(protocol["excluded_nodes"])
        ),
        "abort_arm_pass": arm["all_pass"],
        "abort_fault_injected": arm["crash_operation"] == "abort"
            and arm["broker_exit_status"] == 86,
        "target_not_executed": arm["gates"]["target_execution_semantics"],
        "both_homes_radio_continue": all(
            value > 0 for value in arm["radio_records_after_fault_detection"]
        ),
        "all_four_operations_observed": arm["gates"][
            "rpc_all_four_operations_each_home"
        ],
        "single_unlaunched_token": bool(ownership) and all(
            item["maximum_unlaunched_tokens"] <= 1 for item in ownership
        ),
        "retained_token_branch_exercised": bool(ownership) and all(
            item["suppressed_prepare_due_owned_token"] > 0
            for item in ownership
        ),
        "no_source_mismatch": not source_mismatches,
    }
    return {
        "schema": "softwall-v16-abort-node-requalification-v1",
        "protocol": str(protocol_path),
        "protocol_sha256": sha256(protocol_path),
        "arm_result": str(arm_path),
        "arm_result_sha256": sha256(arm_path),
        "nodes": sorted(nodes),
        "totals": {
            "radio_records": sum(
                home["summary"]["records"] for home in arm["homes"]
            ),
            "radio_after_fault": sum(
                arm["radio_records_after_fault_detection"]
            ),
            "fault_arms": 1,
            "suppressed_prepare_due_owned_token": sum(
                item["suppressed_prepare_due_owned_token"]
                for item in ownership
            ),
            "maximum_unlaunched_tokens": max(
                item["maximum_unlaunched_tokens"] for item in ownership
            ),
        },
        "source_hash_mismatches": source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "One A100-node finite-sample post-apply abort reply-loss arm. "
            "It extends the V15 two-node evidence; it is not WCET, production "
            "d_MAC, or cross-GPU-family evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--arm", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.protocol, args.arm)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V16 abort node gate failed")


if __name__ == "__main__":
    main()
