#!/usr/bin/env python3
"""Aggregate V14 pipelined prepare/commit/complete fail-stop arms."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(campaign_path, arm_paths):
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    root = campaign_path.parents[2]
    expected = {row["label"]: row for row in campaign["arms"]}
    arms = []
    nodes = set()
    operation_counts = Counter()
    all_records = []
    artifact_mismatches = []
    source_mismatches = []
    for arm_path in arm_paths:
        arm_path = Path(arm_path).resolve()
        result = json.loads(arm_path.read_text())
        label = arm_path.name[:-len("_result.json")]
        declared = expected.get(label)
        protocol = json.loads(Path(result["protocol"]).read_text())
        operation = result["crash_operation"]
        operation_counts[operation] += 1
        home_nodes = set()
        target_fault_records = 0
        for home in result["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            home_nodes.add(controller["host"])
            telemetry = home["rpc_telemetry"]
            all_records.extend(telemetry["records"])
            if home["home"] == 0:
                target_fault_records += sum(
                    row["operation"] == operation and row["faulted"]
                    for row in telemetry["records"]
                )
        nodes.update(home_nodes)
        for name, digest in result["artifact_sha256"].items():
            if not Path(name).is_file() or sha256(name) != digest:
                artifact_mismatches.append(name)
        for container, digest in protocol["source_sha256"].items():
            relative = campaign["container_source_map"].get(container)
            if relative is None or not (root / relative).is_file() \
                    or sha256(root / relative) != digest:
                source_mismatches.append({
                    "arm": str(arm_path), "source": container,
                })
        arms.append({
            "label": label,
            "operation": operation,
            "result": str(arm_path),
            "configuration_match": bool(declared) and (
                declared["operation"] == operation
                and declared["payload_seeds"] == protocol["payload_seeds"]
                and declared["channel_seeds"] == protocol["channel_seeds"]
            ),
            "nodes": sorted(home_nodes),
            "all_pass": result["all_pass"],
            "radio_records": sum(
                home["summary"]["records"] for home in result["homes"]
            ),
            "radio_after_fault": sum(result["radio_records_after_fault_detection"]),
            "target_fault_records": target_fault_records,
            "target_request_id": result["target_request_id"],
            "control_evidence": result["control_evidence"],
        })
    campaign_source_mismatches = []
    for relative, digest in campaign["source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            campaign_source_mismatches.append(relative)
    commit = [row for row in all_records if row["operation"] == "commit"]
    deferred = [row for row in all_records
                if row["operation"] in {"prepare", "abort", "complete"}]
    model_path = root / campaign["finite_model_artifact"]
    model = json.loads(model_path.read_text())
    gates = {
        "six_expected_arms": len(arms) == len(expected) == 6
            and {arm["label"] for arm in arms} == set(expected),
        "two_arms_per_control_point": operation_counts == Counter({
            "prepare": 2, "commit": 2, "complete": 2,
        }),
        "all_configurations_match": all(
            arm["configuration_match"] for arm in arms
        ),
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "one_new_physical_node": len(nodes) == 1
            and nodes.isdisjoint(set(campaign["excluded_nodes"])),
        "one_target_fault_record_each_arm": all(
            arm["target_fault_records"] == 1 for arm in arms
        ),
        "commit_path_within_bound": bool(commit) and all(
            row["elapsed_ms"] <= campaign["commit_bound_ms"] for row in commit
        ),
        "deferred_path_exceeds_commit_bound_observed": any(
            row["elapsed_ms"] > campaign["commit_bound_ms"] for row in deferred
        ),
        "finite_model_pass": model["all_pass"] and not model["violations"],
        "artifact_hashes_match": not artifact_mismatches,
        "arm_sources_match": not source_mismatches,
        "campaign_sources_match": not campaign_source_mismatches,
    }
    return {
        "schema": "softwall-confirm143-v14-pipelined-control-faults-v1",
        "campaign_protocol": str(campaign_path),
        "campaign_protocol_sha256": sha256(campaign_path),
        "nodes": sorted(nodes),
        "arms": arms,
        "totals": {
            "arms": len(arms),
            "radio_records": sum(arm["radio_records"] for arm in arms),
            "radio_records_after_fault": sum(
                arm["radio_after_fault"] for arm in arms
            ),
            "rpc_records": len(all_records),
            "commit_records": len(commit),
            "maximum_commit_ms": max(
                (row["elapsed_ms"] for row in commit), default=None
            ),
            "deferred_records": len(deferred),
            "maximum_deferred_ms": max(
                (row["elapsed_ms"] for row in deferred), default=None
            ),
            "deferred_over_7ms": sum(
                row["elapsed_ms"] > campaign["commit_bound_ms"]
                for row in deferred
            ),
        },
        "finite_model": {
            "path": str(model_path),
            "states": model["states"],
            "edges": model["edges"],
            "violations": len(model["violations"]),
        },
        "artifact_hash_mismatches": artifact_mismatches,
        "arm_source_hash_mismatches": source_mismatches,
        "campaign_source_hash_mismatches": campaign_source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Two finite-sample arms at each post-apply prepare, commit, and "
            "complete broker fail-stop point on one A100 node. The result checks "
            "local RAN continuity, at-most-once AI, quarantine, and a bounded "
            "launch commit; it is not a broker recovery or WCET proof."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.campaign, args.arms)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("C143 V14 control-fault gate failed")


if __name__ == "__main__":
    main()
