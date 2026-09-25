#!/usr/bin/env python3
"""Aggregate C139 separated socket-timeout/admission-bound arms."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


REQUIRED_OPERATIONS = ("prepare", "commit", "complete")
CONTAINER_SOURCE_MAP = {
    "/softwall/global_trace_fully_budgeted_fault_contained_controller.py":
        "scripts_for_node/softwall_same_gpu/global_trace_fully_budgeted_fault_contained_controller.py",
    "/softwall/global_trace_qualified_control_bound_controller.py":
        "scripts_for_node/softwall_same_gpu/global_trace_qualified_control_bound_controller.py",
    "/softwall/deadline_fault_contained_global_trace_client.py":
        "scripts_for_node/softwall_same_gpu/deadline_fault_contained_global_trace_client.py",
    "/softwall/instrumented_deadline_global_trace_client_v2.py":
        "scripts_for_node/softwall_same_gpu/instrumented_deadline_global_trace_client_v2.py",
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
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_rpc_records(arm_results, socket_timeout_ms, admission_bound_ms):
    counts = Counter()
    faulted_counts = Counter()
    maxima = {}
    faulted_maxima = {}
    records = []
    telemetry_on_every_home = True
    per_arm_target_fault = True
    for arm in arm_results:
        operation = arm.get("crash_operation")
        target = 0
        for home in arm.get("homes", []):
            telemetry = home.get("rpc_telemetry")
            telemetry_on_every_home = telemetry_on_every_home and bool(
                telemetry
                and telemetry.get("schema") == "softwall-broker-rpc-telemetry-v2"
            )
            for record in (telemetry or {}).get("records", []):
                records.append(record)
                name = record["operation"]
                elapsed = float(record["elapsed_ms"])
                counts[name] += 1
                maxima[name] = max(elapsed, maxima.get(name, elapsed))
                if record.get("faulted"):
                    faulted_counts[name] += 1
                    faulted_maxima[name] = max(
                        elapsed, faulted_maxima.get(name, elapsed)
                    )
                    if home.get("home") == 0 and name == operation:
                        target += 1
        per_arm_target_fault = per_arm_target_fault and target == 1

    maximum = max(maxima.values()) if maxima else None
    faulted_maximum = max(faulted_maxima.values()) if faulted_maxima else None
    socket_timeout_elapsed_exceeded = sum(
        bool(row.get("wall_timeout_exceeded")) for row in records
    )
    admission_bound_exceeded = sum(
        float(row["elapsed_ms"]) > admission_bound_ms for row in records
    )
    gates = {
        "telemetry_on_every_home": telemetry_on_every_home,
        "prepare_commit_complete_coverage": all(counts[name] > 0 for name in REQUIRED_OPERATIONS),
        "faulted_call_each_arm": per_arm_target_fault,
        "faulted_call_present_at_each_control_point": all(
            faulted_counts[name] > 0 for name in REQUIRED_OPERATIONS
        ),
        "socket_timeout_and_admission_bound_distinct": (
            0 < socket_timeout_ms < admission_bound_ms
        ),
        "no_admission_bound_exceedance": admission_bound_exceeded == 0,
        "maximum_within_admission_bound": (
            maximum is not None and maximum <= admission_bound_ms
        ),
    }
    return {
        "socket_timeout_ms": socket_timeout_ms,
        "declared_per_rpc_admission_bound_ms": admission_bound_ms,
        "records": len(records),
        "by_operation": dict(sorted(counts.items())),
        "faulted_by_operation": dict(sorted(faulted_counts.items())),
        "max_elapsed_ms_by_operation": dict(sorted(maxima.items())),
        "max_faulted_elapsed_ms_by_operation": dict(sorted(faulted_maxima.items())),
        "maximum_elapsed_ms": maximum,
        "maximum_faulted_elapsed_ms": faulted_maximum,
        "socket_timeout_elapsed_exceeded": socket_timeout_elapsed_exceeded,
        "admission_bound_exceeded": admission_bound_exceeded,
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def analyze(campaign_path, arm_paths):
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    project_root = campaign_path.parents[2]
    expected = {row["label"]: row for row in campaign["arms"]}
    arms = []
    arm_results = []
    nodes = set()
    artifact_mismatches = []
    arm_source_mismatches = []
    operation_counts = Counter()

    for arm_path in arm_paths:
        arm_path = Path(arm_path).resolve()
        result = json.loads(arm_path.read_text())
        arm_results.append(result)
        label = arm_path.name[:-len("_result.json")]
        expected_arm = expected.get(label)
        operation = result["crash_operation"]
        operation_counts[operation] += 1
        protocol_path = Path(result["protocol"])
        protocol = json.loads(protocol_path.read_text())
        home_nodes = set()
        for home in result["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            home_nodes.add(controller["host"])
        nodes.update(home_nodes)
        for name, digest in result["artifact_sha256"].items():
            path = Path(name)
            if not path.is_file() or sha256(path) != digest:
                artifact_mismatches.append(str(path))
        for container_path, digest in protocol["source_sha256"].items():
            relative = CONTAINER_SOURCE_MAP.get(container_path)
            if relative is None:
                arm_source_mismatches.append(container_path)
            else:
                path = project_root / relative
                if not path.is_file() or sha256(path) != digest:
                    arm_source_mismatches.append(relative)
        arms.append({
            "label": label,
            "operation": operation,
            "result": str(arm_path),
            "expected_arm": expected_arm,
            "configuration_match": bool(expected_arm) and (
                expected_arm["operation"] == operation
                and expected_arm["payload_seeds"] == protocol["payload_seeds"]
                and expected_arm["channel_seeds"] == protocol["channel_seeds"]
            ),
            "nodes": sorted(home_nodes),
            "all_pass": result["all_pass"],
            "radio_records": sum(home["summary"]["records"] for home in result["homes"]),
            "radio_records_after_fault_detection": sum(result["radio_records_after_fault_detection"]),
            "ai_units_before_containment": sum(
                home["summary"]["atomic_exchange"] for home in result["homes"]
            ),
            "control_evidence": result["control_evidence"],
        })

    source_mismatches = []
    for relative, digest in campaign["source_sha256"].items():
        path = project_root / relative
        if not path.is_file() or sha256(path) != digest:
            source_mismatches.append(relative)

    control = summarize_rpc_records(
        arm_results,
        float(campaign["global_broker_rpc_timeout_ms"]),
        float(campaign["global_broker_rpc_admission_bound_ms"]),
    )
    gates = {
        "six_expected_arms": len(arms) == len(expected) == 6
            and {arm["label"] for arm in arms} == set(expected),
        "two_arms_per_control_point": operation_counts == Counter({
            "prepare": 2, "commit": 2, "complete": 2,
        }),
        "all_arm_configurations_match": all(arm["configuration_match"] for arm in arms),
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "all_artifact_hashes_match": not artifact_mismatches,
        "all_arm_sources_match": not arm_source_mismatches,
        "prospective_sources_match": not source_mismatches,
        "single_new_physical_node": len(nodes) == 1
            and nodes.isdisjoint(set(campaign["excluded_nodes"])),
        "control_fault_telemetry": control["all_pass"],
    }
    return {
        "schema": "softwall-confirm139-qualified-control-bound-v1",
        "campaign_protocol": str(campaign_path),
        "campaign_protocol_sha256": sha256(campaign_path),
        "nodes": sorted(nodes),
        "excluded_nodes": campaign["excluded_nodes"],
        "arms": arms,
        "totals": {
            "arms": len(arms),
            "radio_records": sum(arm["radio_records"] for arm in arms),
            "radio_records_after_fault_detection": sum(
                arm["radio_records_after_fault_detection"] for arm in arms
            ),
            "ai_units_before_containment": sum(
                arm["ai_units_before_containment"] for arm in arms
            ),
        },
        "control_evidence": control,
        "artifact_hash_mismatches": artifact_mismatches,
        "arm_source_hash_mismatches": arm_source_mismatches,
        "source_hash_mismatches": source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Finite-sample wall-clock qualification of the actual prepare, commit, "
            "and complete broker fail-stop detection calls with a 5 ms socket timeout "
            "and distinct prospective 7 ms admission charge on one new A100 node. "
            "This is not WCET, production d_MAC, "
            "cross-family, scheduler-isolation, restart, or exactly-once evidence."
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
        raise SystemExit("C139 qualified control-bound gate failed")


if __name__ == "__main__":
    main()
