#!/usr/bin/env python3
"""Validate V16 four-point fault qualification and envelope classification."""

import argparse
import collections
import hashlib
import json
from pathlib import Path

from build_softwall_envelope_v16 import V15_ID, V16_ID


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(grid_path, prediction_path, v15_validation_path,
             v15_node_paths, abort_node_paths, model_path):
    prediction = load(prediction_path)
    v15_validation = load(v15_validation_path)
    v15_nodes = [load(path) for path in v15_node_paths]
    abort_nodes = [load(path) for path in abort_node_paths]
    model = load(model_path)
    rows = {row["id"]: row for row in prediction["results"]}
    v15 = rows[V15_ID]
    v16 = rows[V16_ID]
    ai45 = next(row for row in v16["ai_classes"]
                if row["bound_ms"] == 45.0)

    operation_arms = []
    for node in v15_nodes:
        for path in node["fault_results"]:
            arm = load(path)
            operation_arms.append((arm["crash_operation"], node["nodes"][0], arm))
    for node in abort_nodes:
        arm = load(node["arm_result"])
        operation_arms.append((arm["crash_operation"], node["nodes"][0], arm))
    operation_counts = collections.Counter(op for op, _, _ in operation_arms)
    operation_nodes = {
        op: sorted(node for item_op, node, _ in operation_arms if item_op == op)
        for op in ("prepare", "abort", "commit", "complete")
    }
    all_nodes = sorted({node for _, node, _ in operation_arms})
    gates = {
        "prior_v15_validation_pass": v15_validation["all_pass"],
        "v15_three_point_mode_demoted": (
            v15["status"] == "UQ"
            and v15["four_point_control_fault_required"]
            and not v15["four_point_control_fault_qualified"]
        ),
        "v16_four_point_mode_qsu": (
            v16["status"] == "QSU"
            and v16["four_point_control_fault_required"]
            and v16["four_point_control_fault_qualified"]
        ),
        "only_commit_is_ran_critical": (
            v16["control_rpc_count"] == 1
            and v16["control_critical_operations"] == ["commit"]
            and sorted(v16["control_deferred_operations"])
                == ["abort", "complete", "prepare"]
        ),
        "ai45_exchange_geometry_preserved": (
            ai45["effective_transaction_bound_ms"] == 54.0
            and not ai45["static_admissible"]
            and ai45["exchange_only_candidate"]
        ),
        "four_operations_two_arms_each": operation_counts == {
            "prepare": 2, "abort": 2, "commit": 2, "complete": 2,
        },
        "four_distinct_physical_nodes": len(all_nodes) == 4,
        "all_eight_fault_arms_pass": len(operation_arms) == 8 and all(
            arm["all_pass"] for _, _, arm in operation_arms
        ),
        "abort_target_never_executed": all(
            arm["gates"]["target_execution_semantics"]
            for op, _, arm in operation_arms if op == "abort"
        ),
        "radio_continues_after_every_fault": all(
            all(value > 0 for value in arm["radio_records_after_fault_detection"])
            for _, _, arm in operation_arms
        ),
        "single_token_invariant_all_nodes": all(
            node["totals"]["maximum_unlaunched_tokens"] == 1
            for node in v15_nodes + abort_nodes
        ),
        "retained_token_branch_all_nodes": all(
            node["totals"]["suppressed_prepare_due_owned_token"] > 0
            for node in v15_nodes + abort_nodes
        ),
        "finite_model_all_four_operations": (
            model["all_pass"]
            and {row["transition"].split("_")[0]
                 for row in model["fault_protocol_model"]["transition_values"]}
                >= {"prepare", "abort", "commit", "complete"}
            and not model["fault_protocol_model"]["violations"]
        ),
        "all_node_source_hashes_match": all(
            not node["source_hash_mismatches"]
            for node in v15_nodes + abort_nodes
        ),
    }
    artifacts = [grid_path, prediction_path, v15_validation_path,
                 model_path] + list(v15_node_paths) + list(abort_node_paths)
    return {
        "schema": "softwall-envelope-v16-validation-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "classification_counts": prediction["counts"],
        "physical_fault_matrix": {
            "operation_counts": dict(operation_counts),
            "operation_nodes": operation_nodes,
            "distinct_nodes": all_nodes,
            "fault_arms": len(operation_arms),
            "radio_records": sum(
                node["totals"]["radio_records"]
                for node in v15_nodes + abort_nodes
            ),
            "radio_after_fault": sum(
                node["totals"]["radio_after_fault"]
                for node in v15_nodes + abort_nodes
            ),
            "ai45_exchanges": sum(
                node["totals"]["ai45_exchanges"] for node in v15_nodes
            ),
            "suppressed_prepares": sum(
                node["totals"]["suppressed_prepare_due_owned_token"]
                for node in v15_nodes + abort_nodes
            ),
            "maximum_unlaunched_tokens": max(
                node["totals"]["maximum_unlaunched_tokens"]
                for node in v15_nodes + abort_nodes
            ),
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "artifact_sha256": {
            str(Path(path).resolve()): sha256(path) for path in artifacts
        },
        "claim_boundary": (
            "Finite-sample A100 qualification: each of prepare, abort, commit, "
            "and complete has two independent-node post-apply fail-stop arms. "
            "Operations were not injected as a Cartesian product on every node; "
            "production d_MAC, WCET, and cross-family claims remain out of scope."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--v15-validation", required=True)
    parser.add_argument("--v15-nodes", nargs=2, required=True)
    parser.add_argument("--abort-nodes", nargs=2, required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(
        args.grid, args.prediction, args.v15_validation, args.v15_nodes,
        args.abort_nodes, args.model,
    )
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V16 envelope validation failed")


if __name__ == "__main__":
    main()
