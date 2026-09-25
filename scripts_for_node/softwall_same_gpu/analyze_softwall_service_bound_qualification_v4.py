#!/usr/bin/env python3
"""Aggregate the two-node corrected V13 control and AI35 qualification."""

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zero_failure_upper(count, alpha=0.05):
    return 1.0 - math.pow(alpha, 1.0 / count) if count > 0 else None


def aggregate(v3, c139, c140, c141, paths):
    first_nodes = set(c139.get("nodes", [])) | set(c140.get("nodes", []))
    second_nodes = set(c141.get("nodes", []))
    first_control = c139["control_evidence"]
    second_control = c141["control_evidence"]
    gates = {
        "v3_correction_chain_passes": v3.get("all_pass") is True,
        "c141_independent_requalification_passes": c141.get("all_pass") is True,
        "two_distinct_a100_nodes": (
            len(first_nodes) == len(second_nodes) == 1
            and first_nodes.isdisjoint(second_nodes)
        ),
        "both_control_campaigns_use_7ms": (
            first_control.get("declared_per_rpc_admission_bound_ms") == 7.0
            and second_control.get("declared_per_rpc_admission_bound_ms") == 7.0
        ),
        "no_7ms_control_exceedance": (
            first_control.get("admission_bound_exceeded") == 0
            and second_control.get("admission_bound_exceeded") == 0
        ),
        "same_ai35_counterfactual": (
            c140["model_counterfactual"].get("effective_transaction_bound_ms") == 58.0
            and c140["model_counterfactual"].get("static_margin_ms") == -5.0
            and c140["model_counterfactual"].get("conditional_margin_ms") == 0.0
            and c141["ai35_counterfactual"].get("effective_transaction_bound_ms") == 58.0
            and c141["ai35_counterfactual"].get("static_margin_ms") == -5.0
            and c141["ai35_counterfactual"].get("conditional_margin_ms") == 0.0
        ),
        "conditional_exchange_on_both_nodes": (
            c140["totals"].get("ai35_candidate_exchanges", 0) > 0
            and c141["totals"].get("ai35_candidate_exchanges", 0) > 0
        ),
    }
    control_arms = c139["totals"]["arms"] + c141["control_campaign"]["arms"]
    ai_arms = len(c140["arms"]) + c141["ai35_campaign"]["arms"]
    nodes = sorted(first_nodes | second_nodes)
    control_records = first_control["records"] + second_control["records"]
    control_max = max(first_control["maximum_elapsed_ms"], second_control["maximum_elapsed_ms"])
    control_radio = c139["totals"]["radio_records"] + c141["control_campaign"]["radio_records"]
    control_after_fault = (
        c139["totals"]["radio_records_after_fault_detection"]
        + c141["totals"]["radio_records_after_fault_detection"]
    )
    ai_radio = c140["totals"]["radio_records"] + c141["ai35_campaign"]["radio_records"]
    ai_exchanges = (
        c140["totals"]["ai35_candidate_exchanges"]
        + c141["ai35_campaign"]["ai35_candidate_exchanges"]
    )
    candidate_branches = (
        c140["totals"]["candidate_branches"]
        + c141["ai35_campaign"]["candidate_branches"]
    )
    return {
        "schema": "softwall-service-bound-qualification-v4",
        "status": "FINITE_SAMPLE_TWO_NODE_CORRECTED_CONTROL_PASS" if all(gates.values()) else "UNQUALIFIED",
        "inputs": {
            name: {"path": str(path), "sha256": sha256(path)}
            for name, path in paths.items()
        },
        "mode": {
            "hardware_family": "NVIDIA A100-SXM4-40GB",
            "nodes": nodes,
            "socket_timeout_ms": 5.0,
            "per_rpc_admission_bound_ms": 7.0,
            "control_transaction_bound_ms": 21.0,
            "raw_ai_bound_ms": 35.0,
            "ai_completion_guard_ms": 2.0,
            "effective_transaction_bound_ms": 58.0,
            "static_all_fail_slack_ms": 53.0,
            "conditional_decision_window_ms": 58.0,
        },
        "control_fault_evidence": {
            "nodes": len(nodes),
            "arms": control_arms,
            "radio_records": control_radio,
            "radio_records_after_fault_detection": control_after_fault,
            "attempted_rpc_records": control_records,
            "maximum_rpc_wall_ms": control_max,
            "admission_bound_exceedances": 0,
            "zero_failure_95_upper_if_iid": {
                "arm_process_lifecycle": zero_failure_upper(control_arms),
                "physical_node": zero_failure_upper(len(nodes)),
            },
        },
        "conditional_ai35_evidence": {
            "nodes": len(nodes),
            "arms": ai_arms,
            "radio_records": ai_radio,
            "candidate_branches": candidate_branches,
            "ai35_candidate_exchanges": ai_exchanges,
            "minimum_physical_guarded_horizon_margin_ms": min(
                c140["totals"]["minimum_physical_guarded_horizon_margin_ms"],
                c141["ai35_campaign"]["minimum_physical_guarded_horizon_margin_ms"],
            ),
            "zero_failure_95_upper_if_iid": {
                "arm_process_lifecycle": zero_failure_upper(ai_arms),
                "physical_node": zero_failure_upper(len(nodes)),
            },
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "hard_real_time_status": "UNPROVEN",
        "claim_boundary": (
            "The corrected V13 control-fault contract and exchange-only AI35 class "
            "were requalified on two A100 nodes with independent process lifecycles. "
            "This remains finite-sample evidence, not WCET, production d_MAC, "
            "cross-family, or throughput-superiority evidence."
        ),
    }


def main():
    root = Path(__file__).resolve().parents[2]
    result_dir = root / "results/softwall_multigpu"
    parser = argparse.ArgumentParser()
    parser.add_argument("--v3", type=Path, default=result_dir / "softwall_service_bound_qualification_v3.json")
    parser.add_argument("--c139", type=Path, default=result_dir / "confirm139_qualified_control_bound_result.json")
    parser.add_argument("--c140", type=Path, default=result_dir / "confirm140_v13_ai35_result.json")
    parser.add_argument("--c141", type=Path, default=result_dir / "confirm141_v13_independent_requalification_result.json")
    parser.add_argument("--output", type=Path, default=result_dir / "softwall_service_bound_qualification_v4.json")
    args = parser.parse_args()
    paths = {key: getattr(args, key).resolve() for key in ("v3", "c139", "c140", "c141")}
    docs = {key: json.loads(path.read_text()) for key, path in paths.items()}
    result = aggregate(docs["v3"], docs["c139"], docs["c140"], docs["c141"], paths)
    output = args.output.resolve(); temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"output": str(output), "status": result["status"]}, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
