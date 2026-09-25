#!/usr/bin/env python3
"""Combine C141 corrected-control and AI35 results on an independent node."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def combine(control, ai35, control_path, ai35_path, prior_nodes):
    control_nodes = set(control.get("nodes", []))
    ai_nodes = set(ai35.get("nodes", []))
    nodes = sorted(control_nodes | ai_nodes)
    gates = {
        "control_campaign_passes": control.get("all_pass") is True,
        "ai35_campaign_passes": ai35.get("all_pass") is True,
        "same_single_physical_node": len(nodes) == 1 and control_nodes == ai_nodes,
        "node_is_independent": set(nodes).isdisjoint(set(prior_nodes)),
        "six_control_arms": control.get("totals", {}).get("arms") == 6,
        "two_ai35_arms": len(ai35.get("arms", [])) == 2,
        "seven_ms_control_bound": (
            control.get("control_evidence", {}).get("declared_per_rpc_admission_bound_ms") == 7.0
            and control.get("control_evidence", {}).get("admission_bound_exceeded") == 0
        ),
        "corrected_ai35_class": (
            ai35.get("model_counterfactual", {}).get("effective_transaction_bound_ms") == 58.0
            and ai35.get("model_counterfactual", {}).get("static_margin_ms") == -5.0
            and ai35.get("model_counterfactual", {}).get("conditional_margin_ms") == 0.0
            and ai35.get("totals", {}).get("ai35_candidate_exchanges", 0) > 0
        ),
    }
    return {
        "schema": "softwall-confirm141-v13-independent-requalification-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "nodes": nodes,
        "prior_corrected_mode_nodes": sorted(prior_nodes),
        "inputs": {
            "control": str(control_path),
            "control_sha256": sha256(control_path),
            "ai35": str(ai35_path),
            "ai35_sha256": sha256(ai35_path),
        },
        "totals": {
            "arms": control.get("totals", {}).get("arms", 0) + len(ai35.get("arms", [])),
            "radio_records": (
                control.get("totals", {}).get("radio_records", 0)
                + ai35.get("totals", {}).get("radio_records", 0)
            ),
            "radio_records_after_fault_detection": control.get("totals", {}).get(
                "radio_records_after_fault_detection", 0
            ),
            "attempted_fault_campaign_rpcs": control.get("control_evidence", {}).get("records", 0),
            "ai35_candidate_exchanges": ai35.get("totals", {}).get("ai35_candidate_exchanges", 0),
        },
        "control_evidence": control.get("control_evidence"),
        "control_campaign": {
            "arms": control.get("totals", {}).get("arms", 0),
            "radio_records": control.get("totals", {}).get("radio_records", 0),
            "radio_records_after_fault_detection": control.get("totals", {}).get(
                "radio_records_after_fault_detection", 0
            ),
        },
        "ai35_campaign": {
            "arms": len(ai35.get("arms", [])),
            "radio_records": ai35.get("totals", {}).get("radio_records", 0),
            "candidate_branches": ai35.get("totals", {}).get("candidate_branches", 0),
            "ai35_candidate_exchanges": ai35.get("totals", {}).get("ai35_candidate_exchanges", 0),
            "minimum_physical_guarded_horizon_margin_ms": ai35.get("totals", {}).get(
                "minimum_physical_guarded_horizon_margin_ms"
            ),
        },
        "ai35_counterfactual": {
            key: ai35.get("model_counterfactual", {}).get(key)
            for key in (
                "static_all_fail_slack_ms", "effective_transaction_bound_ms",
                "conditional_decision_window_ms", "static_margin_ms", "conditional_margin_ms",
            )
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Independent-node finite-sample requalification of the corrected V13 "
            "7 ms control admission bound and AI35 conditional class within the A100 "
            "family. This is not WCET, production d_MAC, cross-family, or throughput "
            "superiority evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--control", required=True)
    parser.add_argument("--ai35", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--prior-node", action="append", default=["nid001252"])
    args = parser.parse_args()
    control_path = Path(args.control).resolve()
    ai35_path = Path(args.ai35).resolve()
    result = combine(
        json.loads(control_path.read_text()),
        json.loads(ai35_path.read_text()),
        control_path,
        ai35_path,
        args.prior_node,
    )
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("C141 independent V13 requalification failed")


if __name__ == "__main__":
    main()
