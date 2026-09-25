#!/usr/bin/env python3
"""Validate the V15 prediction against C145-C146 ownership evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from build_softwall_envelope_v15 import V14_ID, V15_ID


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(grid_path, prediction_path, c145_path, c146_path, model_path,
             regression_path, reproducibility_path):
    prediction = load(prediction_path)
    c145 = load(c145_path)
    c146 = load(c146_path)
    model = load(model_path)
    regression = load(regression_path)
    reproducibility = load(reproducibility_path)
    rows = {row["id"]: row for row in prediction["results"]}
    v14 = rows[V14_ID]
    v15 = rows[V15_ID]
    ai45 = next(row for row in v15["ai_classes"]
                if row["bound_ms"] == 45.0)
    node_results = (c145, c146)
    gates = {
        "v14_uncovered_mode_is_uq": v14["status"] == "UQ",
        "v15_mode_is_qsu": v15["status"] == "QSU",
        "single_token_ownership_qualified": (
            v15["single_token_ownership_required"]
            and v15["single_token_ownership_qualified"]
        ),
        "only_commit_is_critical": (
            v15["control_rpc_count"] == 1
            and v15["required_control_transaction_budget_ms"] == 7.0
            and v15["control_critical_operations"] == ["commit"]
        ),
        "ai45_predicted_exchange_only": (
            ai45["effective_transaction_bound_ms"] == 54.0
            and not ai45["static_admissible"]
            and ai45["exchange_only_candidate"]
        ),
        "two_new_nodes": set(c145["nodes"] + c146["nodes"])
            == {"nid001244", "nid003417"},
        "both_node_campaigns_pass": all(row["all_pass"] for row in node_results),
        "six_fault_arms": sum(row["totals"]["fault_arms"]
                              for row in node_results) == 6,
        "physical_ownership_branch_covered": all(
            row["totals"]["suppressed_prepare_due_owned_token"] > 0
            and row["totals"]["maximum_unlaunched_tokens"] == 1
            for row in node_results
        ),
        "regression_reproduces_and_fixes_v14": (
            regression["all_pass"]
            and regression["v14"]["untracked_held_tokens"] == 1
            and regression["v15"]["untracked_held_tokens"] == 0
        ),
        "composed_finite_model_pass": (
            model["all_pass"]
            and not model["fault_protocol_model"]["violations"]
            and not model["single_token_ownership_model"]["violations"]
        ),
        "model_semantically_reproducible": (
            reproducibility["all_pass"]
            and reproducibility["canonical_variants"] == 1
            and reproducibility["runs"] >= 16
        ),
        "v15_expected_counts": prediction["counts"] == {
            "QSU": 6, "QSN": 0, "MI": 3, "UQ": 12,
        },
    }
    paths = [grid_path, prediction_path, c145_path, c146_path, model_path,
             regression_path, reproducibility_path]
    return {
        "schema": "softwall-envelope-v15-validation-summary-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "inputs": {str(path): sha256(path) for path in paths},
        "model_counts": prediction["counts"],
        "v14_status": v14["status"],
        "v15_mode": v15,
        "c145": c145["totals"],
        "c146": c146["totals"],
        "finite_model": {
            "fault_states": model["fault_protocol_model"]["states"],
            "fault_edges": model["fault_protocol_model"]["edges"],
            "ownership_states": model["single_token_ownership_model"]["states"],
            "ownership_edges": model["single_token_ownership_model"]["edges"],
            "violations": (
                len(model["fault_protocol_model"]["violations"])
                + len(model["single_token_ownership_model"]["violations"])
            ),
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "V15 closes the finite-sample model/implementation loop for one-token "
            "ownership, pipelined global control, and exchange-only AI45 on two "
            "additional A100 nodes. It does not qualify production d_MAC, WCET, "
            "durable broker restart, or another GPU family."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("grid", "prediction", "c145", "c146", "model", "regression",
                 "reproducibility"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.grid, args.prediction, args.c145, args.c146,
                      args.model, args.regression, args.reproducibility)
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V15 validation failed")


if __name__ == "__main__":
    main()
