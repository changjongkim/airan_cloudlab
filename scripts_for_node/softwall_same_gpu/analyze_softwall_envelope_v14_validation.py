#!/usr/bin/env python3
"""Validate the V14 prediction against C142-C144 evidence."""

import argparse
import hashlib
import json
from pathlib import Path

from build_softwall_envelope_v14 import V14_ID


def load(path):
    return json.loads(Path(path).read_text())


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(grid_path, prediction_path, c142_path, c143_path, c144_path,
             model_path):
    prediction = load(prediction_path)
    c142 = load(c142_path)
    c143 = load(c143_path)
    c144 = load(c144_path)
    model = load(model_path)
    rows = {row["id"]: row for row in prediction["results"]}
    mode = rows[V14_ID]
    ai45 = next(row for row in mode["ai_classes"]
                if row["bound_ms"] == 45.0)
    gates = {
        "v14_mode_is_qsu": mode["status"] == "QSU",
        "only_commit_is_critical": (
            mode["control_rpc_count"] == 1
            and mode["required_control_transaction_budget_ms"] == 7.0
            and mode["control_critical_operations"] == ["commit"]
        ),
        "ai45_predicted_exchange_only": (
            ai45["effective_transaction_bound_ms"] == 54.0
            and not ai45["static_admissible"]
            and ai45["exchange_only_candidate"]
        ),
        "c142_ai45_pass": c142["all_pass"]
            and c142["totals"]["ai45_candidate_exchanges"] > 0,
        "c143_fault_matrix_pass": c143["all_pass"]
            and c143["totals"]["arms"] == 6,
        "deferred_rpc_really_exceeded_commit_bound": (
            c143["totals"]["deferred_over_7ms"] > 0
            and c143["totals"]["maximum_commit_ms"] <= 7.0
        ),
        "c144_independent_node_pass": c144["all_pass"],
        "two_v14_nodes": set(c142["nodes"] + c144["nodes"])
            == {"nid001361", "nid002049"},
        "finite_model_pass": model["all_pass"] and not model["violations"],
        "v14_expected_counts": prediction["counts"] == {
            "QSU": 6, "QSN": 0, "MI": 3, "UQ": 11,
        },
    }
    paths = [grid_path, prediction_path, c142_path, c143_path, c144_path,
             model_path]
    return {
        "schema": "softwall-envelope-v14-validation-summary-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "inputs": {str(path): sha256(path) for path in paths},
        "model_counts": prediction["counts"],
        "v14_mode": mode,
        "c142": c142["totals"],
        "c143": c143["totals"],
        "c144": c144["totals"],
        "finite_model": {
            "states": model["states"], "edges": model["edges"],
            "violations": len(model["violations"]),
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "V14 closes a finite-sample model/implementation loop for pipelined "
            "global control and an exchange-only AI45 class on two A100 nodes. "
            "It does not qualify production d_MAC, cross-family behavior, or WCET."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("grid", "prediction", "c142", "c143", "c144", "model"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.grid, args.prediction, args.c142, args.c143,
                      args.c144, args.model)
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V14 validation failed")


if __name__ == "__main__":
    main()
