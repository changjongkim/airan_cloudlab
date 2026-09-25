#!/usr/bin/env python3
"""Close the V13 correction chain from falsification through physical validation."""

import argparse
import hashlib
import json
from pathlib import Path

from build_softwall_envelope_v13 import NEW_ID, OLD_ID


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    return json.loads(Path(path).read_text())


def validate(grid_path, prediction_path, regression_path, c138_arm_path,
             c139_path, c140_path):
    grid = load(grid_path)
    prediction = load(prediction_path)
    regression = load(regression_path)
    c138 = load(c138_arm_path)
    c139 = load(c139_path)
    c140 = load(c140_path)
    results = {row["id"]: row for row in prediction["results"]}
    c138_failed = [name for name, passed in c138["gates"].items() if not passed]
    gates = {
        "c138_isolated_5ms_wall_bound_failure": (
            not c138["all_pass"]
            and c138_failed == ["rpc_wall_bound"]
            and c138["control_evidence"]["max_elapsed_ms"] > 5.0
            and c138["control_evidence"]["wall_timeout_exceeded"] > 0
        ),
        "c139_corrected_bound_passes": (
            c139["all_pass"]
            and c139["control_evidence"]["socket_timeout_ms"] == 5.0
            and c139["control_evidence"]["declared_per_rpc_admission_bound_ms"] == 7.0
            and c139["control_evidence"]["socket_timeout_elapsed_exceeded"] > 0
            and c139["control_evidence"]["admission_bound_exceeded"] == 0
        ),
        "c140_corrected_exchange_passes": (
            c140["all_pass"]
            and c140["totals"]["ai35_candidate_exchanges"] > 0
            and c140["model_counterfactual"]["static_margin_ms"] == -5.0
            and c140["model_counterfactual"]["conditional_margin_ms"] == 0.0
        ),
        "v13_supersedes_old_mode": (
            results[OLD_ID]["status"] == "UQ"
            and results[NEW_ID]["status"] == "QSU"
        ),
        "v12_v13_regression_passes": regression["all_pass"],
        "v13_expected_counts": prediction["counts"] == {
            "QSU": 5, "QSN": 0, "MI": 3, "UQ": 11,
        },
    }
    paths = [grid_path, prediction_path, regression_path, c138_arm_path,
             c139_path, c140_path]
    return {
        "schema": "softwall-envelope-v13-validation-summary-v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "inputs": {str(path): sha256(path) for path in paths},
        "model_counts": prediction["counts"],
        "c138": {
            "node": load(c138["homes"][0]["controller"])["host"],
            "failed_gates": c138_failed,
            "rpc_max_elapsed_ms": c138["control_evidence"]["max_elapsed_ms"],
            "socket_timeout_elapsed_exceeded": c138["control_evidence"]["wall_timeout_exceeded"],
        },
        "c139": {
            "nodes": c139["nodes"],
            "arms": c139["totals"]["arms"],
            "radio_records": c139["totals"]["radio_records"],
            "radio_records_after_fault_detection": c139["totals"]["radio_records_after_fault_detection"],
            "rpc_records": c139["control_evidence"]["records"],
            "rpc_max_elapsed_ms": c139["control_evidence"]["maximum_elapsed_ms"],
            "socket_timeout_elapsed_exceeded": c139["control_evidence"]["socket_timeout_elapsed_exceeded"],
            "admission_bound_exceeded": c139["control_evidence"]["admission_bound_exceeded"],
        },
        "c140": {
            "nodes": c140["nodes"],
            "radio_records": c140["totals"]["radio_records"],
            "candidate_branches": c140["totals"]["candidate_branches"],
            "ai35_candidate_exchanges": c140["totals"]["ai35_candidate_exchanges"],
            "minimum_physical_guarded_horizon_margin_ms": c140["totals"]["minimum_physical_guarded_horizon_margin_ms"],
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "V13 closes one finite-sample model/implementation loop: C138 "
            "falsifies timeout-equals-wall-bound, C139 qualifies the separated "
            "7 ms admission charge, and C140 realizes the resulting exchange-only "
            "AI35 class. It is not WCET, production d_MAC, independent-node C140, "
            "or cross-family evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    for name in ("grid", "prediction", "regression", "c138_arm", "c139", "c140"):
        parser.add_argument("--" + name.replace("_", "-"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = validate(args.grid, args.prediction, args.regression,
                      args.c138_arm, args.c139, args.c140)
    output = Path(args.output)
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V13 validation failed")


if __name__ == "__main__":
    main()
