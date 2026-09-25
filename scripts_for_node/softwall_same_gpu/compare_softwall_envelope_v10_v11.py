#!/usr/bin/env python3
"""Record the physical-ring/executor-phase correction from envelope v10 to v11."""

import argparse
import hashlib
import json
from pathlib import Path


TARGET = "sharded_home_g2_c8_broker_crash_full_budget_control_matrix_c132_c134"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def scenario(document, scenario_id=TARGET):
    return next(item for item in document["results"] if item["id"] == scenario_id)


def ai_class(item, bound_ms=40.0):
    return next(
        value for value in item["ai_classes"]
        if float(value["bound_ms"]) == float(bound_ms)
    )


def compare(v10_path, v11_path, scenario_id=TARGET):
    v10_path = Path(v10_path).resolve()
    v11_path = Path(v11_path).resolve()
    old = scenario(json.loads(v10_path.read_text()), scenario_id)
    new = scenario(json.loads(v11_path.read_text()), scenario_id)
    old_ai = ai_class(old)
    new_ai = ai_class(new)
    effective = float(new_ai["effective_transaction_bound_ms"])
    margins = [float(value) - effective
               for value in new["home_max_safe_exchange_transaction_ms"]]
    gates = {
        "v10_assumed_four_admissions_per_home":
            old["endpoint_admitted_cells_by_home"] == [4, 4],
        "v11_caps_admission_at_physical_ring_slots":
            new["endpoint_admitted_cells_by_home"] == [2, 2]
            and new["home_endpoint_ring_depths"] == [[1, 1], [1, 1]],
        "v11_keeps_rejected_recoveries_live_at_exchange":
            new["rejected_recovery_before_exchange"] == [False, False],
        "v10_rejected_ai40_at_decision_time":
            not old_ai["exchange_only_candidate"],
        "v11_admits_ai40_at_decision_time":
            new_ai["exchange_only_candidate"]
            and new_ai["exchange_only_candidate_homes"] == [0, 1],
        "effective_transaction_is_55_ms": effective == 55.0,
        "v11_window_is_58_ms_per_home":
            new["home_max_safe_exchange_transaction_ms"] == [58.0, 58.0],
        "v11_margin_is_3_ms_per_home": margins == [3.0, 3.0],
        "classification_is_unchanged": old["status"] == new["status"] == "QSU",
    }
    return {
        "schema": "softwall-envelope-v10-v11-regression-v1",
        "scope": (
            "Regression audit for the frozen two-home, four-cell-per-home, "
            "full-control-budget mode. It checks model fidelity, not WCET."
        ),
        "inputs": {
            "v10": str(v10_path),
            "v10_sha256": sha256(v10_path),
            "v11": str(v11_path),
            "v11_sha256": sha256(v11_path),
        },
        "scenario": scenario_id,
        "v10": {
            "admitted_by_home": old["endpoint_admitted_cells_by_home"],
            "rejected_by_home": old["endpoint_rejected_cells_by_home"],
            "released_recovery_slack_ms": old["home_max_released_recovery_slack_ms"],
            "decision_window_ms": old["home_max_safe_exchange_transaction_ms"],
            "ai40_exchange_only_candidate": old_ai["exchange_only_candidate"],
        },
        "v11": {
            "ring_depths_by_home": new["home_endpoint_ring_depths"],
            "admitted_by_home": new["endpoint_admitted_cells_by_home"],
            "rejected_by_home": new["endpoint_rejected_cells_by_home"],
            "rejected_recovery_before_exchange":
                new["rejected_recovery_before_exchange"],
            "released_recovery_slack_ms": new["home_max_released_recovery_slack_ms"],
            "decision_window_ms": new["home_max_safe_exchange_transaction_ms"],
            "effective_ai40_transaction_ms": effective,
            "model_margin_ms": margins,
            "ai40_exchange_only_candidate": new_ai["exchange_only_candidate"],
        },
        "interpretation": (
            "Finite endpoint credits reduce admitted NeuralRx jobs from four to "
            "two per home. In this executor the two rejected recoveries remain "
            "live, so two successful NeuralRx observations can release 50 ms "
            "while two recoveries still gate the transaction. The resulting "
            "58 ms window admits the 55 ms AI-plus-control class with 3 ms margin."
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v10", required=True)
    parser.add_argument("--v11", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.v10, args.v11)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("v10-to-v11 regression gate failed")


if __name__ == "__main__":
    main()
