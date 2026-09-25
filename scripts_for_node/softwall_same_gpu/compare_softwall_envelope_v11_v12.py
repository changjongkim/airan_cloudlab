#!/usr/bin/env python3
"""Record the complete AI-guard correction from envelope v11 to v12."""

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


def compare(v11_path, v12_path, scenario_id=TARGET):
    v11_path = Path(v11_path).resolve()
    v12_path = Path(v12_path).resolve()
    old_doc = json.loads(v11_path.read_text())
    new_doc = json.loads(v12_path.read_text())
    old = scenario(old_doc, scenario_id)
    new = scenario(new_doc, scenario_id)
    old_ai = ai_class(old)
    new_ai = ai_class(new)
    windows = [float(value) for value in new[
        "home_max_safe_exchange_transaction_ms"
    ]]
    effective = float(new_ai["effective_transaction_bound_ms"])
    margins = [window - effective for window in windows]
    gates = {
        "mode_counts_unchanged": old_doc["counts"] == new_doc["counts"],
        "physical_geometry_unchanged": (
            old["endpoint_admitted_cells_by_home"]
            == new["endpoint_admitted_cells_by_home"] == [2, 2]
            and old["home_max_safe_exchange_transaction_ms"]
            == new["home_max_safe_exchange_transaction_ms"] == [58.0, 58.0]
        ),
        "v11_omitted_ai_completion_guard": (
            float(old_ai["effective_transaction_bound_ms"]) == 55.0
            and "ai_completion_guard_ms" not in old_ai
        ),
        "v12_charges_two_ms_ai_completion_guard": (
            float(new_ai["control_budget_ms"]) == 15.0
            and float(new_ai["ai_completion_guard_ms"]) == 2.0
            and effective == 57.0
        ),
        "candidate_survives_complete_charge": (
            new_ai["exchange_only_candidate"]
            and new_ai["exchange_only_candidate_homes"] == [0, 1]
        ),
        "corrected_margin_is_one_ms_per_home": margins == [1.0, 1.0],
        "classification_is_unchanged": old["status"] == new["status"] == "QSU",
    }
    return {
        "schema": "softwall-envelope-v11-v12-ai-guard-regression-v1",
        "scope": (
            "Regression audit for the frozen two-home full-control mode. "
            "V12 charges the runtime's physical AI completion guard separately; "
            "the comparison is a model correction, not new physical evidence."
        ),
        "inputs": {
            "v11": str(v11_path),
            "v11_sha256": sha256(v11_path),
            "v12": str(v12_path),
            "v12_sha256": sha256(v12_path),
        },
        "scenario": scenario_id,
        "v11": {
            "effective_ai40_transaction_ms": old_ai[
                "effective_transaction_bound_ms"
            ],
            "model_margin_ms": [
                float(value) - float(old_ai["effective_transaction_bound_ms"])
                for value in old["home_max_safe_exchange_transaction_ms"]
            ],
            "ai40_exchange_only_candidate": old_ai["exchange_only_candidate"],
        },
        "v12": {
            "raw_ai_bound_ms": new_ai["bound_ms"],
            "broker_control_budget_ms": new_ai["control_budget_ms"],
            "ai_completion_guard_ms": new_ai["ai_completion_guard_ms"],
            "effective_ai40_transaction_ms": effective,
            "decision_window_ms": windows,
            "model_margin_ms": margins,
            "ai40_exchange_only_candidate": new_ai["exchange_only_candidate"],
        },
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--v11", required=True)
    parser.add_argument("--v12", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = compare(args.v11, args.v12)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("v11-to-v12 regression gate failed")


if __name__ == "__main__":
    main()
