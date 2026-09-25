#!/usr/bin/env python3
"""Audit coarse conditional-slack geometry at the actual decision boundary.

The deterministic envelope's W+Delta test is a capacity-geometry condition.
The current controller performs its joint after-observation exchange only after
all admitted NeuralRx replies have been observed and all rejected NeuralRx jobs
have run conventional recovery.  This analyzer conservatively charges those
steps before asking whether a bounded AI transaction fits with one recovery
still pending.
"""

import argparse
import hashlib
import json
from pathlib import Path


EPSILON = 1e-9


def home_decision_window(scenario, result, home):
    schedule = [
        item for item in result["endpoint_schedule"]
        if item["home"] == home and item["admitted"]
    ]
    admitted = len(schedule)
    rejected = scenario["home_cell_counts"][home] - admitted
    recovery = float(scenario["recovery_bound_ms"])
    if admitted < 2:
        return {
            "home": home,
            "admitted_nrx": admitted,
            "rejected_nrx": rejected,
            "exchange_phase_possible": False,
            "latest_nrx_observation_bound_ms": None,
            "rejected_conventional_charge_ms": rejected * recovery,
            "decision_time_bound_ms": None,
            "one_pending_recovery_start_ms": None,
            "max_safe_exchange_transaction_ms": 0.0,
        }
    latest_observation = max(item["predicted_finish_ms"] for item in schedule)
    rejected_charge = rejected * recovery
    decision_time = latest_observation + rejected_charge
    one_pending_start = (
        float(scenario["deadline_ms"])
        - float(scenario["guard_ms"])
        - recovery
    )
    window = max(0.0, one_pending_start - decision_time)
    return {
        "home": home,
        "admitted_nrx": admitted,
        "rejected_nrx": rejected,
        "exchange_phase_possible": True,
        "latest_nrx_observation_bound_ms": latest_observation,
        "rejected_conventional_charge_ms": rejected_charge,
        "decision_time_bound_ms": decision_time,
        "one_pending_recovery_start_ms": one_pending_start,
        "max_safe_exchange_transaction_ms": window,
    }


def audit_scenario(scenario, result):
    windows = [
        home_decision_window(scenario, result, home)
        for home in range(len(scenario["home_cell_counts"]))
    ]
    ai = []
    geometry_candidates = 0
    decision_candidates = 0
    eliminated = 0
    any_service = False
    for coarse in result["ai_classes"]:
        effective = float(coarse["effective_transaction_bound_ms"])
        decision_homes = [
            item["home"] for item in windows
            if item["exchange_phase_possible"]
            and effective <= item["max_safe_exchange_transaction_ms"] + EPSILON
            and item["home"] not in coarse["static_admissible_homes"]
        ]
        geometry_homes = list(coarse["exchange_only_candidate_homes"])
        removed_homes = sorted(set(geometry_homes) - set(decision_homes))
        geometry_candidates += len(geometry_homes)
        decision_candidates += len(decision_homes)
        eliminated += len(removed_homes)
        any_service = any_service or coarse["static_admissible"] or bool(decision_homes)
        ai.append({
            "bound_ms": coarse["bound_ms"],
            "effective_transaction_bound_ms": effective,
            "static_admissible_homes": coarse["static_admissible_homes"],
            "geometry_exchange_candidate_homes": geometry_homes,
            "decision_time_exchange_candidate_homes": decision_homes,
            "geometry_only_homes": removed_homes,
        })
    if result["status"] in ("MI", "UQ"):
        decision_status = result["status"]
    else:
        decision_status = "QSU" if any_service else "QSN"
    return {
        "id": scenario["id"],
        "original_status": result["status"],
        "decision_time_status": decision_status,
        "status_changed": result["status"] != decision_status,
        "home_windows": windows,
        "ai_classes": ai,
        "geometry_exchange_home_class_pairs": geometry_candidates,
        "decision_time_exchange_home_class_pairs": decision_candidates,
        "geometry_only_home_class_pairs": eliminated,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--grid", required=True)
    parser.add_argument("--prediction", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    grid_path = Path(args.grid).resolve()
    prediction_path = Path(args.prediction).resolve()
    output_path = Path(args.output).resolve()
    grid = json.loads(grid_path.read_text())
    prediction = json.loads(prediction_path.read_text())
    scenarios = {item["id"]: item for item in grid["scenarios"]}
    audits = [
        audit_scenario(scenarios[result["id"]], result)
        for result in prediction["results"]
    ]
    output = {
        "schema": "softwall-decision-time-envelope-audit-v1",
        "status": "PASS",
        "scope": (
            "Conservative audit of the current observe-all controller order. "
            "It charges the latest admitted NRx path bound and sequential "
            "conventional bounds for rejected NRx jobs before a one-pending-"
            "recovery exchange. It is not an online-arrival or WCET proof."
        ),
        "analyzer_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "grid": str(grid_path),
        "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "prediction": str(prediction_path),
        "prediction_sha256": hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest(),
        "scenario_count": len(audits),
        "geometry_exchange_home_class_pairs": sum(
            item["geometry_exchange_home_class_pairs"] for item in audits
        ),
        "decision_time_exchange_home_class_pairs": sum(
            item["decision_time_exchange_home_class_pairs"] for item in audits
        ),
        "geometry_only_home_class_pairs": sum(
            item["geometry_only_home_class_pairs"] for item in audits
        ),
        "classification_changes": [
            item["id"] for item in audits if item["status_changed"]
        ],
        "audits": audits,
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(output_path)


if __name__ == "__main__":
    main()
