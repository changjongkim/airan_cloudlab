#!/usr/bin/env python3
"""Enumerate prospective completion-triggered conditional exchanges.

The deployed controllers represented by envelope v10 wait for every admitted
NeuralRx reply before attempting a recovery-credit/AI-lease exchange.  This
analyzer asks a narrower prospective question: if the controller repaired the
certificate after each bounded completion cohort, which AI transaction classes
could fit in at least one observed-success branch while the all-fail schedule
for every unresolved request remains executable?

This is a deterministic model audit.  It does not qualify the prospective
executor or its new NRx/AI co-run mode on hardware.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path


EPSILON = 1e-9


def enumerate_home_prefix_branches(scenario, result, home):
    """Enumerate outcomes visible at each endpoint-completion cohort."""
    cells = int(scenario["home_cell_counts"][home])
    recovery = float(scenario["recovery_bound_ms"])
    capacity = float(scenario["deadline_ms"]) - float(scenario["guard_ms"])
    admitted = sorted(
        (
            item for item in result["endpoint_schedule"]
            if item["home"] == home and item["admitted"]
        ),
        key=lambda item: (
            float(item["predicted_finish_ms"]), item["cell"]
        ),
    )
    cohort_times = sorted({
        float(item["predicted_finish_ms"]) for item in admitted
    })
    branches = []
    for observed_at in cohort_times:
        visible = [
            item for item in admitted
            if float(item["predicted_finish_ms"]) <= observed_at + EPSILON
        ]
        for outcomes in itertools.product((False, True), repeat=len(visible)):
            successful_cells = [
                item["cell"] for item, success in zip(visible, outcomes)
                if success
            ]
            successes = len(successful_cells)
            remaining = cells - successes
            # A conditional exchange needs both a deleted credit and at least
            # one unresolved/failed/rejected recovery to retain as its gate.
            if successes == 0 or remaining == 0:
                continue
            earliest_recovery = capacity - remaining * recovery
            window = max(0.0, earliest_recovery - observed_at)
            branches.append({
                "observed_at_ms": observed_at,
                "observed_nrx": len(visible),
                "successful_cells": successful_cells,
                "successes": successes,
                "remaining_recovery_obligations": remaining,
                "earliest_recovery_start_ms": earliest_recovery,
                "max_safe_transaction_ms": window,
            })
    return branches


def audit_scenario(scenario, result):
    homes = []
    for home in range(len(scenario["home_cell_counts"])):
        branches = enumerate_home_prefix_branches(scenario, result, home)
        best = max(
            branches,
            key=lambda item: (
                item["max_safe_transaction_ms"],
                -item["observed_at_ms"],
                item["successes"],
            ),
            default=None,
        )
        homes.append({
            "home": home,
            "branch_count": len(branches),
            "max_safe_transaction_ms": (
                0.0 if best is None else best["max_safe_transaction_ms"]
            ),
            "best_branch": best,
        })

    ai_classes = []
    current_pairs = 0
    prospective_pairs = 0
    new_pairs = 0
    for current in result["ai_classes"]:
        effective = float(current["effective_transaction_bound_ms"])
        prospective_homes = [
            item["home"] for item in homes
            if item["home"] not in current["static_admissible_homes"]
            and effective <= item["max_safe_transaction_ms"] + EPSILON
        ]
        current_homes = list(current["exchange_only_candidate_homes"])
        gained = sorted(set(prospective_homes) - set(current_homes))
        current_pairs += len(current_homes)
        prospective_pairs += len(prospective_homes)
        new_pairs += len(gained)
        ai_classes.append({
            "bound_ms": current["bound_ms"],
            "effective_transaction_bound_ms": effective,
            "static_admissible_homes": current["static_admissible_homes"],
            "observe_all_exchange_homes": current_homes,
            "incremental_exchange_homes": prospective_homes,
            "new_incremental_homes": gained,
        })
    return {
        "id": scenario["id"],
        "current_status": result["status"],
        "homes": homes,
        "ai_classes": ai_classes,
        "observe_all_exchange_home_class_pairs": current_pairs,
        "incremental_exchange_home_class_pairs": prospective_pairs,
        "new_incremental_home_class_pairs": new_pairs,
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
        audit_scenario(scenarios[item["id"]], item)
        for item in prediction["results"]
    ]
    candidates = [
        item["id"] for item in audits
        if item["new_incremental_home_class_pairs"]
    ]
    output = {
        "schema": "softwall-incremental-exchange-prospective-v1",
        "status": (
            "PROSPECTIVE_UNQUALIFIED"
            if candidates else "HYPOTHESIS_REJECTED_NO_NEW_CANDIDATE"
        ),
        "scope": (
            "Exact outcome-subset enumeration at bounded endpoint-completion "
            "cohorts for the synchronous grid. Every unresolved request keeps "
            "its recovery obligation. The prospective executor and the new "
            "in-flight NRx/AI co-run order are not physically qualified."
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
        "observe_all_exchange_home_class_pairs": sum(
            item["observe_all_exchange_home_class_pairs"] for item in audits
        ),
        "incremental_exchange_home_class_pairs": sum(
            item["incremental_exchange_home_class_pairs"] for item in audits
        ),
        "new_incremental_home_class_pairs": sum(
            item["new_incremental_home_class_pairs"] for item in audits
        ),
        "candidate_scenarios": candidates,
        "audits": audits,
        "claim_boundary": (
            "A positive candidate is a model-derived experiment target, not "
            "a qualified mode or a throughput result. When no new pair exists, "
            "incremental observation adds no modeled service class over the "
            "mode-specific observe-all executor."
        ),
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(output_path)


if __name__ == "__main__":
    main()
