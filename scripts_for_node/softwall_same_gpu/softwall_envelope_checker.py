#!/usr/bin/env python3
"""Deterministic envelope checker for simultaneous-release SoftWall modes."""

import argparse
import hashlib
import json
from pathlib import Path


def endpoint_admission_for_homes(home_cell_counts, deadline_ms, guard_ms, recovery_ms, endpoint_bounds_ms):
    """Conservative shared-endpoint list schedule for simultaneous home batches."""
    available = [0.0 for _ in endpoint_bounds_ms]
    decisions = []
    jobs = []
    global_cell = 0
    for home, cells in enumerate(home_cell_counts):
        for local_cell in range(cells):
            cutoff = deadline_ms - guard_ms - (cells - local_cell) * recovery_ms
            jobs.append((cutoff, home, local_cell, global_cell))
            global_cell += 1
    for cutoff, home, local_cell, cell in sorted(jobs):
        choices = [
            (available[index] + bound, index)
            for index, bound in enumerate(endpoint_bounds_ms)
        ]
        finish, endpoint = min(choices) if choices else (float("inf"), None)
        admitted = finish <= cutoff
        if admitted:
            available[endpoint] = finish
        decisions.append({
            "cell": cell,
            "home": home,
            "home_local_cell": local_cell,
            "cutoff_ms": cutoff,
            "endpoint": endpoint if admitted else None,
            "predicted_finish_ms": finish if admitted else None,
            "admitted": admitted,
        })
    return decisions


def endpoint_admission(cells, deadline_ms, guard_ms, recovery_ms, endpoint_bounds_ms):
    return endpoint_admission_for_homes(
        [cells], deadline_ms, guard_ms, recovery_ms, endpoint_bounds_ms
    )


def evaluate(scenario):
    cells = scenario["cells"]
    deadline = scenario["deadline_ms"]
    guard = scenario["guard_ms"]
    recovery = scenario["recovery_bound_ms"]
    home_cells = scenario.get("home_cell_counts", [cells])
    if sum(home_cells) != cells:
        raise ValueError("home_cell_counts must sum to cells")
    mandatory_demand = cells * recovery
    capacity = deadline - guard
    home_demand = [count * recovery for count in home_cells]
    home_slack = [capacity - demand for demand in home_demand]
    base_slack = min(home_slack)
    memory_feasible = scenario["home_memory_feasible"]
    bounds_qualified = scenario["timing_bounds_qualified"]
    admission = endpoint_admission_for_homes(
        home_cells, deadline, guard, recovery, scenario["endpoint_path_bounds_ms"]
    )
    admitted = sum(item["admitted"] for item in admission)
    home_delta = [max(0.0, (count - 1) * recovery) if admitted else 0.0 for count in home_cells]
    delta_max = max(home_delta) if home_delta else 0.0
    ai = []
    for bound in scenario["ai_bounds_ms"]:
        static_homes = [index for index, slack in enumerate(home_slack) if bound <= slack]
        exchange_homes = [
            index for index, slack in enumerate(home_slack)
            if slack < bound <= slack + home_delta[index]
        ]
        ai.append({
            "bound_ms": bound,
            "static_admissible": bool(static_homes),
            "static_admissible_homes": static_homes,
            "exchange_only_candidate": bool(exchange_homes),
            "exchange_only_candidate_homes": exchange_homes,
            "too_large_even_after_exchange": not static_homes and not exchange_homes,
        })
    if not memory_feasible:
        status = "MI"
        reason = "home receiver residency is infeasible"
    elif any(demand > capacity for demand in home_demand):
        status = "MI"
        reason = "all-fail mandatory recovery demand exceeds deadline capacity"
    elif not bounds_qualified:
        status = "UQ"
        reason = "required timing/lifecycle bounds are not qualified"
    elif any(item["static_admissible"] or item["exchange_only_candidate"] for item in ai):
        status = "QSU"
        reason = "safe mode admits at least one bounded AI class"
    else:
        status = "QSN"
        reason = "mandatory path is safe but no listed AI class fits"
    return {
        "id": scenario["id"],
        "gpu_count": scenario["gpu_count"],
        "cells": cells,
        "status": status,
        "reason": reason,
        "mandatory_capacity_ms": capacity,
        "mandatory_demand_ms": mandatory_demand,
        "home_cell_counts": home_cells,
        "home_mandatory_demand_ms": home_demand,
        "home_base_all_fail_slack_ms": home_slack,
        "base_all_fail_slack_ms": base_slack,
        "home_max_released_recovery_slack_ms": home_delta,
        "max_released_recovery_slack_ms": delta_max,
        "endpoint_admitted_cells": admitted,
        "endpoint_rejected_cells": cells - admitted,
        "endpoint_schedule_is_timing_only_counterfactual": not memory_feasible or not bounds_qualified,
        "endpoint_schedule": admission,
        "ai_classes": ai,
        "evidence": scenario.get("evidence", []),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    config = json.loads(input_path.read_text())
    results = [evaluate(scenario) for scenario in config["scenarios"]]
    output = {
        "schema": "softwall-deterministic-envelope-v1",
        "checker_sha256": hashlib.sha256(Path(__file__).resolve().read_bytes()).hexdigest(),
        "input": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "scope": config["scope"],
        "classification": config["classification"],
        "results": results,
        "counts": {
            key: sum(item["status"] == key for item in results)
            for key in ("QSU", "QSN", "MI", "UQ")
        },
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(output_path)


if __name__ == "__main__":
    main()
