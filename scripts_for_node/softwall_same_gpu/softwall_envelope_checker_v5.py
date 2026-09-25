#!/usr/bin/env python3
"""Deterministic envelope checker with home-local conditional slack.

Version 5 preserves the v4 safety and control-budget checks, but derives the
maximum recovery-credit release independently for each home.  The bound is
based on the number of optional NeuralRx jobs actually admitted at that home,
not on the total number admitted elsewhere.
"""

import argparse
import hashlib
import json
from pathlib import Path


def endpoint_admission_for_homes(home_cell_counts, deadline_ms, guard_ms,
                                 recovery_ms, endpoint_bounds_ms):
    """Schedule simultaneous jobs on a shared pool in deadline order.

    An endpoint e exposes completion slots p_e, 2*p_e, ... because all jobs
    have release zero and the same endpoint-specific service bound p_e.  The
    earliest-finish choice consumes the smallest unused slot across endpoints.
    """
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


def endpoint_admission(cells, deadline_ms, guard_ms, recovery_ms,
                       endpoint_bounds_ms):
    return endpoint_admission_for_homes(
        [cells], deadline_ms, guard_ms, recovery_ms, endpoint_bounds_ms
    )


def endpoint_admission_for_disjoint_homes(
        home_cell_counts, deadline_ms, guard_ms, recovery_ms,
        home_endpoint_bounds_ms):
    """Schedule each home's requests only on that home's endpoint pool."""
    if len(home_cell_counts) != len(home_endpoint_bounds_ms):
        raise ValueError("each home must have one endpoint-bound list")
    decisions = []
    global_offset = 0
    for home, (cells, endpoint_bounds) in enumerate(
            zip(home_cell_counts, home_endpoint_bounds_ms)):
        local = endpoint_admission_for_homes(
            [cells], deadline_ms, guard_ms, recovery_ms, endpoint_bounds
        )
        for item in local:
            item["home"] = home
            item["cell"] += global_offset
            item["endpoint"] = (
                "h{}:e{}".format(home, item["endpoint"])
                if item["endpoint"] is not None else None
            )
            decisions.append(item)
        global_offset += cells
    return decisions


def admitted_cells_by_home(admission, home_count):
    counts = [0 for _ in range(home_count)]
    for item in admission:
        if item["admitted"]:
            counts[item["home"]] += 1
    return counts


def max_conditional_release_by_home(admitted_by_home, recovery_ms):
    """Upper-bound credit deletion while a recovery remains to gate an AI lease.

    With k admitted optional NRx jobs at a home, the after-observation exchange
    phase exists only while at least one recovery remains pending.  At most
    k-1 admitted jobs can therefore have succeeded and deleted their recovery
    credits.  A home with no local admission receives no conditional slack.
    """
    return [max(0, count - 1) * recovery_ms for count in admitted_by_home]


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
    control_fault_qualified = scenario.get("control_fault_qualified", True)
    control_budget_required = scenario.get("control_rpc_budget_required", False)
    control_rpc_bound = scenario.get("control_rpc_bound_ms", 0.0)
    control_rpc_count = scenario.get("control_rpc_count", 0)
    control_budget = scenario.get("control_transaction_budget_ms", 0.0)
    control_budget_accounted = scenario.get(
        "control_budget_accounted_in_admission", not control_budget_required
    )
    required_control_budget = control_rpc_bound * control_rpc_count
    control_budget_qualified = (
        not control_budget_required
        or (
            control_rpc_bound > 0
            and control_rpc_count > 0
            and control_budget_accounted
            and control_budget >= required_control_budget
        )
    )
    if "home_endpoint_bounds_ms" in scenario:
        admission = endpoint_admission_for_disjoint_homes(
            home_cells, deadline, guard, recovery,
            scenario["home_endpoint_bounds_ms"],
        )
        endpoint_scope = "disjoint-per-home"
    else:
        admission = endpoint_admission_for_homes(
            home_cells, deadline, guard, recovery,
            scenario["endpoint_path_bounds_ms"],
        )
        endpoint_scope = "shared-global"
    admitted_by_home = admitted_cells_by_home(admission, len(home_cells))
    rejected_by_home = [
        count - admitted
        for count, admitted in zip(home_cells, admitted_by_home)
    ]
    admitted = sum(admitted_by_home)
    home_delta = max_conditional_release_by_home(admitted_by_home, recovery)
    delta_max = max(home_delta) if home_delta else 0.0
    ai = []
    effective_control_budget = control_budget if control_budget_accounted else 0.0
    for bound in scenario["ai_bounds_ms"]:
        effective_bound = bound + effective_control_budget
        static_homes = [
            index for index, slack in enumerate(home_slack)
            if effective_bound <= slack
        ]
        exchange_homes = [
            index for index, slack in enumerate(home_slack)
            if slack < effective_bound <= slack + home_delta[index]
        ]
        ai.append({
            "bound_ms": bound,
            "control_budget_ms": effective_control_budget,
            "effective_transaction_bound_ms": effective_bound,
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
    elif not control_fault_qualified:
        status = "UQ"
        reason = "declared control-plane fault class is not qualified"
    elif not control_budget_qualified:
        status = "UQ"
        reason = "full synchronous control path is not bounded and charged to AI admission"
    elif any(item["static_admissible"] or item["exchange_only_candidate"]
             for item in ai):
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
        "endpoint_admitted_cells_by_home": admitted_by_home,
        "endpoint_rejected_cells_by_home": rejected_by_home,
        "endpoint_scope": endpoint_scope,
        "control_fault_model": scenario.get("control_fault_model", "nominal"),
        "control_fault_qualified": control_fault_qualified,
        "control_rpc_budget_required": control_budget_required,
        "control_rpc_bound_ms": control_rpc_bound,
        "control_rpc_count": control_rpc_count,
        "required_control_transaction_budget_ms": required_control_budget,
        "declared_control_transaction_budget_ms": control_budget,
        "control_budget_accounted_in_admission": control_budget_accounted,
        "control_budget_qualified": control_budget_qualified,
        "ambiguous_token_policy": scenario.get("ambiguous_token_policy"),
        "endpoint_schedule_is_timing_only_counterfactual": (
            not memory_feasible or not bounds_qualified
        ),
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
        "schema": "softwall-deterministic-envelope-v5",
        "checker_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
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
