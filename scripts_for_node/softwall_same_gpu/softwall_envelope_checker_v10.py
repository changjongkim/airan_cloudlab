#!/usr/bin/env python3
"""Deterministic envelope checker with explicit RAN-critical control cost.

Version 10 preserves the v8 timing, ring, and completion-fence checks while
interpreting control_rpc_count as only the synchronous operations that can
block the RAN executor. Deferred control operations remain lifecycle state but
do not consume the local recovery horizon. Version 10 also requires an explicit
single-token ownership qualification for pipelined broker modes.
"""

import argparse
import hashlib
import json
from pathlib import Path


def endpoint_admission_for_homes(home_cell_counts, deadline_ms, guard_ms,
                                 recovery_ms, endpoint_bounds_ms,
                                 endpoint_ring_depths=None):
    """Schedule simultaneous jobs on a shared pool in deadline order."""
    available = [0.0 for _ in endpoint_bounds_ms]
    used = [0 for _ in endpoint_bounds_ms]
    if endpoint_ring_depths is None:
        endpoint_ring_depths = [sum(home_cell_counts) for _ in endpoint_bounds_ms]
    if len(endpoint_ring_depths) != len(endpoint_bounds_ms):
        raise ValueError("each endpoint bound must have one ring depth")
    if any(int(depth) <= 0 for depth in endpoint_ring_depths):
        raise ValueError("endpoint ring depths must be positive")
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
            if used[index] < int(endpoint_ring_depths[index])
        ]
        finish, endpoint = min(choices) if choices else (float("inf"), None)
        admitted = finish <= cutoff
        if admitted:
            available[endpoint] = finish
            used[endpoint] += 1
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
                       endpoint_bounds_ms, endpoint_ring_depths=None):
    return endpoint_admission_for_homes(
        [cells], deadline_ms, guard_ms, recovery_ms, endpoint_bounds_ms,
        endpoint_ring_depths,
    )


def endpoint_admission_for_disjoint_homes(
        home_cell_counts, deadline_ms, guard_ms, recovery_ms,
        home_endpoint_bounds_ms, home_endpoint_ring_depths=None):
    """Schedule each home's requests only on that home's endpoint pool."""
    if len(home_cell_counts) != len(home_endpoint_bounds_ms):
        raise ValueError("each home must have one endpoint-bound list")
    if home_endpoint_ring_depths is None:
        home_endpoint_ring_depths = [None for _ in home_cell_counts]
    if len(home_cell_counts) != len(home_endpoint_ring_depths):
        raise ValueError("each home must have one endpoint-ring-depth list")
    decisions = []
    global_offset = 0
    for home, (cells, endpoint_bounds, ring_depths) in enumerate(
            zip(home_cell_counts, home_endpoint_bounds_ms,
                home_endpoint_ring_depths)):
        local = endpoint_admission_for_homes(
            [cells], deadline_ms, guard_ms, recovery_ms, endpoint_bounds,
            ring_depths,
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


def max_conditional_release_by_home(admitted_by_home, recovery_ms,
                                    home_cell_counts=None,
                                    rejected_recovery_before_exchange=None):
    """Bound credit deletion while a recovery still gates an AI lease."""
    if home_cell_counts is None:
        return [max(0, count - 1) * recovery_ms for count in admitted_by_home]
    if rejected_recovery_before_exchange is None:
        rejected_recovery_before_exchange = [True for _ in admitted_by_home]
    released = []
    for admitted, cells, rejected_first in zip(
            admitted_by_home, home_cell_counts,
            rejected_recovery_before_exchange):
        deletions = (
            max(0, admitted - 1)
            if rejected_first else min(admitted, max(0, cells - 1))
        )
        released.append(deletions * recovery_ms)
    return released


def decision_time_exchange_windows(admission, home_cell_counts, deadline_ms,
                                   guard_ms, recovery_ms,
                                   rejected_recovery_before_exchange):
    """Conservative per-home window at the observe-all decision point.

    Executor mode determines whether rejected NeuralRx jobs are recovered
    before the exchange or remain as live obligations in the compacted tail.
    """
    windows = []
    for home, cells in enumerate(home_cell_counts):
        admitted = [
            item for item in admission
            if item["home"] == home and item["admitted"]
        ]
        admitted_count = len(admitted)
        rejected_count = cells - admitted_count
        rejected_first = rejected_recovery_before_exchange[home]
        max_successes = (
            max(0, admitted_count - 1)
            if rejected_first else min(admitted_count, max(0, cells - 1))
        )
        if max_successes < 1:
            windows.append({
                "home": home,
                "admitted_nrx": admitted_count,
                "rejected_nrx": rejected_count,
                "rejected_recovery_before_exchange": rejected_first,
                "exchange_phase_possible": False,
                "latest_nrx_observation_bound_ms": None,
                "rejected_conventional_charge_ms": rejected_count * recovery_ms,
                "decision_time_bound_ms": None,
                "one_pending_recovery_start_ms": None,
                "max_safe_exchange_transaction_ms": 0.0,
            })
            continue
        latest_observation = max(
            item["predicted_finish_ms"] for item in admitted
        )
        rejected_charge = rejected_count * recovery_ms if rejected_first else 0.0
        decision_time = latest_observation + rejected_charge
        remaining_recovery = (
            admitted_count - max_successes
            if rejected_first else cells - max_successes
        )
        earliest_recovery_start = (
            deadline_ms - guard_ms - remaining_recovery * recovery_ms
        )
        windows.append({
            "home": home,
            "admitted_nrx": admitted_count,
            "rejected_nrx": rejected_count,
            "rejected_recovery_before_exchange": rejected_first,
            "max_observed_successes_used": max_successes,
            "remaining_recovery_obligations": remaining_recovery,
            "exchange_phase_possible": True,
            "latest_nrx_observation_bound_ms": latest_observation,
            "rejected_conventional_charge_ms": rejected_charge,
            "decision_time_bound_ms": decision_time,
            "one_pending_recovery_start_ms": earliest_recovery_start,
            "max_safe_exchange_transaction_ms": max(
                0.0, earliest_recovery_start - decision_time
            ),
        })
    return windows


def evaluate(scenario):
    cells = scenario["cells"]
    deadline = scenario["deadline_ms"]
    guard = scenario["guard_ms"]
    recovery = scenario["recovery_bound_ms"]
    home_cells = scenario.get("home_cell_counts", [cells])
    if sum(home_cells) != cells:
        raise ValueError("home_cell_counts must sum to cells")
    rejected_first = scenario.get("rejected_recovery_before_exchange", True)
    if isinstance(rejected_first, bool):
        rejected_first = [rejected_first for _ in home_cells]
    if len(rejected_first) != len(home_cells):
        raise ValueError("rejected recovery ordering must cover every home")
    mandatory_demand = cells * recovery
    capacity = deadline - guard
    home_demand = [count * recovery for count in home_cells]
    home_slack = [capacity - demand for demand in home_demand]
    base_slack = min(home_slack)
    memory_feasible = scenario["home_memory_feasible"]
    bounds_qualified = scenario["timing_bounds_qualified"]
    control_fault_qualified = scenario.get("control_fault_qualified", True)
    control_budget_required = scenario.get("control_rpc_budget_required", False)
    ownership_required = scenario.get("single_token_ownership_required", False)
    ownership_qualified = scenario.get(
        "single_token_ownership_qualified", not ownership_required
    )
    control_rpc_bound = scenario.get("control_rpc_bound_ms", 0.0)
    control_rpc_count = scenario.get("control_rpc_count", 0)
    control_budget = scenario.get("control_transaction_budget_ms", 0.0)
    control_budget_accounted = scenario.get(
        "control_budget_accounted_in_admission", not control_budget_required
    )
    if "ai_completion_guard_ms" not in scenario:
        raise ValueError("AI completion guard must be explicit in every v8 mode")
    ai_completion_guard = scenario["ai_completion_guard_ms"]
    if ai_completion_guard < 0:
        raise ValueError("AI completion guard must be nonnegative")
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
            scenario.get("home_endpoint_ring_depths"),
        )
        endpoint_scope = "disjoint-per-home"
    else:
        admission = endpoint_admission_for_homes(
            home_cells, deadline, guard, recovery,
            scenario["endpoint_path_bounds_ms"],
            scenario.get("endpoint_ring_depths"),
        )
        endpoint_scope = "shared-global"
    admitted_by_home = admitted_cells_by_home(admission, len(home_cells))
    rejected_by_home = [
        count - admitted
        for count, admitted in zip(home_cells, admitted_by_home)
    ]
    admitted = sum(admitted_by_home)
    home_delta = max_conditional_release_by_home(
        admitted_by_home, recovery, home_cells, rejected_first
    )
    delta_max = max(home_delta) if home_delta else 0.0
    decision_windows = decision_time_exchange_windows(
        admission, home_cells, deadline, guard, recovery, rejected_first
    )
    ai = []
    effective_control_budget = control_budget if control_budget_accounted else 0.0
    for bound in scenario["ai_bounds_ms"]:
        effective_bound = (
            bound + effective_control_budget + ai_completion_guard
        )
        static_homes = [
            index for index, slack in enumerate(home_slack)
            if effective_bound <= slack
        ]
        geometry_exchange_homes = [
            index for index, slack in enumerate(home_slack)
            if slack < effective_bound <= slack + home_delta[index]
        ]
        exchange_homes = [
            item["home"] for item in decision_windows
            if item["exchange_phase_possible"]
            and item["home"] not in static_homes
            and effective_bound <= item["max_safe_exchange_transaction_ms"]
        ]
        ai.append({
            "bound_ms": bound,
            "control_budget_ms": effective_control_budget,
            "ai_completion_guard_ms": ai_completion_guard,
            "effective_transaction_bound_ms": effective_bound,
            "static_admissible": bool(static_homes),
            "static_admissible_homes": static_homes,
            "exchange_only_geometry_candidate": bool(geometry_exchange_homes),
            "exchange_only_geometry_candidate_homes": geometry_exchange_homes,
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
        reason = "declared RAN-critical control path is not bounded and charged to AI admission"
    elif ownership_required and not ownership_qualified:
        status = "UQ"
        reason = "pipelined broker mode lacks qualified single-token ownership"
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
        "home_decision_time_exchange_windows": decision_windows,
        "home_max_safe_exchange_transaction_ms": [
            item["max_safe_exchange_transaction_ms"]
            for item in decision_windows
        ],
        "endpoint_admitted_cells": admitted,
        "endpoint_rejected_cells": cells - admitted,
        "endpoint_admitted_cells_by_home": admitted_by_home,
        "endpoint_rejected_cells_by_home": rejected_by_home,
        "endpoint_scope": endpoint_scope,
        "endpoint_ring_depths": scenario.get("endpoint_ring_depths"),
        "home_endpoint_ring_depths": scenario.get("home_endpoint_ring_depths"),
        "rejected_recovery_before_exchange": rejected_first,
        "control_fault_model": scenario.get("control_fault_model", "nominal"),
        "control_fault_qualified": control_fault_qualified,
        "control_rpc_budget_required": control_budget_required,
        "control_rpc_bound_ms": control_rpc_bound,
        "control_rpc_count": control_rpc_count,
        "control_critical_operations": scenario.get("control_critical_operations"),
        "control_deferred_operations": scenario.get("control_deferred_operations"),
        "required_control_transaction_budget_ms": required_control_budget,
        "declared_control_transaction_budget_ms": control_budget,
        "control_budget_accounted_in_admission": control_budget_accounted,
        "control_budget_qualified": control_budget_qualified,
        "single_token_ownership_required": ownership_required,
        "single_token_ownership_qualified": ownership_qualified,
        "ownership_contract": scenario.get("ownership_contract"),
        "ai_completion_guard_ms": ai_completion_guard,
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
        "schema": "softwall-deterministic-envelope-v10",
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
