#!/usr/bin/env python3.11
"""C167 analytical model for safely reclaimable shared-GPU capacity.

The model uses identical, simultaneously visible AI units at one decision
event.  Failures are beta-binomial within a RAN home and independent across
homes.  Homes are mapped round-robin to recovery GPU lanes.  This restricted
model has a closed-form count for each policy and an exhaustive small-state
checker that enumerates lane assignments and all AI/recovery orders.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
from collections import defaultdict
from pathlib import Path


DECISION_MS = 45.0
GUARD_MS = 2.0
CONTROL_MS = 5.0
AI_CLASSES = {
    64: {"service_bound_ms": 35.0, "value": 64},
    256: {"service_bound_ms": 65.0, "value": 256},
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def home_sizes(cells: int, homes: int) -> tuple[int, ...]:
    if not 1 <= homes <= cells:
        raise ValueError("homes must be between one and cells")
    return tuple(cells // homes + (index < cells % homes) for index in range(homes))


def beta_binomial_pmf(count: int, total: int, fail_probability: float,
                      correlation: float) -> float:
    if correlation == 0.0:
        return math.comb(total, count) * fail_probability**count * (1-fail_probability)**(total-count)
    if correlation == 1.0:
        if count == 0:
            return 1.0 - fail_probability
        if count == total:
            return fail_probability
        return 0.0
    concentration = 1.0 / correlation - 1.0
    alpha = fail_probability * concentration
    beta = (1.0 - fail_probability) * concentration
    log_value = (
        math.lgamma(total + 1) - math.lgamma(count + 1) - math.lgamma(total-count + 1)
        + math.lgamma(count + alpha) + math.lgamma(total-count + beta)
        - math.lgamma(total + alpha + beta)
        + math.lgamma(alpha + beta) - math.lgamma(alpha) - math.lgamma(beta)
    )
    return math.exp(log_value)


@functools.lru_cache(maxsize=None)
def lane_failure_distribution(cells: int, homes: int, gpus: int,
                              fail_probability: float,
                              correlation: float) -> tuple[tuple[tuple[int, ...], float], ...]:
    sizes = home_sizes(cells, homes)
    home_pmfs = [
        tuple(beta_binomial_pmf(k, size, fail_probability, correlation)
              for k in range(size + 1))
        for size in sizes
    ]
    aggregated: dict[tuple[int, ...], float] = defaultdict(float)
    for failures_by_home in itertools.product(*(range(size + 1) for size in sizes)):
        probability = math.prod(
            home_pmfs[index][failures]
            for index, failures in enumerate(failures_by_home)
        )
        lane_failures = [0] * gpus
        for home, failures in enumerate(failures_by_home):
            lane_failures[home % gpus] += failures
        aggregated[tuple(lane_failures)] += probability
    total_probability = sum(aggregated.values())
    if not math.isclose(total_probability, 1.0, abs_tol=1e-10):
        raise AssertionError(f"failure distribution does not sum to one: {total_probability}")
    return tuple(sorted(aggregated.items()))


def lane_potential_counts(cells: int, homes: int, gpus: int) -> tuple[int, ...]:
    counts = [0] * gpus
    for home, size in enumerate(home_sizes(cells, homes)):
        counts[home % gpus] += size
    return tuple(counts)


def count_from_capacities(capacities: tuple[int, ...], offered: int) -> int:
    return min(offered, sum(max(0, value) for value in capacities))


def ai_first_capacities(recoveries: tuple[int, ...], recovery_ms: float,
                        window_ms: float, ai_deadline_ms: float,
                        transaction_ms: float) -> tuple[int, ...]:
    return tuple(max(0, math.floor(
        min(ai_deadline_ms, window_ms - count * recovery_ms) / transaction_ms
        + 1e-12
    )) for count in recoveries)


def recovery_first_capacities(recoveries: tuple[int, ...], recovery_ms: float,
                              window_ms: float, ai_deadline_ms: float,
                              transaction_ms: float) -> tuple[int, ...]:
    return tuple(max(0, math.floor(
        (min(ai_deadline_ms, window_ms) - count * recovery_ms) / transaction_ms
        + 1e-12
    )) for count in recoveries)


def balanced_blind_assignment(gpus: int, per_lane_capacity: int,
                              offered: int) -> tuple[int, ...]:
    assigned = [0] * gpus
    for _ in range(offered):
        eligible = [lane for lane in range(gpus) if assigned[lane] < per_lane_capacity]
        if not eligible:
            break
        lane = min(eligible, key=lambda item: (assigned[item], item))
        assigned[lane] += 1
    return tuple(assigned)


def evaluate_realization(*, potential: tuple[int, ...], failures: tuple[int, ...],
                         recovery_ms: float, window_ms: float,
                         ai_service_ms: float, ai_deadline_ms: float,
                         offered: int) -> dict:
    transaction_ms = CONTROL_MS + ai_service_ms
    static_caps = ai_first_capacities(
        potential, recovery_ms, window_ms, ai_deadline_ms, transaction_ms
    )
    softwall_caps = ai_first_capacities(
        failures, recovery_ms, window_ms, ai_deadline_ms, transaction_ms
    )
    recovery_caps = recovery_first_capacities(
        failures, recovery_ms, window_ms, ai_deadline_ms, transaction_ms
    )
    blind_per_lane = max(0, math.floor(
        min(window_ms, ai_deadline_ms) / transaction_ms + 1e-12
    ))
    blind_assignment = balanced_blind_assignment(
        len(failures), blind_per_lane, offered
    )
    debt_blind_violation = any(
        blind_assignment[lane] * transaction_ms
        + failures[lane] * recovery_ms > window_ms + 1e-9
        for lane in range(len(failures))
    )
    static_count = count_from_capacities(static_caps, offered)
    recovery_count = count_from_capacities(recovery_caps, offered)
    softwall_count = count_from_capacities(softwall_caps, offered)
    blind_count = sum(blind_assignment)
    return {
        "static_global_safe": static_count,
        "debt_blind_current_idle": blind_count,
        "recovery_first": recovery_count,
        "softwall": softwall_count,
        "offline_oracle": softwall_count,
        "debt_blind_violation": debt_blind_violation,
        "static_raw_slack_ms": sum(max(0.0, window_ms-r*recovery_ms) for r in potential),
        "softwall_raw_slack_ms": sum(max(0.0, window_ms-r*recovery_ms) for r in failures),
    }


def evaluate_point(*, cells: int, homes: int, gpus: int, success_probability: float,
                   correlation: float, expiry_ms: float, recovery_ms: float,
                   context: int, ai_deadline_ms: float, offered: int) -> dict:
    fail_probability = 1.0 - success_probability
    potential = lane_potential_counts(cells, homes, gpus)
    window_ms = expiry_ms - GUARD_MS - DECISION_MS
    if window_ms <= 0:
        raise ValueError("radio guard precedes decision")
    expected = defaultdict(float)
    for failures, probability in lane_failure_distribution(
        cells, homes, gpus, fail_probability, correlation
    ):
        realized = evaluate_realization(
            potential=potential,
            failures=failures,
            recovery_ms=recovery_ms,
            window_ms=window_ms,
            ai_service_ms=AI_CLASSES[context]["service_bound_ms"],
            ai_deadline_ms=ai_deadline_ms,
            offered=offered,
        )
        for policy in (
            "static_global_safe", "debt_blind_current_idle",
            "recovery_first", "softwall", "offline_oracle",
        ):
            expected[policy] += probability * realized[policy]
        expected["debt_blind_violation_probability"] += (
            probability * int(realized["debt_blind_violation"])
        )
        expected["static_raw_slack_ms"] += probability * realized["static_raw_slack_ms"]
        expected["softwall_raw_slack_ms"] += probability * realized["softwall_raw_slack_ms"]
    oracle = expected["offline_oracle"]
    value = AI_CLASSES[context]["value"]
    return {
        **{key: expected[key] for key in (
            "static_global_safe", "debt_blind_current_idle",
            "recovery_first", "softwall", "offline_oracle",
        )},
        "debt_blind_violation_probability": expected["debt_blind_violation_probability"],
        "static_raw_slack_ms": expected["static_raw_slack_ms"],
        "softwall_raw_slack_ms": expected["softwall_raw_slack_ms"],
        "softwall_timely_value": expected["softwall"] * value,
        "static_timely_value": expected["static_global_safe"] * value,
        "recovery_first_timely_value": expected["recovery_first"] * value,
        "softwall_gap_to_oracle_pct": 100.0 * (oracle-expected["softwall"]) / oracle if oracle else 0.0,
        "recovery_first_gap_to_oracle_pct": 100.0 * (oracle-expected["recovery_first"]) / oracle if oracle else 0.0,
        "static_gap_to_oracle_pct": 100.0 * (oracle-expected["static_global_safe"]) / oracle if oracle else 0.0,
    }


def lane_order_feasible(ai_jobs: int, recoveries: int, transaction_ms: float,
                        recovery_ms: float, window_ms: float,
                        ai_deadline_ms: float, recovery_first: bool = False) -> bool:
    items = "A" * ai_jobs + "R" * recoveries
    if recovery_first:
        orders = (("R",) * recoveries + ("A",) * ai_jobs,)
    else:
        orders = set(itertools.permutations(items))
    for order in orders:
        now = 0.0
        valid = True
        for kind in order:
            now += transaction_ms if kind == "A" else recovery_ms
            if kind == "A" and now > ai_deadline_ms + 1e-9:
                valid = False
                break
        if valid and now <= window_ms + 1e-9:
            return True
    return False


def exact_policy_count(recoveries: tuple[int, ...], offered: int,
                       transaction_ms: float, recovery_ms: float,
                       window_ms: float, ai_deadline_ms: float,
                       recovery_first: bool = False) -> int:
    best = 0
    for allocation in itertools.product(range(offered + 1), repeat=len(recoveries)):
        assigned = sum(allocation)
        if assigned > offered or assigned <= best:
            continue
        if all(lane_order_feasible(
            allocation[lane], recoveries[lane], transaction_ms,
            recovery_ms, window_ms, ai_deadline_ms, recovery_first,
        ) for lane in range(len(recoveries))):
            best = assigned
    return best


def exact_check() -> dict:
    cases = 0
    mismatches = []
    for cells in (2, 3, 4):
        for homes in range(1, min(cells, 2) + 1):
            for gpus in (1, 2):
                potential = lane_potential_counts(cells, homes, gpus)
                for failure_bits in itertools.product((0, 1), repeat=cells):
                    failures_by_home = [0] * homes
                    for cell, failed in enumerate(failure_bits):
                        failures_by_home[cell % homes] += failed
                    failures = [0] * gpus
                    for home, count in enumerate(failures_by_home):
                        failures[home % gpus] += count
                    failures = tuple(failures)
                    for expiry_ms, recovery_ms, context, deadline_ms, offered in itertools.product(
                        (100.0, 155.0), (12.0, 25.0), (64, 256),
                        (50.0, 1000.0), (1, 2, 4),
                    ):
                        window = expiry_ms - GUARD_MS - DECISION_MS
                        transaction = CONTROL_MS + AI_CLASSES[context]["service_bound_ms"]
                        analytic = evaluate_realization(
                            potential=potential, failures=failures,
                            recovery_ms=recovery_ms, window_ms=window,
                            ai_service_ms=AI_CLASSES[context]["service_bound_ms"],
                            ai_deadline_ms=deadline_ms, offered=offered,
                        )
                        exact_static = exact_policy_count(
                            potential, offered, transaction, recovery_ms,
                            window, deadline_ms,
                        )
                        exact_oracle = exact_policy_count(
                            failures, offered, transaction, recovery_ms,
                            window, deadline_ms,
                        )
                        exact_recovery = exact_policy_count(
                            failures, offered, transaction, recovery_ms,
                            window, deadline_ms, recovery_first=True,
                        )
                        cases += 1
                        if (analytic["static_global_safe"] != exact_static
                                or analytic["softwall"] != exact_oracle
                                or analytic["offline_oracle"] != exact_oracle
                                or analytic["recovery_first"] != exact_recovery):
                            mismatches.append({
                                "cells": cells, "homes": homes, "gpus": gpus,
                                "failures": failures, "expiry_ms": expiry_ms,
                                "recovery_ms": recovery_ms, "context": context,
                                "deadline_ms": deadline_ms, "offered": offered,
                                "analytic": analytic,
                                "exact": {
                                    "static": exact_static,
                                    "oracle": exact_oracle,
                                    "recovery_first": exact_recovery,
                                },
                            })
    return {"cases": cases, "mismatches": len(mismatches), "first_mismatches": mismatches[:10]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol = json.loads(args.protocol.read_text())
    if protocol.get("status") != "FROZEN_BEFORE_C167_MODEL":
        raise RuntimeError("C167 protocol is not frozen")
    expected_hash = protocol["source_sha256"]["model"]
    if sha256(Path(__file__).resolve()) != expected_hash:
        raise RuntimeError("C167 model source changed after protocol freeze")

    grid = protocol["grid"]
    fields = [
        "cells", "homes", "gpus", "success_probability", "correlation",
        "expiry_ms", "recovery_ms", "context", "ai_deadline_ms", "offered",
        "static_requests", "debt_blind_requests", "recovery_first_requests",
        "softwall_requests", "oracle_requests", "debt_blind_violation_probability",
        "static_raw_slack_ms", "softwall_raw_slack_ms",
        "static_gap_to_oracle_pct", "recovery_first_gap_to_oracle_pct",
        "softwall_gap_to_oracle_pct", "static_timely_value",
        "recovery_first_timely_value", "softwall_timely_value",
    ]
    rows = []
    reference_rows = []
    softwall_better_static = 0
    recovery_gap_ge_mde = 0
    unsafe_debt_blind = 0
    maximum_increment = None
    for cells, success, correlation, expiry, recovery, context, deadline, gpus, homes, offered in itertools.product(
        grid["cells"], grid["nrx_success_probability"], grid["failure_correlation"],
        grid["radio_expiry_ms"], grid["recovery_bound_ms"], grid["ai_context"],
        grid["ai_deadline_from_decision_ms"], grid["gpus"], grid["homes"],
        grid["ai_units_per_epoch"],
    ):
        if homes > cells:
            continue
        metrics = evaluate_point(
            cells=cells, homes=homes, gpus=gpus,
            success_probability=success, correlation=correlation,
            expiry_ms=expiry, recovery_ms=recovery, context=context,
            ai_deadline_ms=deadline, offered=offered,
        )
        row = [
            cells, homes, gpus, success, correlation, expiry, recovery, context,
            deadline, offered, metrics["static_global_safe"],
            metrics["debt_blind_current_idle"], metrics["recovery_first"],
            metrics["softwall"], metrics["offline_oracle"],
            metrics["debt_blind_violation_probability"],
            metrics["static_raw_slack_ms"], metrics["softwall_raw_slack_ms"],
            metrics["static_gap_to_oracle_pct"],
            metrics["recovery_first_gap_to_oracle_pct"],
            metrics["softwall_gap_to_oracle_pct"],
            metrics["static_timely_value"], metrics["recovery_first_timely_value"],
            metrics["softwall_timely_value"],
        ]
        rows.append(row)
        if metrics["softwall"] > metrics["static_global_safe"] + 1e-12:
            softwall_better_static += 1
        if metrics["recovery_first_gap_to_oracle_pct"] >= protocol["minimum_effect_pct"]:
            recovery_gap_ge_mde += 1
        if metrics["debt_blind_violation_probability"] > 0:
            unsafe_debt_blind += 1
        increment = metrics["softwall_timely_value"] - metrics["static_timely_value"]
        if maximum_increment is None or increment > maximum_increment[0]:
            maximum_increment = (increment, row)
        if (cells == 4 and homes == 2 and gpus == 1 and expiry == 155
                and recovery == 25 and context in (64, 256) and offered in (1, 4)
                and success == 0.5 and correlation in (0.0, 0.5, 1.0)):
            reference_rows.append(dict(zip(fields, row)))

    exact = exact_check()
    all_pass = exact["mismatches"] == 0 and all(
        math.isclose(row[fields.index("softwall_gap_to_oracle_pct")], 0.0, abs_tol=1e-12)
        for row in rows
    )
    result = {
        "schema": "softwall-c167-reclaimable-capacity-model-v1",
        "status": "C167_CAPACITY_MODEL_PASS" if all_pass else "C167_CAPACITY_MODEL_FAIL",
        "all_pass": all_pass,
        "claim_scope": (
            "Analytical batch model with identical AI units, common decision time, "
            "home-local beta-binomial failures, and round-robin home-to-GPU placement. "
            "It predicts qualified-bound capacity; it is not a WCET or physical result."
        ),
        "constants": {
            "decision_ms": DECISION_MS,
            "guard_ms": GUARD_MS,
            "control_per_ai_ms": CONTROL_MS,
            "ai_classes": AI_CLASSES,
            "failure_model": (
                "beta-binomial within each home with the stated intraclass correlation; "
                "homes are independent"
            ),
        },
        "grid": grid,
        "row_fields": fields,
        "rows": rows,
        "exact_checker": exact,
        "reference_rows": reference_rows,
        "summary": {
            "grid_points": len(rows),
            "softwall_better_than_static_points": softwall_better_static,
            "recovery_first_gap_at_least_mde_points": recovery_gap_ge_mde,
            "debt_blind_positive_violation_probability_points": unsafe_debt_blind,
            "maximum_softwall_minus_static_timely_value": maximum_increment[0],
            "maximum_increment_row": dict(zip(fields, maximum_increment[1])),
            "exact_cases": exact["cases"],
            "exact_mismatches": exact["mismatches"],
        },
        "physical_validation_points": protocol["physical_validation_points"],
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": protocol["source_sha256"],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "summary": result["summary"],
        "reference_rows": reference_rows,
    }, indent=2))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
