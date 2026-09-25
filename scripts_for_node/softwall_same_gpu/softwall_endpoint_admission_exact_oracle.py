#!/usr/bin/env python3
"""Independent exact oracle for the SoftWall endpoint admission model.

The envelope checker uses a deterministic earliest-completion list schedule.
This program independently enumerates accept/reject/endpoint choices for small
instances, verifies the emitted schedule, and compares admitted cardinality.
It is an audit tool; it is not on the runtime control path.
"""

import argparse
from functools import lru_cache
import hashlib
import json
from pathlib import Path


EPSILON = 1e-9


def build_jobs(home_cell_counts, deadline_ms, guard_ms, recovery_ms):
    jobs = []
    global_cell = 0
    for home, cells in enumerate(home_cell_counts):
        for local_cell in range(cells):
            cutoff = deadline_ms - guard_ms - (cells - local_cell) * recovery_ms
            jobs.append({
                "cell": global_cell,
                "home": home,
                "home_local_cell": local_cell,
                "cutoff_ms": float(cutoff),
            })
            global_cell += 1
    return sorted(
        jobs,
        key=lambda item: (
            item["cutoff_ms"], item["home"], item["home_local_cell"],
            item["cell"],
        ),
    )


def exact_maximum_admission(jobs, endpoint_bounds_ms):
    """Enumerate all rejection/endpoint choices and return an exact witness."""
    bounds = tuple(float(value) for value in endpoint_bounds_ms)
    if not bounds:
        return {"admitted": 0, "assignment": [None for _ in jobs],
                "states": 1}
    choices = {}

    @lru_cache(maxsize=None)
    def solve(index, loads):
        if index == len(jobs):
            return 0
        best = solve(index + 1, loads)
        best_choice = None
        candidates = []
        for endpoint, bound in enumerate(bounds):
            finish = round(loads[endpoint] + bound, 9)
            if finish <= jobs[index]["cutoff_ms"] + EPSILON:
                next_loads = list(loads)
                next_loads[endpoint] = finish
                admitted = 1 + solve(index + 1, tuple(next_loads))
                candidates.append((admitted, finish, endpoint, tuple(next_loads)))
        if candidates:
            # Prefer more admissions, then the earliest finishing witness.
            candidate = min(candidates, key=lambda item: (-item[0], item[1], item[2]))
            if candidate[0] > best:
                best = candidate[0]
                best_choice = (candidate[2], candidate[3])
        choices[(index, loads)] = best_choice
        return best

    initial = tuple(0.0 for _ in bounds)
    admitted = solve(0, initial)
    assignment = []
    loads = initial
    for index in range(len(jobs)):
        choice = choices[(index, loads)]
        if choice is None:
            assignment.append(None)
        else:
            endpoint, loads = choice
            assignment.append(endpoint)
    return {
        "admitted": admitted,
        "assignment": assignment,
        "states": solve.cache_info().currsize,
    }


def slot_greedy_count(jobs, endpoint_bounds_ms):
    """Reference form: match each deadline to the smallest unused slot."""
    used = [0 for _ in endpoint_bounds_ms]
    accepted = 0
    for job in jobs:
        candidates = [
            ((used[endpoint] + 1) * float(bound), endpoint)
            for endpoint, bound in enumerate(endpoint_bounds_ms)
        ]
        finish, endpoint = min(candidates) if candidates else (float("inf"), None)
        if finish <= job["cutoff_ms"] + EPSILON:
            used[endpoint] += 1
            accepted += 1
    return accepted


def verify_prediction_schedule(result, scenario):
    errors = []
    seen_cells = set()
    endpoint_finish = {}
    admitted_by_home = [0 for _ in scenario["home_cell_counts"]]
    for item in result["endpoint_schedule"]:
        cell = item["cell"]
        if cell in seen_cells:
            errors.append("duplicate cell {}".format(cell))
        seen_cells.add(cell)
        if not item["admitted"]:
            if item["endpoint"] is not None or item["predicted_finish_ms"] is not None:
                errors.append("rejected cell {} retains endpoint state".format(cell))
            continue
        admitted_by_home[item["home"]] += 1
        if item["predicted_finish_ms"] > item["cutoff_ms"] + EPSILON:
            errors.append("cell {} misses endpoint cutoff".format(cell))
        endpoint = str(item["endpoint"])
        previous = endpoint_finish.get(endpoint, 0.0)
        if item["predicted_finish_ms"] <= previous + EPSILON:
            errors.append("endpoint {} completion is not increasing".format(endpoint))
        endpoint_finish[endpoint] = item["predicted_finish_ms"]
    if len(seen_cells) != scenario["cells"]:
        errors.append("schedule does not cover every cell")
    if admitted_by_home != result["endpoint_admitted_cells_by_home"]:
        errors.append("reported per-home admission count differs from schedule")
    expected_delta = [
        max(0, count - 1) * scenario["recovery_bound_ms"]
        for count in admitted_by_home
    ]
    if expected_delta != result["home_max_released_recovery_slack_ms"]:
        errors.append("conditional slack is not derived from local admission count")
    return errors


def audit_scenario(scenario, result):
    schedule_errors = verify_prediction_schedule(result, scenario)
    pool_results = []
    if "home_endpoint_bounds_ms" in scenario:
        for home, (cells, bounds) in enumerate(zip(
                scenario["home_cell_counts"],
                scenario["home_endpoint_bounds_ms"])):
            jobs = build_jobs(
                [cells], scenario["deadline_ms"], scenario["guard_ms"],
                scenario["recovery_bound_ms"],
            )
            oracle = exact_maximum_admission(jobs, bounds)
            checker_count = result["endpoint_admitted_cells_by_home"][home]
            pool_results.append({
                "home": home,
                "jobs": cells,
                "endpoints": len(bounds),
                "checker_admitted": checker_count,
                "oracle_admitted": oracle["admitted"],
                "oracle_states": oracle["states"],
                "cardinality_match": checker_count == oracle["admitted"],
            })
    else:
        jobs = build_jobs(
            scenario["home_cell_counts"], scenario["deadline_ms"],
            scenario["guard_ms"], scenario["recovery_bound_ms"],
        )
        bounds = scenario["endpoint_path_bounds_ms"]
        oracle = exact_maximum_admission(jobs, bounds)
        pool_results.append({
            "home": "shared-global",
            "jobs": scenario["cells"],
            "endpoints": len(bounds),
            "checker_admitted": result["endpoint_admitted_cells"],
            "oracle_admitted": oracle["admitted"],
            "oracle_states": oracle["states"],
            "cardinality_match": (
                result["endpoint_admitted_cells"] == oracle["admitted"]
            ),
        })
    return {
        "id": scenario["id"],
        "schedule_errors": schedule_errors,
        "pools": pool_results,
        "all_pass": (
            not schedule_errors
            and all(item["cardinality_match"] for item in pool_results)
        ),
    }


def bounded_exhaustive_audit():
    home_patterns = (
        (1,), (2,), (3,), (4,), (5,), (6,),
        (1, 2), (1, 3), (2, 3), (2, 4), (3, 4),
        (1, 2, 3), (2, 2, 3),
    )
    endpoint_patterns = (
        (5,), (10,), (5, 10), (10, 15), (10, 30),
        (15, 25), (20, 45), (10, 20, 35),
    )
    cases = 0
    gaps = []
    max_states = 0
    for homes in home_patterns:
        for deadline in (30, 45, 60, 75, 90, 120, 155):
            for recovery in (5, 10, 12, 20, 25):
                for bounds in endpoint_patterns:
                    jobs = build_jobs(homes, deadline, 2, recovery)
                    oracle = exact_maximum_admission(jobs, bounds)
                    greedy = slot_greedy_count(jobs, bounds)
                    cases += 1
                    max_states = max(max_states, oracle["states"])
                    if greedy != oracle["admitted"]:
                        gaps.append({
                            "home_cell_counts": list(homes),
                            "deadline_ms": deadline,
                            "guard_ms": 2,
                            "recovery_bound_ms": recovery,
                            "endpoint_bounds_ms": list(bounds),
                            "greedy": greedy,
                            "oracle": oracle["admitted"],
                        })
    return {
        "cases": cases,
        "cardinality_gaps": gaps,
        "max_oracle_states": max_states,
        "all_pass": not gaps,
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
    results = {item["id"]: item for item in prediction["results"]}
    if set(scenarios) != set(results):
        raise ValueError("grid and prediction scenario IDs differ")
    scenario_audits = [
        audit_scenario(scenarios[item["id"]], item)
        for item in prediction["results"]
    ]
    bounded = bounded_exhaustive_audit()
    pool_count = sum(len(item["pools"]) for item in scenario_audits)
    all_pass = all(item["all_pass"] for item in scenario_audits) and bounded["all_pass"]
    output = {
        "schema": "softwall-endpoint-admission-exact-oracle-audit-v1",
        "status": "PASS" if all_pass else "FAIL",
        "scope": (
            "Exact finite-state cardinality and schedule-consistency audit for "
            "the synchronous release-zero, endpoint-specific but job-independent "
            "service-bound model; not a queueing or WCET proof."
        ),
        "oracle_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "grid": str(grid_path),
        "grid_sha256": hashlib.sha256(grid_path.read_bytes()).hexdigest(),
        "prediction": str(prediction_path),
        "prediction_sha256": hashlib.sha256(
            prediction_path.read_bytes()
        ).hexdigest(),
        "scenario_count": len(scenario_audits),
        "endpoint_pool_count": pool_count,
        "scenario_failures": [
            item["id"] for item in scenario_audits if not item["all_pass"]
        ],
        "bounded_exhaustive": bounded,
        "scenarios": scenario_audits,
        "all_pass": all_pass,
    }
    temporary = output_path.with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(output_path)


if __name__ == "__main__":
    main()
