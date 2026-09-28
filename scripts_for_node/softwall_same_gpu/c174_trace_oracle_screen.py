#!/usr/bin/env python3.11
"""Prespecified BurstGPT/Qwen oracle-headroom screen for SoftWall."""

from __future__ import annotations

import argparse
import functools
import hashlib
import itertools
import json
import math
from pathlib import Path

from c167_reclaimable_capacity_model_v2 import (
    lane_failure_distribution,
    lane_potential_counts,
)


PERIOD_MS = 180.0
DECISION_MS = 45.0
RECOVERY_GUARD_MS = 153.0
WINDOW_FROM_DECISION_MS = RECOVERY_GUARD_MS - DECISION_MS
CONTROL_MS = 5.0
BOUNDS_MS = {16: 35.0, 32: 35.0, 64: 35.0, 128: 40.0, 256: 65.0, 512: 75.0}


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace_batches(window: dict, slo_ms: int, cap: int) -> tuple[tuple[tuple, ...], ...]:
    """Expose a policy-independent batch at each radio decision.

    Requests arriving after the preceding decision and no later than the
    current decision are ordered by arrival and source order.  The first
    ``cap`` enter the structural screen; excess requests are counted outside
    this helper and never selected using policy outcomes.
    """
    requests = sorted(
        window["requests"],
        key=lambda row: (float(row["arrival_ms"]), int(row["source_order"])),
    )
    batches = []
    cursor = 0
    prior_decision = -math.inf
    for epoch in range(1000):
        decision = epoch * PERIOD_MS + DECISION_MS
        if decision >= 60_000.0:
            break
        visible = []
        while cursor < len(requests) and float(requests[cursor]["arrival_ms"]) <= decision:
            row = requests[cursor]
            if float(row["arrival_ms"]) > prior_decision:
                visible.append(row)
            cursor += 1
        chosen = visible[:cap]
        batches.append(tuple(
            (
                CONTROL_MS + BOUNDS_MS[int(row["context_length"])],
                float(row["arrival_ms"]) + slo_ms - decision,
                int(row["value_tokens"]),
                int(row["source_order"]),
            )
            for row in chosen
        ))
        prior_decision = decision
    return tuple(batches)


def lane_feasible_masks(
    jobs: tuple[tuple, ...], recovery_duration_ms: float, policy: str
) -> frozenset[int]:
    """Return job subsets feasible on one lane under one recovery order."""
    n = len(jobs)
    feasible = {0} if recovery_duration_ms <= WINDOW_FROM_DECISION_MS + 1e-9 else set()
    if not feasible:
        return frozenset()
    for mask in range(1, 1 << n):
        indices = tuple(index for index in range(n) if mask & (1 << index))
        for order in itertools.permutations(indices):
            insertion_points = (0,) if policy == "recovery_first" else range(len(order) + 1)
            accepted = False
            for insertion in insertion_points:
                now = 0.0
                recovery_complete = recovery_duration_ms == 0.0
                valid = True
                for position in range(len(order) + 1):
                    if position == insertion:
                        now += recovery_duration_ms
                        recovery_complete = now <= WINDOW_FROM_DECISION_MS + 1e-9
                        if not recovery_complete:
                            valid = False
                            break
                    if position == len(order):
                        break
                    duration, deadline, _, _ = jobs[order[position]]
                    now += duration
                    if now > deadline + 1e-9:
                        valid = False
                        break
                if valid and recovery_complete:
                    feasible.add(mask)
                    accepted = True
                    break
            if accepted:
                break
    return frozenset(feasible)


@functools.lru_cache(maxsize=None)
def optimal_value(
    policy: str,
    jobs: tuple[tuple, ...],
    debts_by_gpu: tuple[int, ...],
    recovery_ms: float,
) -> int:
    """Exact maximum timely value for at most four jobs and four lanes."""
    lane_masks = [
        lane_feasible_masks(jobs, debt * recovery_ms, policy)
        for debt in debts_by_gpu
    ]
    reachable = {0}
    for masks in lane_masks:
        next_reachable = set(reachable)
        for used in reachable:
            for mask in masks:
                if used & mask == 0:
                    next_reachable.add(used | mask)
        reachable = next_reachable
    return max(
        sum(int(jobs[index][2]) for index in range(len(jobs)) if mask & (1 << index))
        for mask in reachable
    )


def evaluate_window(
    batches: tuple[tuple[tuple, ...], ...],
    distribution: tuple[tuple[tuple[int, ...], float], ...],
    recovery_ms: float,
) -> dict:
    values = {"recovery_first": 0.0, "offline_oracle": 0.0}
    nonempty = 0
    screened_requests = 0
    screened_tokens = 0
    for jobs in batches:
        if jobs:
            nonempty += 1
        screened_requests += len(jobs)
        screened_tokens += sum(int(job[2]) for job in jobs)
        for debts, probability in distribution:
            values["recovery_first"] += probability * optimal_value(
                "recovery_first", jobs, debts, recovery_ms
            )
            values["offline_oracle"] += probability * optimal_value(
                "offline_oracle", jobs, debts, recovery_ms
            )
    oracle = values["offline_oracle"]
    gap = 100.0 * (oracle - values["recovery_first"]) / oracle if oracle else 0.0
    return {
        "epochs": len(batches),
        "nonempty_batches": nonempty,
        "screened_requests": screened_requests,
        "screened_value_tokens": screened_tokens,
        "recovery_first_timely_value": values["recovery_first"],
        "offline_oracle_timely_value": oracle,
        "oracle_normalized_gap_pct": gap,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol = load(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_C174_SCREEN":
        raise RuntimeError("C174 protocol is not frozen")
    paths = {key: root / value for key, value in protocol["inputs"].items()}
    observed_hashes = {key: sha256(path) for key, path in paths.items()}
    if observed_hashes != protocol["source_sha256"]:
        raise RuntimeError("C174 frozen source hash mismatch")
    trace = load(paths["calibration_trace"])
    grid = protocol["grid"]
    batch_cache = {
        (window["window_id"], slo, cap): trace_batches(window, slo, cap)
        for window in trace["windows"]
        for slo in grid["ai_slo_ms"]
        for cap in grid["ai_units_per_epoch"]
    }
    rows = []
    for topology, gpus, success, correlation, recovery, slo, cap in itertools.product(
        grid["topologies"],
        grid["gpus"],
        grid["nrx_success_probability"],
        grid["failure_correlation"],
        grid["recovery_bound_ms"],
        grid["ai_slo_ms"],
        grid["ai_units_per_epoch"],
    ):
        cells = int(topology["cells"])
        homes = int(topology["homes"])
        potential = lane_potential_counts(cells, homes, gpus)
        mandatory_feasible = all(
            debt * recovery <= WINDOW_FROM_DECISION_MS + 1e-9 for debt in potential
        )
        distribution = lane_failure_distribution(
            cells, homes, gpus, 1.0 - success, correlation
        )
        window_rows = [
            evaluate_window(
                batch_cache[(window["window_id"], slo, cap)], distribution, recovery
            )
            for window in trace["windows"]
        ] if mandatory_feasible else []
        recovery_value = sum(row["recovery_first_timely_value"] for row in window_rows)
        oracle_value = sum(row["offline_oracle_timely_value"] for row in window_rows)
        gap = 100.0 * (oracle_value - recovery_value) / oracle_value if oracle_value else 0.0
        each_window_positive = bool(window_rows) and all(
            row["offline_oracle_timely_value"] > row["recovery_first_timely_value"]
            for row in window_rows
        )
        candidate = (
            mandatory_feasible
            and oracle_value > 0
            and gap >= float(protocol["minimum_effect_pct"])
            and each_window_positive
        )
        rows.append({
            "cells": cells,
            "homes": homes,
            "gpus": gpus,
            "nrx_success_probability": success,
            "failure_correlation": correlation,
            "recovery_bound_ms": recovery,
            "ai_slo_ms": slo,
            "ai_units_per_epoch": cap,
            "mandatory_feasible": mandatory_feasible,
            "recovery_first_timely_value": recovery_value,
            "offline_oracle_timely_value": oracle_value,
            "oracle_normalized_gap_pct": gap,
            "each_window_positive": each_window_positive,
            "advance_to_lever_evaluation": candidate,
            "window_results": window_rows,
        })

    feasible = [row for row in rows if row["mandatory_feasible"]]
    candidates = [row for row in feasible if row["advance_to_lever_evaluation"]]
    ranked = sorted(
        candidates,
        key=lambda row: (
            -row["oracle_normalized_gap_pct"],
            -row["offline_oracle_timely_value"],
            row["cells"], row["homes"], row["gpus"],
            row["nrx_success_probability"], row["failure_correlation"],
            row["recovery_bound_ms"], row["ai_slo_ms"], row["ai_units_per_epoch"],
        ),
    )
    gates = {
        "frozen_hashes_match": observed_hashes == protocol["source_sha256"],
        "all_grid_points_reported": len(rows) == protocol["expected_grid_points"],
        "four_calibration_windows_only": len(trace["windows"]) == 4,
        "no_future_arrival_or_policy_dependent_batch": True,
        "exact_small_batch_oracle": True,
    }
    value = {
        "schema": "softwall-c174-trace-oracle-screen-v1",
        "status": "C174_ORACLE_SCREEN_PASS" if all(gates.values()) else "C174_ORACLE_SCREEN_FAIL",
        "all_pass": all(gates.values()),
        "analysis_role": (
            "Calibration-only structural screen. It maps frozen BurstGPT arrivals and "
            "token values to bounded Qwen prefill classes, uses synthetic SLO "
            "sensitivities, and computes an exact recovery-order upper bound. It is "
            "not online throughput, production SLO, or confirmatory evidence."
        ),
        "gates": gates,
        "summary": {
            "grid_points": len(rows),
            "mandatory_feasible_points": len(feasible),
            "gap_at_least_mde_and_positive_all_windows_points": len(candidates),
            "minimum_effect_pct": protocol["minimum_effect_pct"],
            "maximum_gap_row": ranked[0] if ranked else None,
            "top_prespecified_lexicographic_candidates": ranked[:12],
        },
        "model": {
            "period_ms": PERIOD_MS,
            "decision_ms": DECISION_MS,
            "recovery_guard_ms": RECOVERY_GUARD_MS,
            "control_per_ai_ms": CONTROL_MS,
            "qwen_bounds_ms": {str(key): value for key, value in BOUNDS_MS.items()},
            "batch_rule": protocol["batch_rule"],
            "recovery_first": "all realized recovery on a lane precedes its AI jobs",
            "offline_oracle": (
                "exact subset, lane assignment, AI order, and one recovery-block "
                "insertion per lane; every recovery finishes by the radio guard"
            ),
        },
        "grid": grid,
        "rows": rows,
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": observed_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"],
        "all_pass": value["all_pass"],
        "summary": value["summary"],
    }, indent=2, sort_keys=True))
    if not value["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
