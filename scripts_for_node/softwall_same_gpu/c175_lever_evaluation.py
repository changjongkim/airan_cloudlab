#!/usr/bin/env python3.11
"""Evaluate scheduling levers only on C174 points that passed its 5% gate."""

from __future__ import annotations

import argparse
import functools
import itertools
import json
from pathlib import Path

from c167_reclaimable_capacity_model_v2 import lane_failure_distribution
from c174_trace_oracle_screen import (
    WINDOW_FROM_DECISION_MS,
    load,
    optimal_value,
    sha256,
    trace_batches,
)


@functools.lru_cache(maxsize=None)
def greedy_ai_first_value(
    jobs: tuple[tuple, ...], debts_by_gpu: tuple[int, ...], recovery_ms: float
) -> int:
    """Certificate-preserving EDF/value greedy with before/after recovery phases."""
    lane_time = [0.0] * len(debts_by_gpu)
    remaining = []
    order = sorted(
        range(len(jobs)), key=lambda i: (jobs[i][1], -jobs[i][2], jobs[i][3])
    )
    value = 0
    for index in order:
        duration, deadline, tokens, _ = jobs[index]
        eligible = [
            lane for lane, debt in enumerate(debts_by_gpu)
            if lane_time[lane] + duration <= deadline + 1e-9
            and lane_time[lane] + duration + debt * recovery_ms
                <= WINDOW_FROM_DECISION_MS + 1e-9
        ]
        if not eligible:
            remaining.append(index)
            continue
        lane = min(eligible, key=lambda item: (lane_time[item] + duration, item))
        lane_time[lane] += duration
        value += int(tokens)
    for lane, debt in enumerate(debts_by_gpu):
        lane_time[lane] += debt * recovery_ms
        if lane_time[lane] > WINDOW_FROM_DECISION_MS + 1e-9:
            raise AssertionError("greedy invalidated the recovery certificate")
    for index in remaining:
        duration, deadline, tokens, _ = jobs[index]
        eligible = [
            lane for lane in range(len(debts_by_gpu))
            if lane_time[lane] + duration <= deadline + 1e-9
        ]
        if not eligible:
            continue
        lane = min(eligible, key=lambda item: (lane_time[item] + duration, item))
        lane_time[lane] += duration
        value += int(tokens)
    return value


@functools.lru_cache(maxsize=None)
def weak_compositions(total: int, parts: int) -> tuple[tuple[int, ...], ...]:
    if parts == 1:
        return ((total,),)
    rows = []
    for first in range(total + 1):
        for suffix in weak_compositions(total - first, parts - 1):
            rows.append((first,) + suffix)
    return tuple(rows)


@functools.lru_cache(maxsize=None)
def flexible_recovery_oracle_value(
    jobs: tuple[tuple, ...], debts_by_gpu: tuple[int, ...], recovery_ms: float
) -> int:
    candidates = [
        placement for placement in weak_compositions(sum(debts_by_gpu), len(debts_by_gpu))
        if all(debt * recovery_ms <= WINDOW_FROM_DECISION_MS + 1e-9 for debt in placement)
    ]
    return max(
        (optimal_value("offline_oracle", jobs, placement, recovery_ms)
         for placement in candidates),
        default=0,
    )


def expected_values(
    batches: tuple[tuple[tuple, ...], ...],
    distribution: tuple[tuple[tuple[int, ...], float], ...],
    recovery_ms: float,
) -> dict:
    totals = {"recovery_first": 0.0, "ai_first_greedy": 0.0,
              "fixed_placement_oracle": 0.0, "flexible_recovery_oracle": 0.0}
    for jobs in batches:
        for debts, probability in distribution:
            totals["recovery_first"] += probability * optimal_value(
                "recovery_first", jobs, debts, recovery_ms
            )
            totals["ai_first_greedy"] += probability * greedy_ai_first_value(
                jobs, debts, recovery_ms
            )
            totals["fixed_placement_oracle"] += probability * optimal_value(
                "offline_oracle", jobs, debts, recovery_ms
            )
            totals["flexible_recovery_oracle"] += probability * flexible_recovery_oracle_value(
                jobs, debts, recovery_ms
            )
    return totals


def sionna_radio_lever(channel: dict, allowed_loss: float) -> dict:
    cells = []
    for model, model_data in channel["models"].items():
        for stratum in model_data["strata"]:
            trials = int(stratum["trials"])
            cells.append({
                "model": model,
                "esno_db": float(stratum["esno_db"]),
                "incremental_radio_value": (
                    int(stratum["neural_only_correct"])
                    - int(stratum["conventional_only_correct"])
                ) / trials,
                "nrx_success_probability": int(stratum["neural_correct"]) / trials,
            })
    max_radio = sum(max(0.0, cell["incremental_radio_value"]) for cell in cells)
    floor = max_radio - allowed_loss
    feasible = []
    for mask in range(1 << len(cells)):
        radio = sum(
            cells[index]["incremental_radio_value"]
            for index in range(len(cells)) if mask & (1 << index)
        )
        if radio + 1e-12 < floor:
            continue
        selected = [cells[index] for index in range(len(cells)) if mask & (1 << index)]
        expected_debt = sum(1.0 - cell["nrx_success_probability"] for cell in selected)
        feasible.append((expected_debt, len(selected), -radio, mask, radio))
    best = min(feasible)
    positive_only = [cell for cell in cells if cell["incremental_radio_value"] > 0]
    baseline_debt = sum(1.0 - cell["nrx_success_probability"] for cell in positive_only)
    return {
        "allowed_absolute_radio_loss": allowed_loss,
        "maximum_positive_incremental_radio_value": max_radio,
        "radio_floor": floor,
        "max_radio_positive_cells": len(positive_only),
        "max_radio_expected_recovery_debt": baseline_debt,
        "debt_aware_selected_cells": best[1],
        "debt_aware_expected_recovery_debt": best[0],
        "debt_aware_radio_value": best[4],
        "recovery_debt_reduction": baseline_debt - best[0],
        "selected": [cells[index] for index in range(len(cells)) if best[3] & (1 << index)],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    protocol = load(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_C175_ANALYSIS":
        raise RuntimeError("C175 protocol is not frozen")
    paths = {key: root / value for key, value in protocol["inputs"].items()}
    observed = {key: sha256(path) for key, path in paths.items()}
    if observed != protocol["source_sha256"]:
        raise RuntimeError("C175 source hash mismatch")
    screen = load(paths["c174_result"])
    trace = load(paths["calibration_trace"])
    channel = load(paths["sionna_holdout"])
    candidates = [row for row in screen["rows"] if row["advance_to_lever_evaluation"]]
    if len(candidates) != protocol["expected_candidate_points"]:
        raise RuntimeError("C174 candidate set changed")
    rows = []
    for source in candidates:
        distribution = lane_failure_distribution(
            source["cells"], source["homes"], source["gpus"],
            1.0 - source["nrx_success_probability"], source["failure_correlation"],
        )
        windows = []
        for window in trace["windows"]:
            batches = trace_batches(
                window, source["ai_slo_ms"], source["ai_units_per_epoch"]
            )
            windows.append(expected_values(
                batches, distribution, source["recovery_bound_ms"]
            ))
        totals = {key: sum(row[key] for row in windows) for key in windows[0]}
        oracle = totals["fixed_placement_oracle"]
        rows.append({
            **{key: value for key, value in source.items() if key != "window_results"},
            "lever_values": totals,
            "greedy_gap_to_fixed_oracle_pct": (
                100.0 * (oracle - totals["ai_first_greedy"]) / oracle if oracle else 0.0
            ),
            "flexible_recovery_gain_over_fixed_pct": (
                100.0 * (totals["flexible_recovery_oracle"] - oracle) / oracle
                if oracle else 0.0
            ),
        })
    match_errors = [
        abs(row["lever_values"]["fixed_placement_oracle"] - row["offline_oracle_timely_value"])
        for row in rows
    ]
    recovery_errors = [
        abs(row["lever_values"]["recovery_first"] - row["recovery_first_timely_value"])
        for row in rows
    ]
    best_placement = max(rows, key=lambda row: row["flexible_recovery_gain_over_fixed_pct"])
    worst_greedy = max(rows, key=lambda row: row["greedy_gap_to_fixed_oracle_pct"])
    radio = [sionna_radio_lever(channel, loss) for loss in protocol["radio_loss_sensitivity"]]
    gates = {
        "all_c174_candidates_evaluated": len(rows) == protocol["expected_candidate_points"],
        "fixed_oracle_reproduced": max(match_errors, default=0.0) < 1e-6,
        "recovery_first_reproduced": max(recovery_errors, default=0.0) < 1e-6,
        "greedy_never_exceeds_oracle": all(
            row["lever_values"]["ai_first_greedy"]
            <= row["lever_values"]["fixed_placement_oracle"] + 1e-8 for row in rows
        ),
        "flexible_placement_never_below_fixed": all(
            row["lever_values"]["flexible_recovery_oracle"] + 1e-8
            >= row["lever_values"]["fixed_placement_oracle"] for row in rows
        ),
        "sionna_holdout_source_passed": channel["all_pass"] is True,
    }
    value = {
        "schema": "softwall-c175-lever-evaluation-v1",
        "status": "C175_LEVER_EVALUATION_PASS" if all(gates.values()) else "C175_LEVER_EVALUATION_FAIL",
        "all_pass": all(gates.values()),
        "analysis_role": (
            "Development-only decomposition of the 362 C174 points that passed its "
            "frozen 5% screen. Results select no new trace row and make no "
            "confirmatory, physical-throughput, or production claim."
        ),
        "gates": gates,
        "summary": {
            "candidate_points": len(rows),
            "greedy_within_5pct_of_fixed_oracle_points": sum(
                row["greedy_gap_to_fixed_oracle_pct"] <= 5.0 for row in rows
            ),
            "flexible_recovery_positive_gain_points": sum(
                row["flexible_recovery_gain_over_fixed_pct"] > 1e-9 for row in rows
            ),
            "maximum_flexible_recovery_gain_row": best_placement,
            "maximum_greedy_gap_row": worst_greedy,
            "sionna_debt_aware": radio,
        },
        "rows": rows,
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": observed,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, separators=(",", ":")) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps({"status": value["status"], "summary": value["summary"]}, indent=2))
    if not value["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
