#!/usr/bin/env python3.11
"""Posthoc: distinguish no choice pressure from a sufficient greedy search."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from offline_contingent_phy_ai_v3 import (
    AiUnit, Cell, Problem, all_nrx_subsets, optimize_for_nrx_subset,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / "results/softwall_same_gpu"
    protocol = read(base / "confirm93_seven_ai_capacity_protocol.json")
    screen = read(base / "confirm93_seven_ai_capacity_screen.json")
    model = read(base / protocol["q_model"])
    leaves = {leaf["index"]: leaf for leaf in model["leaves"]}
    rows = []
    for original in screen["records"]:
        if original["status"] != "feasible":
            continue
        cells = tuple(
            Cell(chr(ord("a") + index), 153, 12, 30,
                 leaves[leaf_index]["p_neural_success_smoothed"],
                 leaves[leaf_index]["q_rescue_smoothed"])
            for index, leaf_index in enumerate(original["leaf_indices"])
        )
        ai = tuple(
            AiUnit(f"ai{index}", deadline, 15, 15, value)
            for index, (deadline, value) in enumerate(zip(
                original["ai_deadlines_ms"], original["ai_values"]
            ))
        )
        problem = Problem(cells, ai, original["radio_floor"], 2)
        plans = [optimize_for_nrx_subset(problem, subset)
                 for subset in all_nrx_subsets(problem)]
        feasible = [plan for plan in plans if plan is not None]
        values = [round(plan["expected_ai_value"], 12) for plan in feasible]
        rows.append({
            "case": original["case"], "stratum": original["stratum"],
            "feasible_nrx_subsets": len(feasible),
            "distinct_expected_ai_values": len(set(values)),
            "min_expected_ai": min(values),
            "max_expected_ai": max(values),
            "expected_ai_range": max(values) - min(values),
            "exact_selected_nrx": original["exact_selected_nrx"],
            "greedy_selected_nrx": original["greedy_selected_nrx"],
        })
    summaries = {}
    for stratum in protocol["strata"]:
        subset = [row for row in rows if row["stratum"] == stratum["name"]]
        summaries[stratum["name"]] = {
            "feasible_cases": len(subset),
            "cases_with_multiple_feasible_nrx_subsets": sum(
                row["feasible_nrx_subsets"] > 1 for row in subset),
            "cases_with_distinct_ai_values": sum(
                row["distinct_expected_ai_values"] > 1 for row in subset),
            "max_expected_ai_range": max(
                (row["expected_ai_range"] for row in subset), default=None),
        }
    report = {
        "schema": "softwall-confirm93-choice-surface-posthoc-v1",
        "frozen_screen_unchanged": True,
        "summaries": summaries, "rows": rows,
        "interpretation": "Re-evaluates every feasible NRx subset after the frozen screen. Multiple expected AI values show real conditional choice pressure in the abstract model; they do not establish GPU throughput or PHY-specific algorithmic novelty. This posthoc audit cannot turn a frozen zero exact-greedy gap into a positive result.",
    }
    out = base / "confirm93_choice_surface_posthoc.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
