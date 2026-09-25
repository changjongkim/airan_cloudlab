#!/usr/bin/env python3.11
"""Exploratory staged PHY-first baselines for the frozen seven-AI screen."""

from __future__ import annotations

import argparse
import json
import statistics
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
        min_cardinality = min(len(plan["selected_nrx"]) for plan in feasible)
        minimal = max(
            (plan for plan in feasible
             if len(plan["selected_nrx"]) == min_cardinality),
            key=lambda plan: (round(plan["radio_gain"], 12),
                              round(plan["expected_ai_value"], 12),
                              tuple(plan["selected_nrx"])),
        )
        max_radio = max(
            feasible,
            key=lambda plan: (round(plan["radio_gain"], 12),
                              round(plan["expected_ai_value"], 12),
                              tuple(plan["selected_nrx"])),
        )
        joint = next(plan for plan in feasible
                     if plan["selected_nrx"] == original["exact_selected_nrx"])
        if abs(joint["expected_ai_value"] - original["exact_ai"]) > 1e-9:
            raise AssertionError("frozen joint value did not reproduce")
        rows.append({
            "case": original["case"], "stratum": original["stratum"],
            "radio_floor": original["radio_floor"],
            "joint_selected": joint["selected_nrx"],
            "joint_ai": joint["expected_ai_value"],
            "joint_radio": joint["radio_gain"],
            "minimal_selected": minimal["selected_nrx"],
            "minimal_ai": minimal["expected_ai_value"],
            "minimal_radio": minimal["radio_gain"],
            "max_radio_selected": max_radio["selected_nrx"],
            "max_radio_ai": max_radio["expected_ai_value"],
            "max_radio_gain": max_radio["radio_gain"],
            "joint_minus_minimal_ai":
                joint["expected_ai_value"] - minimal["expected_ai_value"],
            "joint_minus_minimal_radio":
                joint["radio_gain"] - minimal["radio_gain"],
            "joint_minus_max_radio_ai":
                joint["expected_ai_value"] - max_radio["expected_ai_value"],
            "joint_minus_max_radio":
                joint["radio_gain"] - max_radio["radio_gain"],
        })
    summaries = {}
    for stratum in protocol["strata"]:
        subset = [row for row in rows if row["stratum"] == stratum["name"]]
        summaries[stratum["name"]] = {
            "feasible": len(subset),
            "joint_differs_from_minimal": sum(
                row["joint_selected"] != row["minimal_selected"]
                for row in subset),
            "joint_ai_above_minimal": sum(
                row["joint_minus_minimal_ai"] > 1e-9 for row in subset),
            "joint_ai_above_minimal_without_lower_radio": sum(
                row["joint_minus_minimal_ai"] > 1e-9
                and row["joint_minus_minimal_radio"] >= -1e-12
                for row in subset),
            "mean_joint_minus_minimal_ai": statistics.mean(
                row["joint_minus_minimal_ai"] for row in subset),
            "mean_joint_minus_minimal_radio": statistics.mean(
                row["joint_minus_minimal_radio"] for row in subset),
            "joint_ai_above_max_radio": sum(
                row["joint_minus_max_radio_ai"] > 1e-9 for row in subset),
            "joint_ai_above_max_radio_within_0p01_radio": sum(
                row["joint_minus_max_radio_ai"] > 1e-9
                and row["joint_minus_max_radio"] >= -0.01
                for row in subset),
        }
    report = {
        "schema": "softwall-confirm93-decomposed-baselines-posthoc-v1",
        "frozen_screen_unchanged": True,
        "staged_minimal_definition": "Among all-safe NRx subsets meeting the same rescue floor, use the fewest NRx endpoints; maximize q rescue within that cardinality, then use exact visible AI recourse and recovery scheduling.",
        "staged_max_radio_definition": "Among all-safe NRx subsets meeting the same rescue floor, maximize q rescue; tie-break by exact visible AI recourse. This trades AI for more radio quality and is reported separately.",
        "joint_definition": "Frozen Confirm93 exact AI-first plan subject to the same radio floor and safety evaluator; joint 1-swap greedy matched it in every feasible case.",
        "summaries": summaries, "rows": rows,
        "interpretation": "Adaptive posthoc comparator screen on already-used PHY samples. It tests whether coupling NRx subset choice to conditional AI value can beat strong staged PHY-first choices while preserving the stated q floor and exact AI recourse. Positive differences are hypotheses only; a new independent trace, actual radio noninferiority and paired GPU execution are required. Giving the staged baseline joint AI-aware NRx reselection would make it the already-evaluated joint greedy method, not a decomposed comparator.",
    }
    out = base / "confirm93_decomposed_baselines_posthoc.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
