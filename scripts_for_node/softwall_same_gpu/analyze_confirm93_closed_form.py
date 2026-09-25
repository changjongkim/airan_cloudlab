#!/usr/bin/env python3.11
"""Posthoc audit of the seven-AI screen's collapsed branch-value formula."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
    probability = {
        leaf["index"]: leaf["p_neural_success_smoothed"]
        for leaf in model["leaves"]
    }
    rows = []
    for original in screen["records"]:
        if original["status"] != "feasible":
            continue
        selected = original["exact_selected_nrx"]
        all_fail_probability = math.prod(
            1 - probability[original["leaf_indices"][ord(name) - ord("a")]]
            for name in selected
        )
        predicted = (sum(original["ai_values"])
                     - min(original["ai_values"]) * all_fail_probability)
        rows.append({
            "case": original["case"], "stratum": original["stratum"],
            "selected_nrx": selected,
            "actual_exact_ai": original["exact_ai"],
            "closed_form_ai": predicted,
            "absolute_residual": abs(predicted - original["exact_ai"]),
        })
    report = {
        "schema": "softwall-confirm93-closed-form-posthoc-v1",
        "frozen_screen_unchanged": True,
        "formula": "sum(AI values) - min(AI value) * product(1 - p_neural_success_i for selected NRx i)",
        "scope": "Observed for the exact-policy choice in all feasible Confirm93 CPU cases, under the model's independent NRx outcomes and all-visible t0 seven-AI workload. The formula is not asserted for arbitrary arrivals, deadlines, correlated PHY outcomes, or physical GPU timing.",
        "feasible_cases": len(rows),
        "matches_within_1e_minus_9": sum(
            row["absolute_residual"] < 1e-9 for row in rows),
        "max_absolute_residual": max(
            (row["absolute_residual"] for row in rows), default=None),
        "rows": rows,
        "interpretation": "The screened recourse has collapsed to whether any selected NRx succeeds: all seven AI values fit on success, while an all-fail branch loses the least-valued unit. This makes the objective a simple success-union probability subject to the radio rescue floor. It explains why strong local greedy can match exact here; it does not prove global greedy optimality in other models. This posthoc explanation cannot revise the frozen zero-gap result.",
    }
    out = base / "confirm93_closed_form_posthoc.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("feasible_cases", "matches_within_1e_minus_9",
                       "max_absolute_residual")}, indent=2))


if __name__ == "__main__":
    main()
