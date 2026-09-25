#!/usr/bin/env python3.11
"""Reproducible exploratory screen for a joint PHY/recovery/AI decision gap."""

from __future__ import annotations

import argparse
import dataclasses
import json
import random
from pathlib import Path

from offline_contingent_phy_ai import (
    AiUnit, Cell, Problem, exact_joint_policy, greedy_radio_then_ai,
)


def sample_problem(rng: random.Random) -> Problem:
    cells = []
    for index in range(4):
        success = rng.choice((0.35, 0.5, 0.65, 0.8))
        rescue = round(success * rng.choice((0.25, 0.5, 0.75)), 3)
        cells.append(Cell(
            chr(ord("A") + index), rng.randint(38, 65),
            rng.randint(7, 12), rng.randint(11, 26), success, rescue,
        ))
    ai = tuple(AiUnit(
        f"ai{index}", rng.randint(25, 60), rng.randint(7, 14),
        rng.randint(8, 18), rng.randint(1, 3),
    ) for index in range(2))
    return Problem(tuple(cells), ai, rng.choice((0.2, 0.3, 0.4, 0.5)), 2)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    rng = random.Random(protocol["seed"])
    records = []
    feasible = 0
    for index in range(protocol["cases"]):
        problem = sample_problem(rng)
        exact = exact_joint_policy(problem)
        greedy = greedy_radio_then_ai(problem)
        if exact is None:
            if greedy is not None:
                raise AssertionError("greedy cannot be feasible when exact is infeasible")
            continue
        feasible += 1
        if greedy is None:
            raise AssertionError("feasible exact plan missing in local greedy")
        if exact["all_fail_recovery_certificate"] is None or greedy["all_fail_recovery_certificate"] is None:
            raise AssertionError("missing all-fail recovery certificate")
        if exact["expected_ai_value"] + 1e-9 < greedy["expected_ai_value"]:
            raise AssertionError("exact policy below local greedy")
        gap = exact["expected_ai_value"] - greedy["expected_ai_value"]
        if gap > 1e-9:
            records.append({
                "case_index": index,
                "problem": {
                    "cells": [dataclasses.asdict(cell) for cell in problem.cells],
                    "ai": [dataclasses.asdict(unit) for unit in problem.ai],
                    "min_expected_radio_rescues": problem.min_expected_radio_rescues,
                    "nrx_endpoint_capacity": problem.nrx_endpoint_capacity,
                },
                "exact_selected_nrx": exact["selected_nrx"],
                "greedy_selected_nrx": greedy["selected_nrx"],
                "exact_expected_ai_value": exact["expected_ai_value"],
                "greedy_expected_ai_value": greedy["expected_ai_value"],
                "gap": gap,
            })
    records.sort(key=lambda item: (-item["gap"], item["case_index"]))
    report = {
        "schema": "softwall-offline-contingent-phy-ai-screen-result-v1",
        "protocol": str(args.protocol),
        "screened": protocol["cases"],
        "feasible": feasible,
        "positive_gap_cases": len(records),
        "maximum_gap": max((item["gap"] for item in records), default=0),
        "positive_cases": records,
        "interpretation": "Exploratory two-stage synthetic screen only. Positive gaps against a one-exchange local greedy can be generic local-search gaps; they do not establish an AI-RAN novelty or a physical throughput gain.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in
                      ("screened", "feasible", "positive_gap_cases", "maximum_gap")}))


if __name__ == "__main__":
    main()
