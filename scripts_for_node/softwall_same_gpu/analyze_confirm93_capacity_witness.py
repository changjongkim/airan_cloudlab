#!/usr/bin/env python3.11
"""Enumerate the seven-AI all-fail versus success capacity example."""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from offline_contingent_phy_ai_v3 import feasible_schedule
from offline_recovery_exchange import Job


def enumerate_plans(ai_count: int, succeeded: bool) -> list[dict]:
    selected = {"a", "b"}
    unselected = {"c", "d"}
    all_cells = selected | unselected
    plans = []
    for conv_count in range(len(unselected) + 1):
        for early_conventional in itertools.combinations(sorted(unselected), conv_count):
            for early_ai in (None, *range(ai_count)):
                early_jobs = tuple(
                    Job(f"conv{cell}", 0, 12, 30)
                    for cell in early_conventional
                ) + (() if early_ai is None else
                     (Job(f"ai{early_ai}", 0, 15, 30, 1),))
                early_schedule = feasible_schedule(0, early_jobs)
                if early_schedule is None:
                    continue
                pending_radio = (
                    unselected if succeeded else all_cells
                ) - set(early_conventional)
                pending = tuple(
                    Job(f"conv{cell}", 30, 12, 153)
                    for cell in sorted(pending_radio)
                ) + tuple(
                    Job(f"ai{i}", 30, 15, 153, 1)
                    for i in range(ai_count) if i != early_ai
                )
                late_schedule = feasible_schedule(30, pending)
                plans.append({
                    "early_conventional": list(early_conventional),
                    "early_ai": early_ai,
                    "early_schedule": early_schedule,
                    "late_schedule": late_schedule,
                    "feasible": late_schedule is not None,
                    "last_finish_ms": (late_schedule[-1]["finish_ms"]
                                       if late_schedule else None),
                })
    return plans


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / "results/softwall_same_gpu"
    scenarios = {}
    for ai_count in (6, 7):
        for succeeded in (False, True):
            key = f"ai{ai_count}_{'two_nrx_success' if succeeded else 'all_fail'}"
            plans = enumerate_plans(ai_count, succeeded)
            feasible = [plan for plan in plans if plan["feasible"]]
            best = min(feasible, key=lambda plan: plan["last_finish_ms"],
                       default=None)
            scenarios[key] = {
                "enumerated_early_plans": len(plans),
                "feasible_early_plans": len(feasible),
                "best_schedule": best,
            }
    report = {
        "schema": "softwall-confirm93-conditional-capacity-witness-v1",
        "assumptions": {
            "radio_cells": 4, "selected_nrx": ["a", "b"],
            "nrx_result_event_ms": 30, "conventional_bound_ms": 12,
            "ai_bound_ms": 15, "ai_early_cap": 1,
            "single_serial_lane": True,
            "radio_and_ai_deadline_after_guard_ms": 153,
            "all_ai_visible_at_t0": True,
        },
        "scenarios": scenarios,
        "interpretation": "Exhaustive early placement and exact serial-lane scheduling in an abstract two-stage mode. Six AI units fit the all-fail branch; seven do not, but seven fit after both selected NRx successes. This establishes conditional decision pressure under model assumptions, not physical seven-unit burst qualification, online policy advantage, or hard WCET.",
    }
    out = base / "confirm93_conditional_capacity_witness.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: {"feasible_early_plans": value["feasible_early_plans"],
                            "best_finish_ms": value["best_schedule"]["last_finish_ms"]
                                if value["best_schedule"] else None}
                      for key, value in scenarios.items()}, indent=2))


if __name__ == "__main__":
    main()
