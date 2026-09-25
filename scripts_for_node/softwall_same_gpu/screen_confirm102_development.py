#!/usr/bin/env python3.11
"""Development-only v4 screen on the already-used Confirm94 PHY trace.

This script is allowed to tune the prospective Confirm102 protocol, so its
output is exploratory evidence only.  The sealed Confirm102 records must not
be read by this script.
"""

from __future__ import annotations

import argparse
import json
import random
import statistics
from pathlib import Path

from mixed_snr_phy_value_model import TrainOnlyMixedSnrPhyValueModel
from cv_grid_phy_value_model import TrainOnlyCvGridPhyValueModel
from offline_online_recovery_v4 import (
    AiJob, Cell, Config, branch_expectation, exact_joint_policy,
    evaluate_subset, feasible_plans, joint_one_swap_policy, max_radio_policy,
    radio_guarded_joint_policy, radio_guarded_one_swap_policy,
    replay_branch, staged_min_policy,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def actual_correct(rows: list[dict], selected: list[str]) -> int:
    chosen = set(selected)
    return sum(bool(row["conventional_correct"] or
                    (chr(97 + i) in chosen and row["neural_correct"]))
               for i, row in enumerate(rows))


def actual_ai(cells: tuple[Cell, ...], selected: list[str], rows: list[dict],
              jobs: tuple[AiJob, ...], config: Config) -> int:
    chosen = frozenset(selected)
    outcome = {name: bool(rows[ord(name) - 97]["neural_correct"])
               for name in chosen}
    replay = replay_branch(cells, chosen, outcome, jobs, config)
    if not replay["safe"]:
        raise AssertionError("selected plan lost mandatory safety")
    return replay["completed_ai_value"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", choices=("legacy", "cv_grid"), default="legacy")
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    p94 = read(base / "confirm94_independent_phy_protocol.json")
    raw = read(base / "raw" / f"{p94['trace_prefix']}_test.json")
    model = (TrainOnlyMixedSnrPhyValueModel(base / p94["q_model"], root)
             if args.model == "legacy" else
             TrainOnlyCvGridPhyValueModel(
                 base / "confirm102_train_only_cv_grid_model.json", root))
    low_threshold = read(base / "confirm60_low_gate_train_selection.json")["threshold"]
    pool = [row for result in raw["results"] for row in result["records"]]

    # Admission bounds are the C100 observe-first declarations.  Replay
    # durations are rounded upward from C100 maxima: NRx 14.562 -> 15,
    # conventional 5.251 -> 6, while AI uses 22 ms near its two-arm median.
    config = Config(153, 15, 45, 12, 6, 2)
    jobs = tuple(AiJob(
        f"ai{i}", 0 if i < 5 else (60 if i == 5 else 90),
        153, 50, 22, 1,
    ) for i in range(7))
    visible = tuple(job for job in jobs if job.release_ms == 0)
    rng = random.Random(p94["sampling_seed"])
    records = []
    for case in range(100):
        rows = rng.sample(pool, 4)
        estimates = [model.lookup(
            row["observed_features"]["channel_estimate_power"],
            row["observed_features"]["received_grid_power"],
        ) for row in rows]
        floor = rng.choice((0.1, 0.2))
        if any(value is None for value in estimates):
            records.append({"case": case, "status": "out_of_support"})
            continue
        cells = tuple(Cell(chr(97 + i), value.p_neural_success, value.q_rescue)
                      for i, value in enumerate(estimates))
        exact = exact_joint_policy(cells, visible, config, floor)
        greedy = joint_one_swap_policy(cells, visible, config, floor)
        staged = staged_min_policy(cells, visible, config, floor)
        max_radio = max_radio_policy(cells, visible, config, floor)
        if any(plan is None for plan in (exact, greedy, staged, max_radio)):
            records.append({"case": case, "status": "infeasible"})
            continue

        eligible = [i for i, row in enumerate(rows)
                    if row["observed_features"]["channel_estimate_power"] >= low_threshold]
        eligible.sort(key=lambda i: (-cells[i].q_incremental_rescue, cells[i].name))
        low_names = frozenset(cells[i].name for i in eligible[:2])
        low = evaluate_subset(cells, low_names, visible, config, 0.0)
        if low is None:
            raise AssertionError("fixed low gate lost all-fail feasibility")

        plans = feasible_plans(cells, visible, config, floor)
        forced = [evaluate_subset(cells, frozenset(plan["selected_nrx"]),
                                  visible, config, floor,
                                  force_all_mandatory=True)
                  for plan in plans]
        forced_values = [plan["expected_visible_ai_value"]
                         for plan in forced if plan is not None]

        guarded = {
            tolerance: radio_guarded_joint_policy(
                cells, visible, config, floor, tolerance)
            for tolerance in (0.0, 0.005, 0.01, 0.02, 0.03, 0.05)
        }
        guarded_greedy = {
            tolerance: radio_guarded_one_swap_policy(
                cells, visible, config, floor, tolerance)
            for tolerance in guarded
        }
        if any(plan is None for plan in (*guarded.values(), *guarded_greedy.values())):
            raise AssertionError("radio-guarded policy became infeasible")
        policies = {
            "joint_exact": exact, "joint_greedy": greedy,
            "staged_min": staged, "max_radio": max_radio,
            "low_gate": low,
        }
        for tolerance, plan in guarded.items():
            policies[f"guarded_{tolerance:g}"] = plan
            policies[f"guarded_greedy_{tolerance:g}"] = guarded_greedy[tolerance]
        entry = {
            "case": case, "status": "feasible", "radio_floor": floor,
            "conditional_capacity_span": (
                max(plan["expected_visible_ai_value"] for plan in plans)
                - min(plan["expected_visible_ai_value"] for plan in plans)
            ),
            "always_mandatory_capacity_span": max(forced_values) - min(forced_values),
            "policies": {},
        }
        for name, plan in policies.items():
            selected = plan["selected_nrx"]
            expected_full = branch_expectation(
                cells, frozenset(selected), jobs, config
            )["expected_ai_value"]
            entry["policies"][name] = {
                "selected": selected,
                "expected_visible_ai": plan["expected_visible_ai_value"],
                "expected_full_ai": expected_full,
                "radio_gain": plan["radio_gain"],
                "actual_ai": actual_ai(cells, selected, rows, jobs, config),
                "actual_correct": actual_correct(rows, selected),
            }
        records.append(entry)

    feasible = [row for row in records if row["status"] == "feasible"]
    names = tuple(sorted(feasible[0]["policies"]))
    names = tuple(name for name in names if name != "joint_exact")
    comparisons = {}
    for name in names:
        diffs = [row["policies"]["joint_exact"]["actual_ai"]
                 - row["policies"][name]["actual_ai"] for row in feasible]
        expected = [row["policies"]["joint_exact"]["expected_full_ai"]
                    - row["policies"][name]["expected_full_ai"] for row in feasible]
        radio = [row["policies"]["joint_exact"]["radio_gain"]
                 - row["policies"][name]["radio_gain"] for row in feasible]
        comparisons[name] = {
            "different_selection": sum(
                row["policies"]["joint_exact"]["selected"]
                != row["policies"][name]["selected"] for row in feasible
            ),
            "actual_ai_wins_ties_losses": [
                sum(value > 0 for value in diffs),
                sum(value == 0 for value in diffs),
                sum(value < 0 for value in diffs),
            ],
            "actual_ai_sum_difference": sum(diffs),
            "mean_expected_full_ai_difference": statistics.mean(expected),
            "mean_radio_gain_difference": statistics.mean(radio),
            "actual_correct_difference": sum(
                row["policies"]["joint_exact"]["actual_correct"]
                - row["policies"][name]["actual_correct"] for row in feasible
            ),
        }
    report = {
        "schema": "softwall-confirm102-v4-development-screen-v2",
        "scope": "Exploratory tuning on already-used Confirm94 PHY records; not independent evidence and not a physical GPU result.",
        "phy_model": args.model,
        "config": config.__dict__,
        "ai_jobs": [job.__dict__ for job in jobs],
        "visible_at_radio_decision": [job.name for job in visible],
        "future_arrivals_hidden_from_radio_decision": [
            job.name for job in jobs if job.release_ms > 0
        ],
        "cases": len(records), "feasible": len(feasible),
        "conditional_capacity_nonconstant": sum(
            row["conditional_capacity_span"] > 1e-9 for row in feasible),
        "always_mandatory_ablation_zero_span": sum(
            abs(row["always_mandatory_capacity_span"]) <= 1e-9
            for row in feasible
        ),
        "comparisons": comparisons,
        "records": records,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "records"}, indent=2))


if __name__ == "__main__":
    main()
