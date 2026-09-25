#!/usr/bin/env python3
"""Post-hoc structural decomposition of Confirm102 feasibility.

This does not change the frozen C102 outcome.  It separates a radio-floor
failure from a scheduling-safety failure and measures whether the feasible
sets contain a radio-noninferior AI tradeoff.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from cv_grid_phy_value_model import TrainOnlyCvGridPhyValueModel
from offline_online_recovery_v4 import (
    AiJob, Cell, Config, evaluate_subset, feasible_plans, subsets,
)


EPS = 1e-9


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm102_multi_event_policy_protocol.json")
    trace = read(base / protocol["trace"])
    model = TrainOnlyCvGridPhyValueModel(base / protocol["q_model"], root)
    pool = [row for group in trace["results"] for row in group["records"]]

    timing = protocol["timing"]
    config = Config(
        timing["radio_deadline_ms"], timing["nrx_observe_replay_ms"],
        timing["nrx_admission_bound_ms"], timing["conv_admission_bound_ms"],
        timing["conv_replay_ms"], protocol["endpoint_capacity"],
    )
    jobs = tuple(AiJob(**row) for row in protocol["ai_jobs"])
    visible = tuple(job for job in jobs if job.release_ms <= 0)
    rng = random.Random(protocol["sampling_seed"])
    cases = []
    for case in range(protocol["cases"]):
        rows = rng.sample(pool, protocol["cells"])
        estimates = [model.lookup(
            row["observed_features"]["channel_estimate_power"],
            row["observed_features"]["received_grid_power"],
        ) for row in rows]
        floor = rng.choice(protocol["radio_floor_choices"])
        if any(value is None for value in estimates):
            cases.append({"case": case, "status": "out_of_support"})
            continue
        cells = tuple(Cell(chr(97 + i), value.p_neural_success, value.q_rescue)
                      for i, value in enumerate(estimates))
        candidates = []
        for chosen in subsets(cells, config.endpoint_capacity):
            radio_gain = sum(cell.q_incremental_rescue for cell in cells
                             if cell.name in chosen)
            plan_without_floor = evaluate_subset(
                cells, chosen, visible, config, 0.0
            )
            candidates.append({
                "selected": sorted(chosen),
                "radio_gain": radio_gain,
                "safe": plan_without_floor is not None,
                "expected_ai": (
                    plan_without_floor["expected_visible_ai_value"]
                    if plan_without_floor is not None else None
                ),
            })
        max_radio_any = max(row["radio_gain"] for row in candidates)
        safe = [row for row in candidates if row["safe"]]
        max_radio_safe = max((row["radio_gain"] for row in safe), default=None)
        floor_plans = [row for row in safe if row["radio_gain"] + EPS >= floor]
        if max_radio_any + EPS < floor:
            status = "radio_floor_unreachable"
        elif not safe:
            status = "no_safe_subset"
        elif max_radio_safe + EPS < floor:
            status = "safe_subsets_below_radio_floor"
        else:
            status = "feasible"

        entry = {
            "case": case, "radio_floor": floor, "status": status,
            "max_radio_any": max_radio_any,
            "max_radio_safe": max_radio_safe,
            "safe_subset_count": len(safe),
            "floor_feasible_subset_count": len(floor_plans),
        }
        if floor_plans:
            max_radio = max(floor_plans, key=lambda row: (
                round(row["radio_gain"], 12), round(row["expected_ai"], 12),
                -len(row["selected"]), row["selected"],
            ))
            guard_floor = max(
                floor,
                max_radio["radio_gain"] - protocol["maximum_predicted_radio_loss"],
            )
            guarded = [row for row in floor_plans
                       if row["radio_gain"] + EPS >= guard_floor]
            best_guarded_ai = max(row["expected_ai"] for row in guarded)
            any_ai_tradeoff = max(row["expected_ai"] for row in floor_plans)
            entry.update({
                "max_radio_selection": max_radio["selected"],
                "max_radio_expected_ai": max_radio["expected_ai"],
                "guarded_subset_count": len(guarded),
                "guarded_ai_improvement_available": (
                    best_guarded_ai > max_radio["expected_ai"] + EPS
                ),
                "outside_guard_ai_improvement_available": (
                    any_ai_tradeoff > best_guarded_ai + EPS
                ),
                "conditional_capacity_nonconstant": (
                    max(row["expected_ai"] for row in floor_plans)
                    - min(row["expected_ai"] for row in floor_plans) > EPS
                ),
            })
        cases.append(entry)

    counts = {}
    for row in cases:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    feasible = [row for row in cases if row["status"] == "feasible"]
    result = {
        "schema": "softwall-confirm102-feasibility-posthoc-v1",
        "status_counts": counts,
        "feasible_cases": len(feasible),
        "feasible_with_nonconstant_conditional_capacity": sum(
            row["conditional_capacity_nonconstant"] for row in feasible
        ),
        "feasible_with_multiple_guarded_subsets": sum(
            row["guarded_subset_count"] > 1 for row in feasible
        ),
        "feasible_with_guarded_ai_improvement_over_max_radio": sum(
            row["guarded_ai_improvement_available"] for row in feasible
        ),
        "feasible_with_ai_improvement_only_outside_radio_guard": sum(
            row["outside_guard_ai_improvement_available"] for row in feasible
        ),
        "interpretation": (
            "Post-hoc decomposition of the frozen C102 cases. Infeasibility "
            "caused by an unreachable predicted-radio floor is distinct from "
            "an all-fail scheduling failure. A nonconstant capacity span does "
            "not itself imply a radio-noninferior policy opportunity."
        ),
        "cases": cases,
    }
    output = base / "confirm102_feasibility_posthoc.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"},
                     indent=2))


if __name__ == "__main__":
    main()
