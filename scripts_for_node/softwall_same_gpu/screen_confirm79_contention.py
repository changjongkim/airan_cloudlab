#!/usr/bin/env python3.11
"""Predeclared exploratory contention screen with held-out PHY features."""

from __future__ import annotations

import hashlib
import json
import os
import random
import statistics
from pathlib import Path

from mixed_snr_phy_value_model import TrainOnlyMixedSnrPhyValueModel
from offline_contingent_phy_ai import (
    AiUnit, Cell, Problem, exact_joint_policy, greedy_radio_then_ai,
)


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"


def main() -> None:
    protocol_name = os.environ.get(
        "SOFTWALL_SCREEN_PROTOCOL", "confirm79_contention_protocol.json",
    )
    protocol = json.loads((BASE / protocol_name).read_text())
    for path, digest in protocol["source_sha256_before_run"].items():
        if hashlib.sha256((ROOT / path).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"frozen source changed: {path}")
    model = TrainOnlyMixedSnrPhyValueModel(
        BASE / "confirm77_train_only_q_model.json", ROOT,
    )
    raw = json.loads((BASE / "raw" / protocol["heldout_raw"]).read_text())
    if raw["seed"] != protocol["heldout_seed"]:
        raise ValueError("held-out trace mismatch")
    pool = [row for result in raw["results"] for row in result["records"]]
    rng = random.Random(protocol["sampling_seed"])
    records = []
    for index in range(protocol["cases"]):
        chosen = rng.sample(pool, protocol["cells"])
        values = [model.lookup(
            row["observed_features"]["channel_estimate_power"],
            row["observed_features"]["received_grid_power"],
        ) for row in chosen]
        deadlines = [rng.randint(*protocol["radio_deadline_range_ms"])
                     for _ in chosen]
        ai_count = rng.choice(protocol["ai_count_choices"])
        ai_durations = [rng.choice(protocol["ai_durations_ms"])
                        for _ in range(ai_count)]
        ai_deadlines = [rng.randint(*protocol["ai_deadline_range_ms"])
                        for _ in range(ai_count)]
        ai_values = [rng.randint(*protocol["ai_value_range"])
                     for _ in range(ai_count)]
        radio_floor = rng.choice(protocol["radio_floor_choices"])
        row = {"index": index, "status": "out_of_support"}
        if any(value is None for value in values):
            records.append(row)
            continue
        cells = tuple(Cell(
            chr(ord("a") + i), deadlines[i] - protocol["commit_guard_ms"],
            protocol["conventional_bound_ms"], protocol["nrx_ready_ms"],
            value.p_neural_success, value.q_rescue,
        ) for i, value in enumerate(values))
        ai = tuple(AiUnit(f"ai{i}", ai_deadlines[i], ai_durations[i],
                          ai_durations[i], ai_values[i])
                   for i in range(ai_count))
        problem = Problem(cells, ai, radio_floor, protocol["nrx_endpoint_capacity"])
        exact = exact_joint_policy(problem)
        greedy = greedy_radio_then_ai(problem)
        if (exact is None) != (greedy is None):
            raise AssertionError("exact and greedy feasibility disagree")
        if exact is None:
            row["status"] = "infeasible"
        else:
            if exact["expected_ai_value"] + 1e-9 < greedy["expected_ai_value"]:
                raise AssertionError("exact below greedy")
            row.update({
                "status": "feasible",
                "leaf_indices": [value.leaf_index for value in values],
                "radio_deadlines_ms": deadlines,
                "ai_deadlines_ms": ai_deadlines,
                "ai_durations_ms": ai_durations,
                "ai_values": ai_values,
                "radio_floor": radio_floor,
                "exact_selected_nrx": exact["selected_nrx"],
                "greedy_selected_nrx": greedy["selected_nrx"],
                "exact_expected_ai_value": exact["expected_ai_value"],
                "greedy_expected_ai_value": greedy["expected_ai_value"],
                "exact_radio_gain": exact["radio_gain"],
                "greedy_radio_gain": greedy["radio_gain"],
                "exact_all_fail_certificate": exact["all_fail_recovery_certificate"],
                "greedy_all_fail_certificate": greedy["all_fail_recovery_certificate"],
            })
        records.append(row)
    feasible = [row for row in records if row["status"] == "feasible"]
    gaps = [row["exact_expected_ai_value"] - row["greedy_expected_ai_value"]
            for row in feasible]
    report = {
        "schema": "softwall-confirm79-exploratory-contention-screen-v1",
        "total": len(records), "feasible": len(feasible),
        "out_of_support": sum(row["status"] == "out_of_support" for row in records),
        "positive_expected_ai_gap": sum(gap > 1e-9 for gap in gaps),
        "different_nrx_subset": sum(row["exact_selected_nrx"] != row["greedy_selected_nrx"]
                                    for row in feasible),
        "mean_exact_ai": statistics.mean(row["exact_expected_ai_value"] for row in feasible),
        "mean_greedy_ai": statistics.mean(row["greedy_expected_ai_value"] for row in feasible),
        "maximum_gap": max(gaps),
        "records": records,
        "interpretation": (
            "Predeclared exploratory CPU screen using actual held-out synthetic "
            "PHY features and random hypothetical timing/AI requests. No GPU "
            "execution or publication-level blind test. Any exact-local gap may "
            "be generic local-search behavior and requires an equal-radio control."
        ),
    }
    output = BASE / protocol.get("output", "confirm79_contention_screen.json")
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "total", "feasible", "out_of_support", "positive_expected_ai_gap",
        "different_nrx_subset", "mean_exact_ai", "mean_greedy_ai",
        "maximum_gap")}, indent=2))


if __name__ == "__main__":
    main()
