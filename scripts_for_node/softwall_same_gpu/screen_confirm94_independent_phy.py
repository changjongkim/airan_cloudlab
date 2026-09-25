#!/usr/bin/env python3.11
"""Prospective independent-PHY CPU screen of joint versus staged policies."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import socket
import statistics
from pathlib import Path

from mixed_snr_phy_value_model import TrainOnlyMixedSnrPhyValueModel
from offline_contingent_phy_ai_v3 import (
    AiUnit, Cell, Problem, all_nrx_subsets, greedy_radio_then_ai,
    optimize_for_nrx_subset, policy_score,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def realized_correct(rows: list[dict], selected: list[str]) -> int:
    names = set(selected)
    return sum(
        bool(row["conventional_correct"]
             or (chr(ord("a") + index) in names and row["neural_correct"]))
        for index, row in enumerate(rows)
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm94_independent_phy_protocol.json")
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    if not source_ok:
        raise RuntimeError("frozen source changed")
    physical = read(base / protocol["physical_qualification"])
    if not physical["all_pass"]:
        raise RuntimeError("physical duration candidate did not qualify")
    prefix = protocol["trace_prefix"]
    trace_path = base / "raw" / f"{prefix}_test.json"
    manifest = read(base / "raw" / f"{prefix}_manifest.json")
    raw = read(trace_path)
    if (raw["seed"] != protocol["phy_seed"]
            or manifest["seed"] != protocol["phy_seed"]
            or manifest["trials_per_snr"] != protocol["trials_per_snr"]
            or manifest["snrs_db"] != protocol["snrs_db"]
            or not raw["observed_features_recorded"]
            or not raw["observed_features_prewarmed"]):
        raise RuntimeError("independent PHY trace mismatch")
    model = TrainOnlyMixedSnrPhyValueModel(
        base / protocol["q_model"], root,
    )
    pool = [row for result in raw["results"] for row in result["records"]]
    rng = random.Random(protocol["sampling_seed"])
    records = []
    for index in range(protocol["cases"]):
        chosen = rng.sample(pool, protocol["cells"])
        values = [model.lookup(
            row["observed_features"]["channel_estimate_power"],
            row["observed_features"]["received_grid_power"],
        ) for row in chosen]
        radio_floor = rng.choice(protocol["radio_floor_choices"])
        row = {"case": index, "status": "out_of_support"}
        if any(value is None for value in values):
            records.append(row)
            continue
        cells = tuple(Cell(
            name=chr(ord("a") + cell),
            deadline_ms=protocol["radio_deadline_ms"] - protocol["commit_guard_ms"],
            conventional_ms=protocol["conventional_bound_ms"],
            nrx_ready_ms=protocol["nrx_bound_ms"],
            neural_success_probability=values[cell].p_neural_success,
            incremental_rescue_probability=values[cell].q_rescue,
        ) for cell in range(protocol["cells"]))
        ai = tuple(AiUnit(
            f"ai{j}", protocol["ai_deadline_ms"],
            protocol["ai_bound_ms"], protocol["ai_bound_ms"], 1,
        ) for j in range(protocol["ai_units"]))
        problem = Problem(cells, ai, radio_floor,
                          protocol["nrx_endpoint_capacity"])
        plans = [optimize_for_nrx_subset(problem, subset)
                 for subset in all_nrx_subsets(problem)]
        feasible = [plan for plan in plans if plan is not None]
        if not feasible:
            row["status"] = "infeasible"
            records.append(row)
            continue
        joint = max(feasible, key=policy_score)
        greedy = greedy_radio_then_ai(problem)
        if greedy is None or joint["expected_ai_value"] + 1e-9 < greedy["expected_ai_value"]:
            raise AssertionError("joint/greedy feasibility or ordering invalid")
        smallest = min(len(plan["selected_nrx"]) for plan in feasible)
        staged = max(
            (plan for plan in feasible if len(plan["selected_nrx"]) == smallest),
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
        row.update({
            "status": "feasible", "radio_floor": radio_floor,
            "leaf_indices": [value.leaf_index for value in values],
            "joint_selected": joint["selected_nrx"],
            "joint_greedy_selected": greedy["selected_nrx"],
            "staged_selected": staged["selected_nrx"],
            "max_radio_selected": max_radio["selected_nrx"],
            "joint_ai": joint["expected_ai_value"],
            "joint_greedy_ai": greedy["expected_ai_value"],
            "staged_ai": staged["expected_ai_value"],
            "max_radio_ai": max_radio["expected_ai_value"],
            "joint_radio_gain": joint["radio_gain"],
            "staged_radio_gain": staged["radio_gain"],
            "max_radio_gain": max_radio["radio_gain"],
            "joint_actual_correct": realized_correct(chosen, joint["selected_nrx"]),
            "staged_actual_correct": realized_correct(chosen, staged["selected_nrx"]),
            "max_radio_actual_correct": realized_correct(chosen, max_radio["selected_nrx"]),
        })
        records.append(row)
    feasible = [row for row in records if row["status"] == "feasible"]
    differences = [row["joint_ai"] - row["staged_ai"] for row in feasible]
    radio_differences = [row["joint_radio_gain"] - row["staged_radio_gain"]
                         for row in feasible]
    report = {
        "schema": "softwall-confirm94-independent-phy-staged-comparator-v1",
        "job": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname(),
        "source_hashes": source_ok,
        "trace_sha256": hashlib.sha256(trace_path.read_bytes()).hexdigest(),
        "total": len(records), "feasible": len(feasible),
        "out_of_support": sum(row["status"] == "out_of_support"
                              for row in records),
        "joint_ai_above_staged": sum(value > 1e-9 for value in differences),
        "joint_exact_above_joint_greedy": sum(
            row["joint_ai"] > row["joint_greedy_ai"] + 1e-9
            for row in feasible),
        "joint_exact_differs_from_joint_greedy": sum(
            row["joint_selected"] != row["joint_greedy_selected"]
            for row in feasible),
        "joint_ai_above_staged_without_lower_expected_radio": sum(
            ai_gain > 1e-9 and radio_gain >= -1e-12
            for ai_gain, radio_gain in zip(differences, radio_differences)),
        "mean_joint_minus_staged_ai": statistics.mean(differences)
            if feasible else None,
        "mean_joint_minus_staged_radio": statistics.mean(radio_differences)
            if feasible else None,
        "joint_actual_correct": sum(row["joint_actual_correct"] for row in feasible),
        "staged_actual_correct": sum(row["staged_actual_correct"] for row in feasible),
        "max_radio_actual_correct": sum(row["max_radio_actual_correct"]
                                        for row in feasible),
        "records": records,
        "interpretation": "Prospectively fixed new-seed synthetic PHY CPU screen. Staged baseline minimizes NRx count meeting the same q rescue floor, maximizes q within that count, then exactly optimizes visible AI/recovery with the same all-fail safety evaluator. Joint greedy already matched the exact joint reference in Confirm93. Actual CRC labels are used only for paired evaluation after selection, never policy input. Positive CPU expected value is not an online or physical 7-AI burst result; independent GPU ABBA and radio noninferiority remain required.",
    }
    output = base / "confirm94_independent_phy_screen.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "total", "feasible", "out_of_support", "joint_ai_above_staged",
        "joint_exact_above_joint_greedy",
        "joint_ai_above_staged_without_lower_expected_radio",
        "mean_joint_minus_staged_ai", "mean_joint_minus_staged_radio",
        "joint_actual_correct", "staged_actual_correct",
        "max_radio_actual_correct")}, indent=2))


if __name__ == "__main__":
    main()
