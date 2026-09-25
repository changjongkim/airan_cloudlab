#!/usr/bin/env python3.11
"""Prospective Confirm102 independent PHY and multi-event policy analysis."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

from cv_grid_phy_value_model import TrainOnlyCvGridPhyValueModel
from offline_online_recovery_v4 import (
    AiJob, Cell, Config, branch_expectation, evaluate_subset, feasible_plans,
    joint_one_swap_policy, max_radio_policy, radio_guarded_joint_policy,
    radio_guarded_one_swap_policy, replay_branch, staged_min_policy, uniform_q,
)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def actual_correct(rows: list[dict], selected: list[str]) -> int:
    chosen = set(selected)
    return sum(bool(row["conventional_correct"] or
                    (chr(97 + i) in chosen and row["neural_correct"]))
               for i, row in enumerate(rows))


def realized(cells: tuple[Cell, ...], selected: list[str], rows: list[dict],
             jobs: tuple[AiJob, ...], config: Config) -> dict:
    chosen = frozenset(selected)
    outcomes = {name: bool(rows[ord(name) - 97]["neural_correct"])
                for name in chosen}
    replay = replay_branch(cells, chosen, outcomes, jobs, config)
    if not replay["safe"]:
        raise AssertionError("a selected policy lost mandatory safety")
    return {
        "actual_ai": replay["completed_ai_value"],
        "actual_correct": actual_correct(rows, selected),
        "mandatory_completed": len(replay["completed_conv"]),
        "finish_ms": replay["finish_ms"],
    }


def comparison(rows: list[dict], proposed: str, baseline: str) -> dict:
    ai = [row["policies"][proposed]["actual_ai"]
          - row["policies"][baseline]["actual_ai"] for row in rows]
    expected = [row["policies"][proposed]["expected_full_ai"]
                - row["policies"][baseline]["expected_full_ai"] for row in rows]
    radio = [row["policies"][proposed]["radio_gain"]
             - row["policies"][baseline]["radio_gain"] for row in rows]
    correct = [row["policies"][proposed]["actual_correct"]
               - row["policies"][baseline]["actual_correct"] for row in rows]
    return {
        "different_selection": sum(
            row["policies"][proposed]["selected"]
            != row["policies"][baseline]["selected"] for row in rows
        ),
        "actual_ai_wins_ties_losses": [sum(value > 0 for value in ai),
                                        sum(value == 0 for value in ai),
                                        sum(value < 0 for value in ai)],
        "actual_ai_sum_difference": sum(ai),
        "mean_expected_full_ai_difference": statistics.mean(expected),
        "maximum_per_case_predicted_radio_loss": max((-value for value in radio),
                                                       default=0.0),
        "mean_radio_gain_difference": statistics.mean(radio),
        "actual_correct_sum_difference": sum(correct),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol_path = base / "confirm102_multi_event_policy_protocol.json"
    protocol = read(protocol_path)
    amendment_path = base / "confirm102_multi_event_policy_protocol_amendment1.json"
    amendment = read(amendment_path)
    if sha256(protocol_path) != amendment["original_protocol_sha256"]:
        raise RuntimeError("original frozen protocol changed")
    source_hashes = all(sha256(root / name) == digest
                        for name, digest in amendment["source_sha256"].items())
    if not source_hashes:
        raise RuntimeError("frozen source changed")
    trace_path = base / protocol["trace"]
    manifest_path = base / protocol["manifest"]
    if (sha256(trace_path) != protocol["trace_sha256"]
            or sha256(manifest_path) != protocol["manifest_sha256"]):
        raise RuntimeError("sealed trace changed")
    trace = read(trace_path)
    manifest = read(manifest_path)
    if (trace["seed"] != protocol["phy_seed"]
            or manifest["seed"] != protocol["phy_seed"]
            or manifest["trials_per_snr"] != protocol["trials_per_snr"]
            or manifest["snrs_db"] != protocol["snrs_db"]
            or not trace["observed_features_recorded"]
            or not trace["observed_features_prewarmed"]):
        raise RuntimeError("trace provenance mismatch")

    model_path = base / protocol["q_model"]
    if sha256(model_path) != protocol["q_model_sha256"]:
        raise RuntimeError("q model changed")
    model = TrainOnlyCvGridPhyValueModel(model_path, root)
    pool = [row for result in trace["results"] for row in result["records"]]
    predictions = []
    by_leaf = defaultdict(list)
    for row in pool:
        value = model.lookup(row["observed_features"]["channel_estimate_power"],
                             row["observed_features"]["received_grid_power"])
        if value is None:
            continue
        rescue = int(row["neural_correct"] and not row["conventional_correct"])
        predictions.append((rescue, value.q_rescue, value.p_neural_success,
                            int(row["neural_correct"]), value.leaf_index))
        by_leaf[value.leaf_index].append((rescue, value.q_rescue))
    train_model = read(model_path)
    train_prevalence = (sum(leaf["rescue_count"] for leaf in train_model["leaves"])
                        / sum(leaf["train_n"] for leaf in train_model["leaves"]))
    q_brier = statistics.mean((truth - q) ** 2 for truth, q, _, _, _ in predictions)
    constant_brier = statistics.mean((truth - train_prevalence) ** 2
                                     for truth, _, _, _, _ in predictions)
    p_brier = statistics.mean((truth - p) ** 2 for _, _, p, truth, _ in predictions)
    q_mace = sum(abs(statistics.mean(truth for truth, _ in values) - values[0][1])
                 * len(values) for values in by_leaf.values()) / len(predictions)
    calibration = {
        "total": len(pool), "in_support": len(predictions),
        "support_fraction": len(predictions) / len(pool),
        "rescue_count": sum(row[0] for row in predictions),
        "q_brier": q_brier, "train_prevalence_constant_brier": constant_brier,
        "q_brier_better_than_constant": q_brier < constant_brier,
        "q_weighted_mace": q_mace, "p_neural_brier": p_brier,
        "gate": (len(predictions) / len(pool) >= protocol["minimum_support_fraction"]
                 and q_brier < constant_brier
                 and q_mace <= protocol["maximum_q_mace"]),
    }

    timing = protocol["timing"]
    config = Config(
        timing["radio_deadline_ms"], timing["nrx_observe_replay_ms"],
        timing["nrx_admission_bound_ms"], timing["conv_admission_bound_ms"],
        timing["conv_replay_ms"], protocol["endpoint_capacity"],
    )
    jobs = tuple(AiJob(**row) for row in protocol["ai_jobs"])
    visible = tuple(job for job in jobs if job.release_ms <= 0)
    threshold = read(base / amendment["low_gate_model"])["threshold"]
    rng = random.Random(protocol["sampling_seed"])
    records = []
    for case in range(protocol["cases"]):
        rows = rng.sample(pool, protocol["cells"])
        estimates = [model.lookup(
            row["observed_features"]["channel_estimate_power"],
            row["observed_features"]["received_grid_power"],
        ) for row in rows]
        floor = rng.choice(protocol["radio_floor_choices"])
        entry = {"case": case, "radio_floor": floor}
        if any(value is None for value in estimates):
            entry["status"] = "out_of_support"
            records.append(entry)
            continue
        cells = tuple(Cell(chr(97 + i), value.p_neural_success, value.q_rescue)
                      for i, value in enumerate(estimates))
        staged = staged_min_policy(cells, visible, config, floor)
        max_radio = max_radio_policy(cells, visible, config, floor)
        joint = joint_one_swap_policy(cells, visible, config, floor)
        guarded = radio_guarded_joint_policy(
            cells, visible, config, floor, protocol["maximum_predicted_radio_loss"])
        guarded_greedy = radio_guarded_one_swap_policy(
            cells, visible, config, floor, protocol["maximum_predicted_radio_loss"])
        if any(plan is None for plan in
               (staged, max_radio, joint, guarded, guarded_greedy)):
            entry["status"] = "infeasible"
            records.append(entry)
            continue
        eligible = [i for i, row in enumerate(rows)
                    if row["observed_features"]["channel_estimate_power"] >= threshold]
        eligible.sort(key=lambda i: (-cells[i].q_incremental_rescue, cells[i].name))
        low_names = frozenset(cells[i].name for i in eligible[:config.endpoint_capacity])
        low = evaluate_subset(cells, low_names, visible, config, 0.0)
        if low is None:
            raise AssertionError("low gate lost all-fail feasibility")
        plans = feasible_plans(cells, visible, config, floor)
        forced_values = []
        for plan in plans:
            forced = evaluate_subset(
                cells, frozenset(plan["selected_nrx"]), visible, config,
                floor, force_all_mandatory=True,
            )
            forced_values.append(forced["expected_visible_ai_value"])
        uniform_cells = uniform_q(cells)
        uniform_floor = min(floor, config.endpoint_capacity
                            * uniform_cells[0].q_incremental_rescue)
        uniform_joint = joint_one_swap_policy(
            uniform_cells, visible, config, uniform_floor)
        uniform_staged = staged_min_policy(
            uniform_cells, visible, config, uniform_floor)

        policies = {
            "proposed_guarded": guarded,
            "proposed_guarded_greedy": guarded_greedy,
            "joint_greedy": joint,
            "staged_min": staged,
            "max_radio": max_radio,
            "low_gate": low,
        }
        entry.update({
            "status": "feasible",
            "conditional_capacity_span": (
                max(plan["expected_visible_ai_value"] for plan in plans)
                - min(plan["expected_visible_ai_value"] for plan in plans)
            ),
            "always_mandatory_capacity_span": max(forced_values) - min(forced_values),
            "uniform_q_joint_differs_from_staged": (
                uniform_joint["selected_nrx"] != uniform_staged["selected_nrx"]
            ),
            "policies": {},
        })
        for name, plan in policies.items():
            selected = plan["selected_nrx"]
            expectation = branch_expectation(
                cells, frozenset(selected), jobs, config)
            if not expectation["safe"]:
                raise AssertionError("expected replay lost safety")
            entry["policies"][name] = {
                "selected": selected,
                "radio_gain": plan["radio_gain"],
                "expected_visible_ai": plan["expected_visible_ai_value"],
                "expected_full_ai": expectation["expected_ai_value"],
                **realized(cells, selected, rows, jobs, config),
            }
        records.append(entry)

    feasible = [row for row in records if row["status"] == "feasible"]
    comparisons = {
        baseline: comparison(feasible, "proposed_guarded", baseline)
        for baseline in ("proposed_guarded_greedy", "joint_greedy",
                         "staged_min", "max_radio", "low_gate")
    }
    structural_gate = (
        len(feasible) >= protocol["minimum_feasible_cases"]
        and all(row["policies"]["proposed_guarded"]["selected"]
                == row["policies"]["proposed_guarded_greedy"]["selected"]
                for row in feasible)
        and all(abs(row["always_mandatory_capacity_span"]) <= 1e-9
                for row in feasible)
        and comparisons["max_radio"]["maximum_per_case_predicted_radio_loss"]
            <= protocol["maximum_predicted_radio_loss"] + 1e-9
    )
    outcome_gate = (
        comparisons["max_radio"]["actual_ai_sum_difference"] > 0
        and comparisons["max_radio"]["actual_correct_sum_difference"] >= 0
        and comparisons["staged_min"]["actual_ai_sum_difference"] > 0
        and comparisons["staged_min"]["actual_correct_sum_difference"] >= 0
    )
    report = {
        "schema": "softwall-confirm102-independent-multi-event-policy-v1",
        "protocol_sha256": sha256(protocol_path),
        "protocol_amendment_sha256": sha256(amendment_path),
        "source_hashes": source_hashes,
        "trace_sha256": sha256(trace_path),
        "calibration": calibration,
        "total_cases": len(records), "feasible_cases": len(feasible),
        "out_of_support_cases": sum(row["status"] == "out_of_support"
                                    for row in records),
        "infeasible_cases": sum(row["status"] == "infeasible" for row in records),
        "conditional_capacity_nonconstant": sum(
            row["conditional_capacity_span"] > 1e-9 for row in feasible),
        "always_mandatory_ablation_zero_span": sum(
            abs(row["always_mandatory_capacity_span"]) <= 1e-9
            for row in feasible),
        "uniform_q_joint_differs_from_staged": sum(
            row["uniform_q_joint_differs_from_staged"] for row in feasible),
        "comparisons": comparisons,
        "calibration_gate": calibration["gate"],
        "structural_gate": structural_gate,
        "outcome_gate": outcome_gate,
        "all_pass": calibration["gate"] and structural_gate and outcome_gate,
        "records": records,
        "interpretation": (
            "Prospectively frozen independent synthetic-PHY CPU replay. "
            "Admission uses C100 declared bounds while deterministic replay "
            "durations only model observed early completion and are not WCETs. "
            "Equal-value/equal-size AI jobs remove heterogeneous knapsack as "
            "the source of a gap. Future arrivals are hidden at the radio "
            "decision. This is not a GPU throughput, actual AI-RAN workload, "
            "production DU, or hard-real-time result."
        ),
    }
    output = base / "confirm102_independent_multi_event_policy.json"
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: value for key, value in report.items()
                      if key not in ("records",)}, indent=2))


if __name__ == "__main__":
    main()
