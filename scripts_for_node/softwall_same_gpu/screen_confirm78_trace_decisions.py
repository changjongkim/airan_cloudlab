#!/usr/bin/env python3.11
"""Exploratory joint-versus-strong-greedy screen using held-out PHY features.

This is a two-stage CPU decision model. It does not dispatch a GPU request or
qualify a deadline. True SNR and held-out PHY outcomes are never policy inputs.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from mixed_snr_phy_value_model import TrainOnlyMixedSnrPhyValueModel
from offline_contingent_phy_ai import (
    AiUnit, Cell, Problem, exact_joint_policy, greedy_radio_then_ai,
)


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
BASE = ROOT / "results/softwall_same_gpu"


def main() -> None:
    protocol = json.loads((BASE / "confirm78_trace_decision_protocol.json").read_text())
    for source, digest in protocol["source_sha256_before_run"].items():
        if hashlib.sha256((ROOT / source).read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"frozen source changed: {source}")
    model = TrainOnlyMixedSnrPhyValueModel(
        BASE / "confirm77_train_only_q_model.json", ROOT,
    )
    raw = json.loads((BASE / "raw" / protocol["heldout_raw"]).read_text())
    if raw["seed"] != protocol["heldout_seed"]:
        raise ValueError("held-out seed mismatch")
    records = []
    for ai_pattern in protocol["ai_patterns"]:
        for result in raw["results"]:
            snr = result["snr_db"]  # Report stratum only; never a policy input.
            for episode in range(protocol["episodes_per_snr"]):
                rows = result["records"][3 * episode:3 * episode + 3]
                values = [model.lookup(
                    row["observed_features"]["channel_estimate_power"],
                    row["observed_features"]["received_grid_power"],
                ) for row in rows]
                if any(value is None for value in values):
                    records.append({"pattern": ai_pattern["name"], "snr_db_report_only": snr,
                                    "episode": episode, "status": "out_of_support"})
                    continue
                cells = tuple(Cell(
                    name=chr(ord("a") + index),
                    deadline_ms=protocol["radio_deadline_ms"],
                    conventional_ms=protocol["conventional_bound_ms"],
                    nrx_ready_ms=protocol["nrx_ready_ms"],
                    neural_success_probability=value.p_neural_success,
                    incremental_rescue_probability=value.q_rescue,
                ) for index, value in enumerate(values))
                ai = tuple(AiUnit(**unit) for unit in ai_pattern["units"])
                problem = Problem(cells, ai, protocol["min_expected_radio_rescues"],
                                  protocol["nrx_endpoint_capacity"])
                exact = exact_joint_policy(problem)
                greedy = greedy_radio_then_ai(problem)
                if (exact is None) != (greedy is None):
                    raise AssertionError("exact/greedy feasibility mismatch")
                if exact is None:
                    status = "infeasible"
                    row = {"pattern": ai_pattern["name"], "snr_db_report_only": snr,
                           "episode": episode, "status": status}
                else:
                    if exact["expected_ai_value"] + 1e-9 < greedy["expected_ai_value"]:
                        raise AssertionError("exact below greedy")
                    row = {
                        "pattern": ai_pattern["name"], "snr_db_report_only": snr,
                        "episode": episode, "status": "feasible",
                        "model_leaf_indices": [value.leaf_index for value in values],
                        "exact_selected_nrx": exact["selected_nrx"],
                        "greedy_selected_nrx": greedy["selected_nrx"],
                        "exact_ai": exact["expected_ai_value"],
                        "greedy_ai": greedy["expected_ai_value"],
                        "exact_radio_gain": exact["radio_gain"],
                        "greedy_radio_gain": greedy["radio_gain"],
                        "exact_all_fail_certificate": exact["all_fail_recovery_certificate"],
                        "greedy_all_fail_certificate": greedy["all_fail_recovery_certificate"],
                    }
                records.append(row)
    feasible = [row for row in records if row["status"] == "feasible"]
    report = {
        "schema": "softwall-confirm78-heldout-feature-decision-screen-v1",
        "protocol": "results/softwall_same_gpu/confirm78_trace_decision_protocol.json",
        "trace_sha256": hashlib.sha256(
            (BASE / "raw" / protocol["heldout_raw"]).read_bytes()).hexdigest(),
        "model_sha256": hashlib.sha256(
            (BASE / "confirm77_train_only_q_model.json").read_bytes()).hexdigest(),
        "total": len(records),
        "feasible": len(feasible),
        "out_of_support": sum(row["status"] == "out_of_support" for row in records),
        "positive_expected_ai_gap": sum(
            row["exact_ai"] > row["greedy_ai"] + 1e-9 for row in feasible),
        "different_nrx_subset": sum(
            row["exact_selected_nrx"] != row["greedy_selected_nrx"] for row in feasible),
        "mean_exact_ai": sum(row["exact_ai"] for row in feasible) / len(feasible),
        "mean_greedy_ai": sum(row["greedy_ai"] for row in feasible) / len(feasible),
        "records": records,
        "interpretation": (
            "Exploratory CPU two-stage decision screen on held-out synthetic PHY "
            "features with a train-only q model. The AI workload is hypothetical. "
            "No actual GPU joint policy, correlated-outcome expected-value model, "
            "production DU expiry or hard timing certificate."
        ),
    }
    output = BASE / "confirm78_trace_decision_screen.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "total", "feasible", "out_of_support", "positive_expected_ai_gap",
        "different_nrx_subset", "mean_exact_ai", "mean_greedy_ai")}, indent=2))


if __name__ == "__main__":
    main()
