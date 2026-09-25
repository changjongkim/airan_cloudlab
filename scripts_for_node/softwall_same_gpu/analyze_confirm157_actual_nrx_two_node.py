#!/usr/bin/env python3.11
"""Combine the passing C157A development and C157B frozen-node holdout."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-result", type=Path, required=True)
    parser.add_argument("--first-protocol", type=Path, required=True)
    parser.add_argument("--second-result", type=Path, required=True)
    parser.add_argument("--second-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    first = load(args.first_result)
    first_protocol = load(args.first_protocol)
    second = load(args.second_result)
    second_protocol = load(args.second_protocol)
    summaries = (first["summary"], second["summary"])
    gates = {
        "both_campaigns_pass": first["all_pass"] and second["all_pass"],
        "development_then_holdout": (
            first_protocol["campaign"] == "development"
            and second_protocol["campaign"] == "holdout"
        ),
        "distinct_job_node_seed": (
            first["job_id"] != second["job_id"]
            and first["node"] != second["node"]
            and first_protocol["seed_base"] != second_protocol["seed_base"]
        ),
        "holdout_excludes_development_node": (
            first["node"] in second_protocol["excluded_nodes"]
        ),
        "same_frozen_source_and_mode": (
            first_protocol["source_sha256"] == second_protocol["source_sha256"]
            and first_protocol["mode"] == second_protocol["mode"]
            and first_protocol["placement"] == second_protocol["placement"]
        ),
        "same_prospective_outcome_rule": (
            first_protocol["expected_actual_success_keys"]
            == second_protocol["expected_actual_success_keys"]
            and first_protocol["expected_actual_recovery_keys"]
            == second_protocol["expected_actual_recovery_keys"]
            and first_protocol["outcome_source"]
            == second_protocol["outcome_source"]
        ),
        "exact_two_node_totals": (
            sum(row["submitted_debts"] for row in summaries) == 10
            and sum(row["accepted_debts"] for row in summaries) == 8
            and sum(row["rejected_debts"] for row in summaries) == 2
            and sum(row["actual_nrx_successes"] for row in summaries) == 4
            and sum(row["physical_recoveries"] for row in summaries) == 4
            and sum(row["oracle_equivalent_recoveries"] for row in summaries) == 4
            and sum(row["qwen_units"] for row in summaries) == 2
            and sum(row["radio_commits"] for row in summaries) == 8
            and sum(row["deadline_misses"] for row in summaries) == 0
        ),
    }
    value = {
        "schema": "softwall-confirm157-two-node-actual-nrx-v1",
        "status": (
            "TWO_NODE_ACTUAL_NRX_TRANSITION_PASS_WARM_SYNTHETIC"
            if all(gates.values()) else "TWO_NODE_ACTUAL_NRX_TRANSITION_FAIL"
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
        "campaigns": [
            {
                "result": str(args.first_result),
                "protocol": str(args.first_protocol),
                "job_id": first["job_id"],
                "node": first["node"],
                "seed_base": first_protocol["seed_base"],
                "summary": first["summary"],
            },
            {
                "result": str(args.second_result),
                "protocol": str(args.second_protocol),
                "job_id": second["job_id"],
                "node": second["node"],
                "seed_base": second_protocol["seed_base"],
                "summary": second["summary"],
            },
        ],
        "totals": {
            "submitted_debts": sum(row["submitted_debts"] for row in summaries),
            "accepted_debts": sum(row["accepted_debts"] for row in summaries),
            "rejected_debts": sum(row["rejected_debts"] for row in summaries),
            "actual_nrx_successes": sum(
                row["actual_nrx_successes"] for row in summaries
            ),
            "physical_recoveries": sum(
                row["physical_recoveries"] for row in summaries
            ),
            "oracle_equivalent_recoveries": sum(
                row["oracle_equivalent_recoveries"] for row in summaries
            ),
            "qwen_units": sum(row["qwen_units"] for row in summaries),
            "radio_commits": sum(row["radio_commits"] for row in summaries),
            "deadline_misses": sum(row["deadline_misses"] for row in summaries),
            "max_nrx_release_to_complete_ms": max(
                row["max_nrx_release_to_complete_ms"] for row in summaries
            ),
            "max_radio_release_to_commit_ms": max(
                row["max_radio_release_to_commit_ms"] for row in summaries
            ),
        },
        "claim_scope": (
            "Two A100 nodes, disjoint seeds, identical frozen source/mode, and "
            "actual TensorRT NeuralRx CRC-driven transitions in a warm synthetic "
            "timing contract. This does not establish WCET, cold/long-idle "
            "qualification, production d_MAC, fault coverage, throughput "
            "superiority, or cross-family generality."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates, "totals": value["totals"]
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
