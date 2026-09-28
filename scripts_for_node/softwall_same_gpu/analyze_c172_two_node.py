#!/usr/bin/env python3.11
"""Deterministically aggregate the prespecified C172 two-node gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SCENARIOS = (
    "E4_softwall_reject",
    "E4_debt_blind_launch",
    "E6a_softwall_admit",
    "E6b_softwall_reject",
    "E6b_shadow_launch",
)
SAFE_SCENARIOS = (
    "E4_softwall_reject",
    "E6a_softwall_admit",
    "E6b_softwall_reject",
)


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rule_of_three_zero_upper(n: int) -> float | None:
    return 3.0 / n if n else None


def sum_scenario(results: list[dict], scenario: str, field: str) -> int:
    return sum(int(row["per_scenario"][scenario][field]) for row in results)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--development", type=Path, required=True)
    parser.add_argument("--holdout", type=Path, required=True)
    parser.add_argument("--development-protocol", type=Path, required=True)
    parser.add_argument("--holdout-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    result_paths = [args.development, args.holdout]
    protocol_paths = [args.development_protocol, args.holdout_protocol]
    results = [load(path) for path in result_paths]
    protocols = [load(path) for path in protocol_paths]

    per_scenario = {}
    for scenario in SCENARIOS:
        per_scenario[scenario] = {
            "selected_rounds": sum_scenario(results, scenario, "n"),
            "jointly_bound_valid_rounds": sum_scenario(
                results, scenario, "jointly_bound_valid_rounds"
            ),
            "guard_violations": sum_scenario(results, scenario, "guard_violations"),
            "deadline_violations": sum_scenario(
                results, scenario, "deadline_violations"
            ),
            "bound_valid_guard_violations": sum_scenario(
                results, scenario, "bound_valid_guard_violations"
            ),
            "bound_valid_deadline_violations": sum_scenario(
                results, scenario, "bound_valid_deadline_violations"
            ),
            "timing_invalid_attempts": sum_scenario(
                results, scenario, "timing_invalid_attempts"
            ),
            "unused_timing_valid_attempts": sum_scenario(
                results, scenario, "unused_timing_valid_attempts"
            ),
        }

    safe_selected_n = sum(per_scenario[name]["selected_rounds"] for name in SAFE_SCENARIOS)
    safe_bound_valid_n = sum(
        per_scenario[name]["jointly_bound_valid_rounds"] for name in SAFE_SCENARIOS
    )
    safe_guard_violations = sum(
        per_scenario[name]["guard_violations"] for name in SAFE_SCENARIOS
    )
    safe_deadline_violations = sum(
        per_scenario[name]["deadline_violations"] for name in SAFE_SCENARIOS
    )
    safe_bound_valid_guard_violations = sum(
        per_scenario[name]["bound_valid_guard_violations"] for name in SAFE_SCENARIOS
    )
    safe_bound_valid_deadline_violations = sum(
        per_scenario[name]["bound_valid_deadline_violations"] for name in SAFE_SCENARIOS
    )

    counts = {
        key: sum(int(result["counts"][key]) for result in results)
        for key in results[0]["counts"]
    }
    gates = {
        "both_frozen_runs_pass": all(result["all_pass"] for result in results),
        "two_distinct_nonexcluded_nodes": (
            len({result["host"] for result in results}) == 2
            and all(
                result["host"] not in protocol["node_exclusion_rule"]["excluded_before_allocation"]
                for result, protocol in zip(results, protocols)
            )
        ),
        "development_then_holdout_roles": [result["campaign"] for result in results]
        == ["development", "holdout"],
        "holdout_reverses_scenario_order": (
            protocols[0]["reverse_scenarios"] is False
            and protocols[1]["reverse_scenarios"] is True
        ),
        "independent_seeds": protocols[0]["seed_base"] != protocols[1]["seed_base"],
        "identical_frozen_sources": protocols[0]["source_sha256"]
        == protocols[1]["source_sha256"],
        "identical_capacity_model": protocols[0]["capacity_model"]["sha256"]
        == protocols[1]["capacity_model"]["sha256"],
        "forty_selected_rounds_per_scenario": all(
            row["selected_rounds"] == 40 for row in per_scenario.values()
        ),
        "debt_blind_E4_violates_guard_and_deadline": (
            per_scenario["E4_debt_blind_launch"]["guard_violations"] == 40
            and per_scenario["E4_debt_blind_launch"]["deadline_violations"] == 40
        ),
        "shadow_E6b_violates_guard": (
            per_scenario["E6b_shadow_launch"]["guard_violations"] == 40
        ),
        "softwall_selected_rounds_have_zero_violations": (
            safe_guard_violations == 0 and safe_deadline_violations == 0
        ),
        "softwall_bound_valid_rounds_have_zero_violations": (
            safe_bound_valid_guard_violations == 0
            and safe_bound_valid_deadline_violations == 0
        ),
    }

    value = {
        "schema": "softwall-c172-debt-blind-two-node-v1",
        "status": "C172_TWO_NODE_PASS" if all(gates.values()) else "C172_TWO_NODE_FAIL",
        "all_pass": all(gates.values()),
        "analysis_role": (
            "Deterministic aggregation of the two-node gate already fixed in both "
            "C172 protocols. No threshold, exclusion, or physical outcome is added."
        ),
        "gates": gates,
        "nodes": [result["host"] for result in results],
        "jobs": [result["slurm_job_id"] for result in results],
        "counts": counts,
        "per_scenario": per_scenario,
        "safe_policy_summary": {
            "selected_rounds": safe_selected_n,
            "guard_violations": safe_guard_violations,
            "deadline_violations": safe_deadline_violations,
            "zero_violation_95pct_rule_of_three_upper": rule_of_three_zero_upper(
                safe_selected_n
            ),
            "jointly_bound_valid_rounds": safe_bound_valid_n,
            "bound_valid_guard_violations": safe_bound_valid_guard_violations,
            "bound_valid_deadline_violations": safe_bound_valid_deadline_violations,
            "bound_valid_zero_95pct_rule_of_three_upper": rule_of_three_zero_upper(
                safe_bound_valid_n
            ),
        },
        "claim_boundary": (
            "Physical diagnostic of a declared B_conv=25 ms contract on two A100 "
            "nodes. Padding makes the allowed upper-bound execution observable; "
            "it is not a production-rate or WCET qualification claim."
        ),
        "inputs": {
            "development": {"path": str(args.development), "sha256": digest(args.development)},
            "holdout": {"path": str(args.holdout), "sha256": digest(args.holdout)},
            "development_protocol": {
                "path": str(args.development_protocol),
                "sha256": digest(args.development_protocol),
            },
            "holdout_protocol": {
                "path": str(args.holdout_protocol),
                "sha256": digest(args.holdout_protocol),
            },
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
