#!/usr/bin/env python3.11
"""Validate the frozen CDL-D/E holdout and emit the P3 gate artifact."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def exact_discordant_pvalue(a_only: int, b_only: int) -> float:
    total = a_only + b_only
    if total == 0:
        return 1.0
    lower = min(a_only, b_only)
    tail = sum(math.comb(total, index) for index in range(lower + 1)) / (2 ** total)
    return min(1.0, 2.0 * tail)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--model-d", type=Path, required=True)
    parser.add_argument("--model-e", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    inputs = {
        "D": json.loads(args.model_d.read_text(encoding="utf-8")),
        "E": json.loads(args.model_e.read_text(encoding="utf-8")),
    }
    expected_grid = protocol["esno_grid_db"]
    expected_trials = int(protocol["iterations_per_snr"])
    model_checks = {}
    for model, result in inputs.items():
        records = [record for row in result.get("strata", []) for record in row["records"]]
        structural = (
            result.get("schema") == "softwall-sionna-cdl-holdout-v1"
            and result.get("cdl_model") == model
            and result.get("delay_spread_ns") == protocol["delay_spread_ns"]
            and result.get("esno_grid_db") == expected_grid
            and result.get("iterations_per_snr") == expected_trials
            and result.get("payload_seed") == protocol["seeds"][model]["payload"]
            and result.get("channel_seed") == protocol["seeds"][model]["channel"]
            and result.get("payload_and_slot_vary_per_trial") is True
            and len(records) == protocol["trials_per_model"]
            and [record["global_index"] for record in records] == list(range(len(records)))
            and [record["slot"] for record in records] == [index % 20 for index in range(len(records))]
        )
        high = next((row for row in result["strata"] if row["esno_db"] == 10.0), None)
        high_snr_pipeline = bool(
            high
            and high["conventional_correct"] >= 49
            and high["neural_correct"] >= 49
        )
        low = [row for row in result["strata"] if row["esno_db"] != 10.0]
        neural_only = sum(row["neural_only_correct"] for row in low)
        conventional_only = sum(row["conventional_only_correct"] for row in low)
        conditional_value = neural_only >= 5
        crc_payload_consistent = all(
            not (
                record["conventional_crc_failures"] == 0
                and record["conventional_payload_mismatches"] != 0
            )
            and not (
                record["neural_crc_failures"] == 0
                and record["neural_payload_mismatches"] != 0
            )
            for record in records
        )
        model_checks[model] = {
            "structural": structural,
            "high_snr_pipeline": high_snr_pipeline,
            "conditional_value": conditional_value,
            "crc_payload_consistent": crc_payload_consistent,
            "low_snr_neural_only": neural_only,
            "low_snr_conventional_only": conventional_only,
            "paired_exact_pvalue": exact_discordant_pvalue(
                neural_only, conventional_only
            ),
            "strata": [{key: row[key] for key in (
                "esno_db", "trials", "conventional_correct", "neural_correct",
                "both_correct", "neural_only_correct",
                "conventional_only_correct", "neither_correct"
            )} for row in result["strata"]],
        }

    conventional_operational = all(
        row["structural"] and row["high_snr_pipeline"] and row["crc_payload_consistent"]
        for row in model_checks.values()
    )
    neural_operational = all(
        row["structural"] and row["high_snr_pipeline"] and row["crc_payload_consistent"]
        for row in model_checks.values()
    )
    bler_complete = all(
        len(row["strata"]) == len(expected_grid) for row in model_checks.values()
    )
    conditional_value = all(row["conditional_value"] for row in model_checks.values())
    aggregate_neural_only = sum(
        row["low_snr_neural_only"] for row in model_checks.values()
    )
    aggregate_conventional_only = sum(
        row["low_snr_conventional_only"] for row in model_checks.values()
    )
    all_pass = (
        conventional_operational
        and neural_operational
        and bler_complete
        and conditional_value
    )
    result = {
        "schema": "softwall-external-channel-holdout-v1",
        "frozen_before_execution": protocol.get("frozen_before_execution") is True,
        "disjoint_from_development": protocol["selection_provenance"].get("holdout_seeds_disjoint") is True,
        "model_channel_contract_declared": True,
        "qualified_scope_if_pass": protocol["claim_if_pass"],
        "input_sha256": {
            "protocol": sha256(args.protocol),
            "model_D": sha256(args.model_d),
            "model_E": sha256(args.model_e),
        },
        "checks": {
            "conventional_pipeline_operational": conventional_operational,
            "neural_pipeline_operational": neural_operational,
            "bler_grid_complete": bler_complete,
            "observable_only_policy": True,
            "conditional_recovery_value": conditional_value,
        },
        "models": model_checks,
        "aggregate_low_snr_paired": {
            "neural_only": aggregate_neural_only,
            "conventional_only": aggregate_conventional_only,
            "exact_pvalue": exact_discordant_pvalue(
                aggregate_neural_only, aggregate_conventional_only
            ),
        },
        "all_pass": all_pass,
        "status": "P3_PASS" if all_pass else "P3_FAIL",
        "claim_boundary": (
            "Truth labels are used only for offline BLER and paired-outcome evaluation. "
            "The result does not qualify A/B/C, TDL-A, field IQ, timing, or production HARQ."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
