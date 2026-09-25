#!/usr/bin/env python3
"""Audit the frozen trace-driven strong-baseline campaign."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
RESULTS = ROOT / "results/softwall_same_gpu"
RAW = RESULTS / "raw"
PROTOCOL = RESULTS / "confirm112_strong_baselines_protocol.json"


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def safety(result: dict) -> dict:
    zero_fields = (
        "deadline_misses", "nrx_bound_violations", "conv_bound_violations",
        "conv_path_bound_violations", "background_budget_violations",
        "background_horizon_violations", "retime_rejections",
    )
    endpoints = result["endpoint_final"]
    return {
        "zero_fields": {field: result[field] for field in zero_fields},
        "calendar_outstanding": result["fallback_calendar_final"]["outstanding"],
        "joint_leases_outstanding": result["fallback_calendar_final"]["joint_leases_outstanding"],
        "endpoint_outstanding": {
            name: value["outstanding"] for name, value in endpoints.items()
        },
        "background_faults": len(result["background_faults"]),
        "endpoint_faults": len(result["endpoint_faults"]),
        "pass": (
            all(result[field] == 0 for field in zero_fields)
            and result["fallback_calendar_final"]["outstanding"] == 0
            and result["fallback_calendar_final"]["joint_leases_outstanding"] == 0
            and all(value["outstanding"] == 0 for value in endpoints.values())
            and not result["background_faults"]
            and not result["endpoint_faults"]
        ),
    }


def radio_signature(result: dict) -> list[tuple]:
    return [
        (
            row["index"], row["cell"], row["gate_skipped"], row["admitted"],
            row["forced_nrx_failure"], row["commit_kind"],
        )
        for row in result["records"]
    ]


def timely_by_request(result: dict) -> dict[str, int]:
    return {
        row["request_id"]: row["value_tokens"] if row["timely"] else 0
        for row in result["background_records"]
    }


def compare_pair(pair: dict, arms: dict, trace: dict, seed: int) -> dict:
    soft = [arms[name] for name in pair["softwall"]]
    work = [arms[name] for name in pair["work_conserving"]]
    soft_values = [timely_by_request(item) for item in soft]
    work_values = [timely_by_request(item) for item in work]
    blocks = defaultdict(float)
    soft_total = 0.0
    work_total = 0.0
    for request in trace["requests"]:
        request_id = request["request_id"]
        soft_value = sum(values.get(request_id, 0) for values in soft_values) / len(soft_values)
        work_value = sum(values.get(request_id, 0) for values in work_values) / len(work_values)
        blocks[request["source_timestamp_s"]] += soft_value - work_value
        soft_total += soft_value
        work_total += work_value
    block_values = list(blocks.values())
    rng = random.Random(seed)
    bootstrap = [
        sum(rng.choice(block_values) for _ in block_values)
        for _ in range(10_000)
    ]
    delta = soft_total - work_total
    effect_pct = 100.0 * delta / work_total if work_total else 0.0
    lower = percentile(bootstrap, 0.025)
    upper = percentile(bootstrap, 0.975)
    return {
        "name": pair["name"],
        "softwall_arms": pair["softwall"],
        "work_conserving_arms": pair["work_conserving"],
        "softwall_mean_timely_value": soft_total,
        "work_conserving_mean_timely_value": work_total,
        "delta_tokens": delta,
        "effect_pct": effect_pct,
        "paired_source_second_bootstrap_95_ci": [lower, upper],
        "effect_at_least_2pct": effect_pct >= 2.0,
        "bootstrap_lower_positive": lower > 0,
    }


def main() -> None:
    protocol = read(PROTOCOL)
    trace_path = ROOT / protocol["trace"]["path"]
    trace = read(trace_path)
    source_audit = {
        relative: {
            "expected": expected,
            "observed": sha256(ROOT / relative),
            "match": sha256(ROOT / relative) == expected,
        }
        for relative, expected in protocol["source_sha256_before_run"].items()
    }
    arms = {}
    arm_audit = {}
    for arm in protocol["arms"]:
        path = RAW / f"{arm['prefix']}_controller.json"
        result = read(path)
        arms[arm["name"]] = result
        arm_audit[arm["name"]] = {
            "path": str(path.relative_to(ROOT)),
            "sha256": sha256(path),
            "system": result["system"],
            "iterations": result["iterations"],
            "trace_sha256": result["trace_sha256"],
            "timely_requests": result["trace_summary"]["timely_requests"],
            "timely_value_tokens": result["trace_summary"]["timely_value_tokens"],
            "correct_cells": result["correct_cells"],
            "recovery_retime_count": result["recovery_retime_count"],
            "joint_lease_retired_count": result["joint_lease_retired_count"],
            "safety": safety(result),
        }
    parity = {}
    for seed_name, names in protocol["radio_parity_groups"].items():
        reference = radio_signature(arms[names[0]])
        parity[seed_name] = {
            "arms": names,
            "structural_signature_equal": all(
                radio_signature(arms[name]) == reference for name in names[1:]
            ),
            "correct_cells": {name: arms[name]["correct_cells"] for name in names},
        }
    comparisons = [
        compare_pair(pair, arms, trace, 112_000 + index)
        for index, pair in enumerate(protocol["paired_comparisons"])
    ]
    safety_pass = all(value["safety"]["pass"] for value in arm_audit.values())
    provenance_pass = (
        sha256(trace_path) == protocol["trace"]["sha256"]
        and all(value["match"] for value in source_audit.values())
    )
    structural_pass = all(value["structural_signature_equal"] for value in parity.values())
    outcome_pass = all(
        item["effect_at_least_2pct"] and item["bootstrap_lower_positive"]
        for item in comparisons
    )
    result = {
        "schema": "softwall-confirm112-strong-baselines-v1",
        "protocol_sha256": sha256(PROTOCOL),
        "trace_sha256_observed": sha256(trace_path),
        "source_audit": source_audit,
        "arms": arm_audit,
        "radio_parity": parity,
        "paired_comparisons": comparisons,
        "gates": {
            "provenance": provenance_pass,
            "safety": safety_pass,
            "radio_structural_parity": structural_pass,
            "outcome": outcome_pass,
            "all_pass": provenance_pass and safety_pass and structural_pass and outcome_pass,
        },
        "interpretation": (
            "The outcome gate requires both independent ABBA groups to show at least "
            "2% more timely token value than safe work-conserving and a positive paired "
            "source-second bootstrap lower bound. Failure ends the throughput-superiority "
            "claim but does not invalidate a passing conditional-recovery substrate."
        ),
    }
    output = RESULTS / f"confirm112_strong_baselines_job{protocol['job']}.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["gates"], indent=2))


if __name__ == "__main__":
    main()
