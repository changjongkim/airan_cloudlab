#!/usr/bin/env python3.11
"""Seal the C159 pre-experiment design without materializing holdout windows."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
DATASET_SHA256 = "46fc9480ef0b748ecb2b51d512ff08c196b031782cbe6f78e28044d768e86d5a"
SOURCES = (
    "docs/current/SOFTWALL_CONFIRM159_TRACE_PROTOCOL_KO.md",
    "scripts_for_node/softwall_same_gpu/build_c159_partitioned_trace.py",
    "scripts_for_node/softwall_same_gpu/test_build_c159_partitioned_trace.py",
    "data/current/softwall_c159_calibration_burst_windows_v1.json",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    dataset = ROOT / "data/public/burstgpt/BurstGPT_1.csv"
    observed = sha256(dataset)
    if observed != DATASET_SHA256:
        raise RuntimeError(f"BurstGPT source mismatch: {observed}")
    calibration_path = ROOT / SOURCES[-1]
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    result = {
        "schema": "softwall-confirm159-experiment-prespec-v1",
        "status": "DESIGN_PRESPEC_HOLDOUT_NOT_MATERIALIZED",
        "predecessor": {
            "result": "results/softwall_multigpu/confirm158_repeated_actual_nrx_two_node.json",
            "required_status": "C158_TWO_NODE_REPEATED_ACTUAL_NRX_PASS",
        },
        "dataset": {
            "path": str(dataset.relative_to(ROOT)),
            "sha256": observed,
            "partitions": {
                "development": [5, 1756661],
                "calibration": [1756661, 3513317],
                "confirmatory_holdout": [3513317, 5269974],
            },
            "holdout_materialized": False,
        },
        "calibration": {
            "path": SOURCES[-1],
            "sha256": sha256(calibration_path),
            "summary": calibration["summary"],
            "window_ids": [item["window_id"] for item in calibration["windows"]],
        },
        "candidate_mode": {
            "radio_period_ms": 180,
            "radio_expiry_ms": 155,
            "nrx_bound_ms": 45,
            "recovery_bound_ms": 25,
            "launch_control_bound_ms": 5,
            "guard_ms": 2,
            "qwen_candidate_bounds_ms": {
                "16": 35, "32": 35, "64": 35,
                "128": 40, "256": 65, "512": 75,
            },
            "qwen_bounds_inherited": False,
        },
        "safe_systems": ["static_global_safe", "event_driven_global_safe", "softwall"],
        "primary_comparison": "softwall_vs_event_driven_global_safe",
        "primary_metric": "timely_value_tokens",
        "minimum_effect_pct": 5.0,
        "qualification": {
            "nodes": 2,
            "minimum_actual_nrx_requests_per_node": 1000,
            "zero_violation_required": True,
            "contexts_requalified": [16, 32, 64, 128, 256, 512],
        },
        "analysis": {
            "paired_unit": "source_second_x_window_x_node_x_seed_block",
            "confidence": 0.95,
            "power": 0.8,
            "minimum_windows": 4,
            "maximum_windows": 12,
            "oracle_upper_bound_screen_before_power": True,
            "qualification_excluded_from_confirmatory_ci": True,
        },
        "source_sha256": {relative: sha256(ROOT / relative) for relative in SOURCES},
        "next_gate": "implement_and_run_C159_Q1_before_materializing_holdout",
    }
    output = ROOT / "results/softwall_multigpu/confirm159_experiment_prespec_v1.json"
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"status": result["status"], "next_gate": result["next_gate"]}, indent=2))


if __name__ == "__main__":
    main()
