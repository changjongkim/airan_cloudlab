#!/usr/bin/env python3.11
"""Freeze the C174 BurstGPT/Qwen oracle screen before outcomes."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
INPUTS = {
    "calibration_trace": "data/current/softwall_c159_calibration_burst_windows_v1.json",
    "c167_protocol": "results/softwall_multigpu/c167_capacity_model_protocol_v2.json",
    "c167_result": "results/softwall_multigpu/c167_reclaimable_capacity_model_v2.json",
    "c173_result": "results/softwall_multigpu/c173_deadline_correction_result_v1.json",
    "screen": "scripts_for_node/softwall_same_gpu/c174_trace_oracle_screen.py",
    "test": "scripts_for_node/softwall_same_gpu/test_c174_trace_oracle_screen.py",
    "capacity_model": "scripts_for_node/softwall_same_gpu/c167_reclaimable_capacity_model_v2.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = ROOT / "results/softwall_multigpu/c174_trace_oracle_protocol_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol {output}")
    grid = {
        "topologies": [
            {"cells": 4, "homes": 2},
            {"cells": 8, "homes": 2},
            {"cells": 8, "homes": 4},
            {"cells": 16, "homes": 4},
        ],
        "gpus": [1, 2, 4],
        "nrx_success_probability": [0.2, 0.5, 0.8],
        "failure_correlation": [0.0, 0.5, 1.0],
        "recovery_bound_ms": [12.0, 25.0],
        "ai_slo_ms": [50, 100, 250, 1000],
        "ai_units_per_epoch": [1, 2, 4],
    }
    expected = math.prod(len(values) for values in grid.values())
    protocol = {
        "schema": "softwall-c174-trace-oracle-protocol-v1",
        "status": "FROZEN_BEFORE_C174_SCREEN",
        "analysis_role": (
            "Find, before any policy implementation, whether an exact recovery-order "
            "oracle exceeds a certificate-preserving recovery-first baseline by at "
            "least 5% on frozen BurstGPT arrivals mapped to Qwen bounds."
        ),
        "inputs": INPUTS,
        "source_sha256": {key: sha256(ROOT / path) for key, path in INPUTS.items()},
        "grid": grid,
        "expected_grid_points": expected,
        "minimum_effect_pct": 5.0,
        "batch_rule": (
            "At each P180 decision, order requests that arrived after the prior "
            "decision and by the current decision by (arrival_ms, source_order); "
            "screen only the first N fixed by ai_units_per_epoch. No carry, future "
            "arrival, realized service, or policy outcome affects the batch."
        ),
        "slo_rationale": (
            "BurstGPT supplies arrivals and token sizes but no SLO. 1000 ms retains "
            "the prior interactive sensitivity; 250 ms spans roughly two P180 "
            "decisions; 50 and 100 ms are explicitly synthetic near-real-time "
            "expiration sensitivities for AI outputs consumed within a current "
            "radio-control horizon. They are not BurstGPT or production claims."
        ),
        "advance_rule": (
            "Advance a grid point to scheduling-lever evaluation only when it is "
            "mandatory-feasible, oracle value is positive, the oracle-normalized "
            "aggregate gap is at least 5%, and oracle value exceeds recovery-first "
            "in each of the four frozen calibration windows."
        ),
        "stop_rule": "One deterministic screen of every grid point; retain positive, tied, infeasible, and zero-value rows.",
        "exclusion_rule": "No trace window, request within the fixed batch rule, topology, outcome, or grid point may be excluded.",
        "holdout_rule": "Calibration only. Do not open or create a confirmatory workload holdout in this screen.",
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(output), "grid_points": expected}, indent=2))


if __name__ == "__main__":
    main()
