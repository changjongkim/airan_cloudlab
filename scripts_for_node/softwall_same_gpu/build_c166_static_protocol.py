#!/usr/bin/env python3.11
"""Freeze the deterministic C166 static-safe completion before analysis."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
INPUTS = {
    "c159_prespec": "results/softwall_multigpu/confirm159_experiment_prespec_v1.json",
    "c159_q3_result": "results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json",
    "calibration_trace": "data/current/softwall_c159_calibration_burst_windows_v1.json",
    "q2_development_coordinator": "results/softwall_multigpu/raw/confirm159q2f_batch_dev_j58857672_coordinator.json",
    "q2_holdout_coordinator": "results/softwall_multigpu/raw/confirm159q2g_batch_holdout_j58858194_coordinator.json",
    "c159_q3_analyzer": "scripts_for_node/softwall_same_gpu/c159_q3_oracle_screen.py",
    "c166_analyzer": "scripts_for_node/softwall_same_gpu/analyze_c166_static_global_safe.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    prespec = json.loads((ROOT / INPUTS["c159_prespec"]).read_text())
    if "static_global_safe" not in prespec.get("safe_systems", []):
        raise RuntimeError("C159 prespec does not name static_global_safe")
    protocol = {
        "schema": "softwall-c166-static-global-safe-protocol-v1",
        "status": "FROZEN_BEFORE_C166_ANALYSIS",
        "analysis_role": (
            "Complete the prespecified but previously omitted static_global_safe "
            "row and make the C159 structural oracle explicit."
        ),
        "inputs": INPUTS,
        "source_sha256": {
            key: sha256(ROOT / relative) for key, relative in INPUTS.items()
        },
        "frozen_semantics": {
            "static_global_safe": (
                "retain all four 25 ms recovery reservations after the 45 ms cutoff; "
                "admit a Qwen class only if it completes by both the 153 ms guard and "
                "the next 180 ms release"
            ),
            "offline_oracle": (
                "same one-unit-per-epoch capacity and qualified bounds as C159; exact "
                "weighted matching sees all calibration arrivals and uses the earliest "
                "legal completion time"
            ),
            "minimum_effect_pct": 5.0,
        },
        "gates": {
            "reproduce_original_three_rows": True,
            "static_row_reported_even_if_tied": True,
            "oracle_row_reported": True,
            "confirmatory_holdout_must_remain_unmaterialized": True,
        },
        "stop_rule": "One deterministic pass; preserve PASS or FAIL without parameter changes.",
        "exclusion_rule": "No rows, nodes, requests, or windows may be excluded.",
    }
    output = ROOT / "results/softwall_multigpu/c166_static_global_safe_protocol_v1.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"protocol": str(output), "status": protocol["status"]}, indent=2))


if __name__ == "__main__":
    main()
