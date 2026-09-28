#!/usr/bin/env python3.11
"""Freeze the C159/C166 deadline correction before recomputation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
INPUTS = {
    "calibration_trace": "data/current/softwall_c159_calibration_burst_windows_v1.json",
    "c159_prespec": "results/softwall_multigpu/confirm159_experiment_prespec_v1.json",
    "c159_q3_result": "results/softwall_multigpu/confirm159_q3_oracle_screen_v1.json",
    "c166_result": "results/softwall_multigpu/c166_static_global_safe_result_v1.json",
    "q2_development_coordinator": "results/softwall_multigpu/raw/confirm159q2f_batch_dev_j58857672_coordinator.json",
    "q2_holdout_coordinator": "results/softwall_multigpu/raw/confirm159q2g_batch_holdout_j58858194_coordinator.json",
    "c159_q3_analyzer": "scripts_for_node/softwall_same_gpu/c159_q3_oracle_screen.py",
    "c166_analyzer": "scripts_for_node/softwall_same_gpu/analyze_c166_static_global_safe.py",
    "c173_analyzer": "scripts_for_node/softwall_same_gpu/analyze_c173_deadline_correction.py",
    "c173_test": "scripts_for_node/softwall_same_gpu/test_analyze_c173_deadline_correction.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = ROOT / "results/softwall_multigpu/c173_deadline_correction_protocol_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol {output}")
    trace = json.loads((ROOT / INPUTS["calibration_trace"]).read_text(encoding="utf-8"))
    protocol = {
        "schema": "softwall-c173-deadline-correction-protocol-v1",
        "status": "FROZEN_BEFORE_C173_ANALYSIS",
        "analysis_role": (
            "Correct the C159/C166 selector from arrival_ms + deadline_ms to the "
            "trace's absolute deadline_ms field, then report every policy again."
        ),
        "inputs": INPUTS,
        "source_sha256": {key: sha256(ROOT / path) for key, path in INPUTS.items()},
        "frozen_semantics": {
            "deadline": "deadline_ms is absolute within each 60 s trace window",
            "eligibility": "arrival_ms <= slot decision_ms and slot completion_ms <= deadline_ms",
            "policies": [
                "static_global_safe",
                "recovery_first_empirical",
                "recovery_first_full_bound",
                "softwall",
                "offline_oracle",
            ],
            "requests": trace["summary"]["selected_requests"],
            "windows": trace["summary"]["selected_windows"],
            "fixed_slo_ms": trace["selection"]["fixed_slo_ms"],
        },
        "minimum_effect_pct": 5.0,
        "stop_rule": "One deterministic pass; preserve every corrected value even if the prior conclusion changes.",
        "exclusion_rule": "No request, window, policy, node stream, or result may be excluded.",
        "holdout_rule": "Do not materialize the unopened C159 confirmatory workload holdout.",
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(output), "status": protocol["status"]}, indent=2))


if __name__ == "__main__":
    main()
