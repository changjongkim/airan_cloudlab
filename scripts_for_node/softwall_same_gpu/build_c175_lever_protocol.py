#!/usr/bin/env python3.11
"""Freeze C175 lever evaluation after C174 screening and before outcomes."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
INPUTS = {
    "c174_protocol": "results/softwall_multigpu/c174_trace_oracle_protocol_v1.json",
    "c174_result": "results/softwall_multigpu/c174_trace_oracle_screen_v1.json",
    "calibration_trace": "data/current/softwall_c159_calibration_burst_windows_v1.json",
    "sionna_holdout": "results/softwall_same_gpu/sionna_cdl_de_holdout_gate_job58868184.json",
    "c174_source": "scripts_for_node/softwall_same_gpu/c174_trace_oracle_screen.py",
    "c175_source": "scripts_for_node/softwall_same_gpu/c175_lever_evaluation.py",
    "c175_test": "scripts_for_node/softwall_same_gpu/test_c175_lever_evaluation.py",
    "capacity_model": "scripts_for_node/softwall_same_gpu/c167_reclaimable_capacity_model_v2.py",
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    output = ROOT / "results/softwall_multigpu/c175_lever_protocol_v1.json"
    if output.exists():
        raise FileExistsError(f"refusing to overwrite frozen protocol {output}")
    c174 = json.loads((ROOT / INPUTS["c174_result"]).read_text(encoding="utf-8"))
    candidates = sum(row["advance_to_lever_evaluation"] for row in c174["rows"])
    protocol = {
        "schema": "softwall-c175-lever-protocol-v1",
        "status": "FROZEN_BEFORE_C175_ANALYSIS",
        "analysis_role": "Evaluate all and only C174 points that passed its frozen 5% gate.",
        "inputs": INPUTS,
        "source_sha256": {key: sha256(ROOT / path) for key, path in INPUTS.items()},
        "expected_candidate_points": candidates,
        "levers": {
            "ai_first_retiming": (
                "EDF then higher-value tie-break; place before recovery only when "
                "the lane's remaining recovery still completes by the guard; drain "
                "recovery, then greedily place remaining jobs"
            ),
            "gpu_placement_upper": (
                "exactly enumerate every distribution of realized recovery debt "
                "across the existing GPU lanes; retain certificate feasibility"
            ),
            "debt_aware_nrx": (
                "exact subset of the ten frozen Sionna CDL-D/E strata minimizing "
                "expected recovery debt under max-radio minus an absolute loss bound"
            ),
        },
        "radio_loss_sensitivity": [0.0, 0.02],
        "stop_rule": "One deterministic pass over all 362 candidates; preserve every positive and negative lever result.",
        "exclusion_rule": "No C174 candidate, trace window, outcome, or Sionna stratum may be excluded.",
        "claim_boundary": "Development model decomposition only; no confirmatory or physical performance claim.",
    }
    output.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"protocol": str(output), "candidate_points": candidates}, indent=2))


if __name__ == "__main__":
    main()
