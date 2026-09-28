#!/usr/bin/env python3.11
"""Freeze C167-v2 after preserving the v1 all-fail-feasibility failure."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
MODEL = ROOT / "scripts_for_node/softwall_same_gpu/c167_reclaimable_capacity_model_v2.py"
V1 = ROOT / "results/softwall_multigpu/c167_reclaimable_capacity_model_v1.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    failure = json.loads(V1.read_text())
    if failure.get("status") != "C167_CAPACITY_MODEL_FAIL" or failure["exact_checker"]["mismatches"] != 144:
        raise RuntimeError("C167-v1 failure history mismatch")
    protocol = {
        "schema": "softwall-c167-capacity-model-protocol-v2",
        "status": "FROZEN_BEFORE_C167_V2_MODEL",
        "predecessor": {
            "path": str(V1.relative_to(ROOT)),
            "sha256": sha256(V1),
            "status": failure["status"],
            "exact_mismatches": 144,
            "diagnosis": (
                "v1 counted slack on an idle GPU even when another lane's initial all-fail "
                "schedule was mandatory-infeasible"
            ),
        },
        "correction": (
            "Apply the global initial all-fail feasibility gate before every safe policy. "
            "Mandatory-infeasible points expose no certified AI capacity."
        ),
        "minimum_effect_pct": 5.0,
        "grid": {
            "cells": [2, 4, 8, 16],
            "nrx_success_probability": [0.2, 0.5, 0.8],
            "failure_correlation": [0.0, 0.5, 1.0],
            "radio_expiry_ms": [100, 155, 220],
            "recovery_bound_ms": [12, 25],
            "ai_context": [64, 256],
            "ai_deadline_from_decision_ms": [50, 100, 1000],
            "gpus": [1, 2, 4],
            "homes": [1, 2, 4],
            "ai_units_per_epoch": [1, 4],
        },
        "ai_deadline_rationale": {
            "50_ms": (
                "synthetic fast closed-loop class representing channel/link-state inference "
                "or anomaly response; Qwen is only the bounded GPU service surrogate"
            ),
            "100_ms": "interactive near-real-time control sensitivity",
            "1000_ms": "the existing C159 BurstGPT/Qwen service objective",
            "non_claim": "These are sensitivity classes, not production AI-RAN SLO claims.",
        },
        "policies": [
            "static_global_safe", "debt_blind_current_idle",
            "certificate_preserving_recovery_first", "softwall", "offline_oracle",
        ],
        "physical_validation_points": [
            {"id": "E4_debt_blind_two_debts_context256", "cells": 4, "homes": 2,
             "gpus": 1, "decision_ms": 45, "unresolved_debts": 2,
             "expiry_ms": 155, "guard_ms": 2, "recovery_bound_ms": 25,
             "context": 256, "expected_debt_blind_guard_excess_ms": 12,
             "expected_softwall": "reject"},
            {"id": "E6a_latest_safe_context64", "cells": 4, "homes": 2,
             "gpus": 1, "decision_ms": 88, "unresolved_debts": 1,
             "expiry_ms": 155, "guard_ms": 2, "recovery_bound_ms": 25,
             "context": 64, "expected_guard_excess_ms": 0,
             "expected_softwall": "admit"},
            {"id": "E6b_first_unsafe_context64", "cells": 4, "homes": 2,
             "gpus": 1, "decision_ms": 89, "unresolved_debts": 1,
             "expiry_ms": 155, "guard_ms": 2, "recovery_bound_ms": 25,
             "context": 64, "expected_shadow_guard_excess_ms": 1,
             "expected_softwall": "reject"},
            {"id": "two_gpu_balanced_two_debts_context64", "cells": 4, "homes": 2,
             "gpus": 2, "decision_ms": 45, "unresolved_by_gpu": [1, 1],
             "expiry_ms": 155, "guard_ms": 2, "recovery_bound_ms": 25,
             "context": 64, "expected_softwall": "admit"},
        ],
        "validation_order": [
            "analytical grid", "exhaustive small-state checker",
            "two-node physical E4/E6 diagnostic", "multi-GPU representative point",
        ],
        "stop_rule": (
            "One frozen corrected grid. Preserve PASS or FAIL; no grid, threshold, or "
            "physical point changes after opening the result."
        ),
        "exclusion_rule": "No analytical state may be excluded.",
        "source_sha256": {"model": sha256(MODEL)},
    }
    output = ROOT / "results/softwall_multigpu/c167_capacity_model_protocol_v2.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"protocol": str(output), "status": protocol["status"]}, indent=2))


if __name__ == "__main__":
    main()
