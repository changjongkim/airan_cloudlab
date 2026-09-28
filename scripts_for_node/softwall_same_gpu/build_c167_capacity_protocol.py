#!/usr/bin/env python3.11
"""Freeze the C167 capacity-model grid and physical validation points."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
MODEL = ROOT / "scripts_for_node/softwall_same_gpu/c167_reclaimable_capacity_model.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    protocol = {
        "schema": "softwall-c167-capacity-model-protocol-v1",
        "status": "FROZEN_BEFORE_C167_MODEL",
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
            {
                "id": "E4_debt_blind_two_debts_context256",
                "cells": 4, "homes": 2, "gpus": 1, "decision_ms": 45,
                "unresolved_debts": 2, "expiry_ms": 155, "guard_ms": 2,
                "recovery_bound_ms": 25, "context": 256,
                "expected_debt_blind_guard_excess_ms": 12,
                "expected_softwall": "reject",
            },
            {
                "id": "E6a_latest_safe_context64",
                "cells": 4, "homes": 2, "gpus": 1, "decision_ms": 88,
                "unresolved_debts": 1, "expiry_ms": 155, "guard_ms": 2,
                "recovery_bound_ms": 25, "context": 64,
                "expected_guard_excess_ms": 0,
                "expected_softwall": "admit",
            },
            {
                "id": "E6b_first_unsafe_context64",
                "cells": 4, "homes": 2, "gpus": 1, "decision_ms": 89,
                "unresolved_debts": 1, "expiry_ms": 155, "guard_ms": 2,
                "recovery_bound_ms": 25, "context": 64,
                "expected_shadow_guard_excess_ms": 1,
                "expected_softwall": "reject",
            },
            {
                "id": "two_gpu_balanced_two_debts_context64",
                "cells": 4, "homes": 2, "gpus": 2, "decision_ms": 45,
                "unresolved_by_gpu": [1, 1], "expiry_ms": 155, "guard_ms": 2,
                "recovery_bound_ms": 25, "context": 64,
                "expected_softwall": "admit",
            },
        ],
        "validation_order": [
            "analytical grid", "exhaustive small-state checker",
            "two-node physical E4/E6 diagnostic", "multi-GPU representative point",
        ],
        "stop_rule": (
            "One frozen grid and exact-check pass. Preserve all rows. Physical validation "
            "may start only if exact mismatches are zero; disagreement is reported without retuning."
        ),
        "exclusion_rule": "No analytical state may be excluded.",
        "source_sha256": {"model": sha256(MODEL)},
    }
    output = ROOT / "results/softwall_multigpu/c167_capacity_model_protocol_v1.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(protocol, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"protocol": str(output), "status": protocol["status"]}, indent=2))


if __name__ == "__main__":
    main()
