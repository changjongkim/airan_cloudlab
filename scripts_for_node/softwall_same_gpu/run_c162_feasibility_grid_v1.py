#!/usr/bin/env python3.11
"""Generate the C162 predictive envelope grid and prespecified points."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
from pathlib import Path

from c162_feasibility_model_v1 import (
    AI_BOUNDS_MS,
    EnvelopePoint,
    exact_flags,
    predict,
)


SOURCES = (
    "c162_feasibility_model_v1.py",
    "test_c162_feasibility_model_v1.py",
    "run_c162_feasibility_grid_v1.py",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def row(point: EnvelopePoint) -> dict:
    prediction = predict(point)
    return {
        "point": dataclasses.asdict(point),
        "prediction": dataclasses.asdict(prediction),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    qualified_rows = []
    mismatch_count = 0
    contexts = (None,) + tuple(AI_BOUNDS_MS)
    for accepted in range(6):
        for unresolved in range(accepted + 1):
            for decision in range(45, 154):
                for context in contexts:
                    point = EnvelopePoint(
                        accepted, unresolved, decision, context
                    )
                    prediction = predict(point)
                    exact_all, exact_current, exact_ai, exact_finish = exact_flags(point)
                    mismatch_count += int(
                        prediction.mandatory_all_fail_safe != exact_all
                        or prediction.current_mandatory_safe != exact_current
                        or prediction.ai_safe != exact_ai
                        or (prediction.ai_safe and prediction.exact_finish_ms != exact_finish)
                    )
                    qualified_rows.append({
                        "accepted_debts": accepted,
                        "unresolved_debts": unresolved,
                        "decision_time_ms": decision,
                        "context_length": context,
                        "state": prediction.state,
                        "reason": prediction.reason,
                        "analytic_finish_ms": prediction.analytic_finish_ms,
                        "exact_finish_ms": exact_finish,
                    })

    provenance_rows = []
    for recovery in (12, 25):
        for lifecycle in ("warm", "cold", "long_idle"):
            for node_status in ("qualified", "failed", "unknown"):
                for fault in (
                    "none", "post_fence_reply_delay", "pre_fence_channel_loss"
                ):
                    for receivers in (4, 6, 8):
                        provenance_rows.append(row(EnvelopePoint(
                            4, 2, 45, 128,
                            recovery_bound_ms=recovery,
                            lifecycle=lifecycle,
                            node_bound_status=node_status,
                            fault_class=fault,
                            resident_receivers_per_home=receivers,
                        )))

    prespecified = {
        "E1_four_unresolved_no_ai": EnvelopePoint(4, 4, 45, None),
        "E2_five_accepted": EnvelopePoint(5, 5, 45, None),
        "E3_two_unresolved_context128": EnvelopePoint(4, 2, 45, 128),
        "E4_two_unresolved_context256": EnvelopePoint(4, 2, 45, 256),
        "E5_one_unresolved_context512": EnvelopePoint(4, 1, 45, 512),
        "E6a_latest_safe_context64": EnvelopePoint(4, 1, 88, 64),
        "E6b_first_unsafe_context64": EnvelopePoint(4, 1, 89, 64),
        "E7a_warm": EnvelopePoint(4, 2, 45, 128),
        "E7b_cold": EnvelopePoint(4, 2, 45, 128, lifecycle="cold"),
        "E8_receiver_oom": EnvelopePoint(
            4, 2, 45, 128, resident_receivers_per_home=8
        ),
        "F1_post_fence_quarantine": EnvelopePoint(
            4, 2, 94, 128, fault_class="post_fence_reply_delay"
        ),
        "F2_pre_fence_quarantine": EnvelopePoint(
            4, 2, 90, 128, fault_class="pre_fence_channel_loss"
        ),
        "U1_nid001044_failed_preflight": EnvelopePoint(
            4, 2, 45, 128, node_bound_status="failed"
        ),
        "U2_conv12_false_qualification": EnvelopePoint(
            4, 2, 45, 128, recovery_bound_ms=12
        ),
    }
    prespecified_rows = {
        name: row(point) for name, point in prespecified.items()
    }
    state_counts = dict(collections.Counter(
        item["state"] for item in qualified_rows
    ))
    provenance_counts = dict(collections.Counter(
        item["prediction"]["state"] for item in provenance_rows
    ))
    expected = {
        "E1_four_unresolved_no_ai": "QSN",
        "E2_five_accepted": "MI",
        "E3_two_unresolved_context128": "QSU",
        "E4_two_unresolved_context256": "QSN",
        "E5_one_unresolved_context512": "QSU",
        "E6a_latest_safe_context64": "QSU",
        "E6b_first_unsafe_context64": "QSN",
        "E7a_warm": "QSU",
        "E7b_cold": "UQ",
        "E8_receiver_oom": "MI",
        "F1_post_fence_quarantine": "QSN",
        "F2_pre_fence_quarantine": "QSN",
        "U1_nid001044_failed_preflight": "UQ",
        "U2_conv12_false_qualification": "UQ",
    }
    gates = {
        "qualified_grid_16023_states": len(qualified_rows) == 16023,
        "analytic_exact_mismatch_zero": mismatch_count == 0,
        "all_four_states_represented_across_grids": (
            set(state_counts) | set(provenance_counts) == {"QSU", "QSN", "MI", "UQ"}
        ),
        "prespecified_points_match": all(
            prespecified_rows[name]["prediction"]["state"] == state
            for name, state in expected.items()
        ),
        "invalid_provenance_never_promoted_safe": all(
            item["prediction"]["state"] not in {"QSU", "QSN"}
            for item in provenance_rows
            if (item["point"]["recovery_bound_ms"] != 25
                or item["point"]["lifecycle"] != "warm"
                or item["point"]["node_bound_status"] != "qualified"
                or 5 <= item["point"]["resident_receivers_per_home"])
        ),
    }
    value = {
        "schema": "softwall-c162-feasibility-grid-v1",
        "status": "C162_MODEL_GRID_PASS" if all(gates.values()) else "C162_MODEL_GRID_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "qualified_grid": {
            "count": len(qualified_rows), "state_counts": state_counts,
            "analytic_exact_mismatch": mismatch_count,
            "axes": {
                "accepted_debts": [0, 5], "unresolved_debts": "0..accepted",
                "decision_time_ms": [45, 153],
                "contexts": list(contexts),
            },
        },
        "provenance_grid": {
            "count": len(provenance_rows), "state_counts": provenance_counts,
            "axes": {
                "recovery_bound_ms": [12, 25],
                "lifecycle": ["warm", "cold", "long_idle"],
                "node_bound_status": ["qualified", "failed", "unknown"],
                "fault_class": ["none", "post_fence_reply_delay", "pre_fence_channel_loss"],
                "receivers_per_home": [4, 6, 8],
            },
        },
        "prespecified_points": prespecified_rows,
        "model_limit": "The classifier predicts scheduling within a mode whose whole-path bounds and provenance are inputs. It does not predict the host-tail mechanism that made nid001044 violate NRx45; that node is UQ until its preflight is qualified.",
        "source_sha256": {
            name: sha256(args.scripts_root / name) for name in SOURCES
        },
        "qualified_rows": qualified_rows,
        "provenance_rows": provenance_rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "qualified_grid": value["qualified_grid"],
                      "provenance_grid": value["provenance_grid"]},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
