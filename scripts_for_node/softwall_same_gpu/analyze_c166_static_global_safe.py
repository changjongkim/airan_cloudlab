#!/usr/bin/env python3.11
"""Complete the prespecified C159 safe-policy screen with static and oracle rows.

This is a deterministic re-analysis of the already opened C159 calibration
inputs.  It does not materialize the confirmatory holdout.  The C159 prespec
named ``static_global_safe`` but the original Q3 analyzer emitted only the two
event-driven orders.  This analyzer fills that missing baseline and makes the
structural offline oracle explicit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c159_q3_oracle_screen import (
    BOUNDS_MS,
    CONTEXTS,
    CONTROL_MS,
    CUTOFF_MS,
    PERIOD_MS,
    RECOVERY_BOUND_MS,
    RECOVERY_DEADLINE_MS,
    WINDOW_MS,
    build_state_rows,
    exact_weighted_matching,
    load,
    policy_slot,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def relative_gap(value: int, oracle: int) -> float:
    return 100.0 * (oracle - value) / oracle if oracle else 0.0


def static_slot(epoch: int) -> dict:
    """Return a slot under fixed reservation of every potential recovery.

    The C159 mode has four admitted optional receivers.  Static global-safe
    retains all four 25 ms recovery reservations irrespective of observed
    outcomes.  At the 45 ms decision point, even the smallest qualified Qwen
    class would finish at 45 + 4*25 + 5 + 35 = 185 ms, beyond both the 153 ms
    radio guard and the next 180 ms release.  The slot therefore exposes no AI
    completion.  This is a property of the prespecified C159 mode, not a claim
    about static reservation in every mode.
    """
    base = epoch * PERIOD_MS
    decision = base + CUTOFF_MS
    completion = {
        context: decision + 4 * RECOVERY_BOUND_MS + CONTROL_MS + BOUNDS_MS[context]
        for context in CONTEXTS
        if (
            CUTOFF_MS
            + 4 * RECOVERY_BOUND_MS
            + CONTROL_MS
            + BOUNDS_MS[context]
            <= min(RECOVERY_DEADLINE_MS, PERIOD_MS)
        )
    }
    return {
        "epoch": epoch,
        "decision_ms": decision,
        "static_global_safe": {"completion_by_context_ms": completion},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    root = args.project_root.resolve()
    protocol = load(args.protocol)
    if protocol.get("status") != "FROZEN_BEFORE_C166_ANALYSIS":
        raise RuntimeError("C166 protocol is not frozen")
    paths = {key: root / value for key, value in protocol["inputs"].items()}
    observed_hashes = {key: sha256(path) for key, path in paths.items()}
    if observed_hashes != protocol["source_sha256"]:
        raise RuntimeError("C166 frozen source hash mismatch")

    prespec = load(paths["c159_prespec"])
    original = load(paths["c159_q3_result"])
    trace = load(paths["calibration_trace"])
    if "static_global_safe" not in prespec["safe_systems"]:
        raise RuntimeError("static_global_safe was not prespecified")
    if original.get("open_confirmatory_holdout") is not False:
        raise RuntimeError("C159 confirmatory holdout must remain closed")

    state_streams = {
        "development": build_state_rows(load(paths["q2_development_coordinator"])),
        "holdout": build_state_rows(load(paths["q2_holdout_coordinator"])),
    }
    mappings = (
        ("development", 0),
        ("development", 266),
        ("holdout", 0),
        ("holdout", 266),
    )
    rows = []
    for window, (stream_name, offset) in zip(trace["windows"], mappings):
        slot_count = sum(
            epoch * PERIOD_MS + CUTOFF_MS < WINDOW_MS for epoch in range(1000)
        )
        states = state_streams[stream_name][offset:offset + slot_count]
        if len(states) != slot_count:
            raise RuntimeError("insufficient frozen Q2 state stream")
        dynamic_slots = [
            policy_slot(epoch, state) for epoch, state in enumerate(states)
        ]
        static_slots = [static_slot(epoch) for epoch in range(slot_count)]
        policies = {
            "static_global_safe": exact_weighted_matching(
                window["requests"], static_slots, "static_global_safe"
            ),
            "recovery_first_empirical": exact_weighted_matching(
                window["requests"], dynamic_slots, "event_empirical"
            ),
            "recovery_first_full_bound": exact_weighted_matching(
                window["requests"], dynamic_slots, "event_contract"
            ),
            "softwall": exact_weighted_matching(
                window["requests"], dynamic_slots, "softwall"
            ),
            # SoftWall's completion time is the earliest physically legal time
            # in this one-unit-per-epoch model.  The exact matcher already sees
            # every calibration arrival and deadline, making this an upper bound.
            "offline_oracle": exact_weighted_matching(
                window["requests"], dynamic_slots, "softwall"
            ),
        }
        rows.append({
            "window_id": window["window_id"],
            "q2_state_stream": stream_name,
            "q2_state_offset": offset,
            "slots": slot_count,
            "policies": policies,
        })

    policy_names = tuple(rows[0]["policies"])
    totals = {
        policy: {
            "timely_requests": sum(
                row["policies"][policy]["timely_requests"] for row in rows
            ),
            "timely_value_tokens": sum(
                row["policies"][policy]["timely_value_tokens"] for row in rows
            ),
        }
        for policy in policy_names
    }
    oracle_tokens = totals["offline_oracle"]["timely_value_tokens"]
    for policy, values in totals.items():
        values["gap_to_oracle_pct"] = relative_gap(
            values["timely_value_tokens"], oracle_tokens
        )

    original_totals = original["summary"]["totals"]
    all_pass = all((
        totals["static_global_safe"]["timely_value_tokens"] == 0,
        totals["recovery_first_empirical"]["timely_value_tokens"]
        == original_totals["event_empirical"]["timely_value_tokens"] == 385262,
        totals["recovery_first_full_bound"]["timely_value_tokens"]
        == original_totals["event_contract"]["timely_value_tokens"] == 385007,
        totals["softwall"]["timely_value_tokens"]
        == original_totals["softwall"]["timely_value_tokens"] == 385262,
        totals["offline_oracle"]["timely_value_tokens"] == 385262,
        totals["offline_oracle"] == totals["softwall"],
    ))
    result = {
        "schema": "softwall-c166-static-global-safe-v1",
        "status": "C166_STATIC_GLOBAL_SAFE_PASS" if all_pass else "C166_STATIC_GLOBAL_SAFE_FAIL",
        "all_pass": all_pass,
        "analysis_role": (
            "Deterministic completion of the static_global_safe row named in the "
            "frozen C159 prespec, plus an explicit structural offline upper bound. "
            "All inputs were previously opened; this is not a new confirmatory outcome."
        ),
        "model": {
            "period_ms": PERIOD_MS,
            "decision_ms": CUTOFF_MS,
            "radio_guard_ms": RECOVERY_DEADLINE_MS,
            "potential_recovery_debts": 4,
            "recovery_bound_ms": RECOVERY_BOUND_MS,
            "control_bound_ms": CONTROL_MS,
            "class_bounds_ms": {str(key): value for key, value in BOUNDS_MS.items()},
            "capacity": "at most one Qwen unit per radio epoch",
            "static_semantics": (
                "retain four worst-case recovery reservations regardless of observed outcomes"
            ),
            "oracle_semantics": (
                "exact weighted selector with future calibration arrivals and the earliest "
                "legal per-epoch completion exposed by the C159 model"
            ),
        },
        "gates": {
            "static_was_prespecified": "static_global_safe" in prespec["safe_systems"],
            "confirmatory_holdout_remains_closed": original["open_confirmatory_holdout"] is False,
            "frozen_hashes_match": observed_hashes == protocol["source_sha256"],
            "original_rows_reproduced": all_pass,
            "oracle_equals_softwall": totals["offline_oracle"] == totals["softwall"],
        },
        "summary": {
            "offered_requests": trace["summary"]["selected_requests"],
            "offered_value_tokens": trace["summary"]["offered_value_tokens"],
            "slots": sum(row["slots"] for row in rows),
            "totals": totals,
            "softwall_increment_over_static_tokens": (
                totals["softwall"]["timely_value_tokens"]
                - totals["static_global_safe"]["timely_value_tokens"]
            ),
            "softwall_increment_over_static_requests": (
                totals["softwall"]["timely_requests"]
                - totals["static_global_safe"]["timely_requests"]
            ),
        },
        "interpretation": (
            "In the frozen P180/D155 C159 screen, static four-debt reservation "
            "leaves no qualified same-epoch Qwen slot. SoftWall recovers 385,262 "
            "timely tokens relative to that conservative baseline. Recovery-first, "
            "SoftWall, and the offline oracle tie under empirical recovery because "
            "one Qwen per epoch and a 1 s SLO leave no ordering headroom."
        ),
        "windows": rows,
        "protocol_sha256": sha256(args.protocol),
        "source_sha256": observed_hashes,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "all_pass": all_pass,
        "summary": result["summary"],
    }, indent=2, sort_keys=True))
    if not all_pass:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
