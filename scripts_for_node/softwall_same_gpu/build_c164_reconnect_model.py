#!/usr/bin/env python3.11
"""Enumerate C164 reconnect reconciliation cases and emit an audit artifact."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from pathlib import Path

from c164_reconnect_state_model_v1 import (
    DurableRecord, LeaseIdentity, ReconnectState, digest_payload,
    issue_lease, lose_channel, reconnect_and_reconcile, safety_invariants,
    worker_fence, worker_launch, worker_prepare,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_identity(token: str = "lease-1", *, epoch: int = 7,
                  context: int = 128, worker: str = "worker-a") -> LeaseIdentity:
    return LeaseIdentity(token, epoch, digest_payload({"context": context}), worker)


def build_stage(stage: str) -> tuple[ReconnectState, LeaseIdentity, DurableRecord | None]:
    lease = make_identity()
    state = issue_lease(ReconnectState(lifecycle_epoch=7), lease).state
    if stage != "absent":
        state = worker_prepare(state, lease).state
    if stage in {"launched", "fenced"}:
        state = worker_launch(state, lease).state
    if stage == "fenced":
        state = worker_fence(state, lease).state
    state = lose_channel(state).state
    return state, lease, state.durable_record


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    scenarios = []
    violations = []
    allowed_retire = 0
    fail_closed = 0
    for stage in ("absent", "prepared", "launched", "fenced"):
        for identity_case in ("match", "wrong_token", "wrong_payload",
                              "wrong_lifecycle", "wrong_worker"):
            for quiesced in (False, True):
                state, lease, observed = build_stage(stage)
                if observed is not None and identity_case != "match":
                    replacements = {
                        "wrong_token": {"token": "other-token"},
                        "wrong_payload": {"payload_sha256": digest_payload({"context": 512})},
                        "wrong_lifecycle": {"lifecycle_epoch": 8},
                        "wrong_worker": {"worker_epoch": "worker-b"},
                    }[identity_case]
                    observed = dataclasses.replace(
                        observed,
                        identity=dataclasses.replace(observed.identity, **replacements),
                    )
                result = reconnect_and_reconcile(
                    state, observed=observed, worker_quiesced=quiesced
                )
                expected_accept = (
                    quiesced and identity_case == "match"
                    and stage in {"prepared", "fenced"}
                )
                checks = {
                    "decision_matches_rule": result.accepted == expected_accept,
                    "lease_retired_only_when_allowed": (
                        result.state.lease is None
                    ) == expected_accept,
                    "all_safety_invariants": all(
                        safety_invariants(result.state).values()
                    ),
                }
                if expected_accept:
                    allowed_retire += 1
                else:
                    fail_closed += 1
                if not all(checks.values()):
                    violations.append({
                        "stage": stage, "identity_case": identity_case,
                        "worker_quiesced": quiesced, "reason": result.reason,
                        "checks": checks,
                    })
                scenarios.append({
                    "stage": stage, "identity_case": identity_case,
                    "worker_quiesced": quiesced,
                    "accepted": result.accepted, "reason": result.reason,
                    "lease_retired": result.state.lease is None,
                    "checks": checks,
                })
    # Three journal-integrity adversaries outside the Cartesian identity grid.
    launched, lease, record = build_stage("launched")
    adversaries = [
        ("launched_to_nonlaunch", launched, DurableRecord(
            lease, "nonlaunch_fenced", record.journal_seq + 1,
            physical_fence_count=1,
        ), "illegal_journal_stage_transition"),
    ]
    fenced, _, fenced_record = build_stage("fenced")
    adversaries.append((
        "journal_rollback", fenced,
        dataclasses.replace(fenced_record, journal_seq=2),
        "journal_rollback_detected",
    ))
    prepared, _, prepared_record = build_stage("prepared")
    adversaries.append((
        "prepared_to_fenced_without_two_journal_steps", prepared,
        DurableRecord(
            lease, "fenced", prepared_record.journal_seq + 1,
            physical_launch_count=1, physical_fence_count=1,
        ),
        "journal_sequence_gap_too_small",
    ))
    for name, state, observed, expected_reason in adversaries:
        result = reconnect_and_reconcile(
            state, observed=observed, worker_quiesced=True
        )
        checks = {
            "fail_closed": not result.accepted and result.state.lease is not None,
            "expected_reason": result.reason == expected_reason,
            "all_safety_invariants": all(safety_invariants(result.state).values()),
        }
        fail_closed += 1
        if not all(checks.values()):
            violations.append({"adversary": name, "reason": result.reason,
                               "checks": checks})
        scenarios.append({"adversary": name, "accepted": result.accepted,
                          "reason": result.reason, "checks": checks})
    checks = {
        "state_product_complete": len(scenarios) == 43,
        "only_two_matched_quiesced_terminal_resolutions": allowed_retire == 2,
        "all_other_cases_fail_closed": fail_closed == 41,
        "zero_invariant_violations": not violations,
    }
    source = Path(__file__)
    model = source.with_name("c164_reconnect_state_model_v1.py")
    test = source.with_name("test_c164_reconnect_state_model_v1.py")
    value = {
        "schema": "softwall-c164-reconnect-model-audit-v1",
        "status": "C164_RECONNECT_MODEL_PASS" if all(checks.values()) else "C164_RECONNECT_MODEL_FAIL",
        "all_pass": all(checks.values()),
        "checks": checks,
        "counts": {
            "state_product": len(scenarios),
            "allowed_retire": allowed_retire,
            "fail_closed": fail_closed,
            "invariant_violations": len(violations),
        },
        "claim_boundary": (
            "Pure same-worker-epoch durable reconciliation model. Physical "
            "channel reconnect, worker-process replacement, production d_MAC, "
            "and WCET remain unqualified."
        ),
        "source_sha256": {
            source.name: sha256(source), model.name: sha256(model),
            test.name: sha256(test),
        },
        "violations": violations,
        "scenarios": scenarios,
    }
    output = root / "results/softwall_multigpu/c164_reconnect_model_v1.json"
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    print(json.dumps({"status": value["status"], **value["counts"]}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
