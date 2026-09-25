#!/usr/bin/env python3.11
"""Fail-closed model for replacing an MPS Qwen worker process.

The model treats a successful MPS terminate_client command as the GPU-context
quiescence primitive.  OS process exit alone is deliberately insufficient.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ReplacementEvidence:
    journal_stage: str
    identity_match: bool
    target_present_before: bool
    terminate_client_success: bool
    process_exit_confirmed: bool
    target_absent_after: bool
    same_mps_server_survived: bool
    mandatory_client_preserved: bool

    def __post_init__(self) -> None:
        if self.journal_stage not in {"prepared", "launched"}:
            raise ValueError("replacement accepts only ambiguous journal stages")


@dataclasses.dataclass(frozen=True)
class ReplacementDecision:
    accepted: bool
    reason: str
    terminal_stage: str | None
    physical_launch_count: int
    quiescence_fence_count: int


def qualify_replacement(evidence: ReplacementEvidence) -> ReplacementDecision:
    checks = (
        (evidence.identity_match, "identity_mismatch"),
        (evidence.target_present_before, "target_not_observed_before"),
        (evidence.terminate_client_success, "mps_terminate_client_not_successful"),
        (evidence.process_exit_confirmed, "old_process_exit_unconfirmed"),
        (evidence.target_absent_after, "old_client_still_registered"),
        (evidence.same_mps_server_survived, "mps_server_epoch_changed"),
        (evidence.mandatory_client_preserved, "mandatory_client_not_preserved"),
    )
    for passed, reason in checks:
        if not passed:
            return ReplacementDecision(
                False, reason, None,
                int(evidence.journal_stage == "launched"), 0,
            )
    terminal = (
        "nonlaunch_fenced" if evidence.journal_stage == "prepared"
        else "quiescence_fenced"
    )
    return ReplacementDecision(
        True, "mps_context_quiescence_certificate_accepted", terminal,
        int(evidence.journal_stage == "launched"), 1,
    )


def safety_invariants(evidence: ReplacementEvidence,
                      decision: ReplacementDecision) -> dict[str, bool]:
    return {
        "retire_requires_quiescence_fence": (
            not decision.accepted or decision.quiescence_fence_count == 1
        ),
        "launched_token_requires_mps_termination": (
            evidence.journal_stage != "launched" or not decision.accepted
            or evidence.terminate_client_success
        ),
        "process_exit_alone_is_insufficient": (
            not decision.accepted
            or (evidence.target_absent_after
                and evidence.terminate_client_success)
        ),
        "mandatory_client_survives_accepted_replacement": (
            not decision.accepted or evidence.mandatory_client_preserved
        ),
    }
