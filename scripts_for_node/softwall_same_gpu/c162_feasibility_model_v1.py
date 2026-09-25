#!/usr/bin/env python3.11
"""Predictive QSU/QSN/MI/UQ model for the qualified SoftWall P180 mode."""

from __future__ import annotations

import dataclasses
from typing import Optional

from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedAILease,
    exact_certificate,
)


MS = 1_000_000
AI_BOUNDS_MS = {16: 35, 32: 35, 64: 35, 128: 40, 256: 65, 512: 75}
STATES = ("QSU", "QSN", "MI", "UQ")
QUALIFIED_TOPOLOGIES = {
    "two_home_shared_4debt",
    "single_home_4debt",
    "four_home_3each",
}
QUALIFIED_FAULTS = {
    "none", "post_fence_reply_delay", "pre_fence_channel_loss"
}


@dataclasses.dataclass(frozen=True)
class EnvelopePoint:
    accepted_debts: int
    unresolved_debts: int
    decision_time_ms: int
    context_length: Optional[int]
    lifecycle: str = "warm"
    recovery_bound_ms: int = 25
    nrx_bound_ms: int = 45
    control_bound_ms: int = 5
    period_ms: int = 180
    expiry_ms: int = 155
    guard_ms: int = 2
    capacity: int = 1
    topology: str = "two_home_shared_4debt"
    resident_receivers_per_home: int = 4
    fault_class: str = "none"
    node_bound_status: str = "qualified"

    def validate(self) -> None:
        if (self.accepted_debts < 0 or self.unresolved_debts < 0
                or self.unresolved_debts > self.accepted_debts
                or self.decision_time_ms < 0 or self.recovery_bound_ms <= 0
                or self.nrx_bound_ms <= 0 or self.control_bound_ms <= 0
                or self.period_ms <= 0 or self.expiry_ms <= 0
                or self.guard_ms <= 0 or self.capacity <= 0
                or self.resident_receivers_per_home <= 0):
            raise ValueError("invalid envelope point")
        if self.context_length is not None and self.context_length not in AI_BOUNDS_MS:
            raise ValueError("unqualified context length")

    @property
    def recovery_deadline_ms(self) -> int:
        return self.expiry_ms - self.guard_ms


@dataclasses.dataclass(frozen=True)
class Prediction:
    state: str
    reason: str
    mandatory_all_fail_safe: bool
    current_mandatory_safe: bool
    ai_safe: bool
    analytic_finish_ms: Optional[int]
    exact_finish_ms: Optional[float]


def _obligations(count: int, point: EnvelopePoint):
    return tuple(
        RecoveryObligation(
            home_id=f"home{index % 2}", request_id=f"r{index}",
            release_ns=point.nrx_bound_ms * MS,
            deadline_ns=point.recovery_deadline_ms * MS,
            service_ns=point.recovery_bound_ms * MS,
        )
        for index in range(count)
    )


def exact_flags(point: EnvelopePoint) -> tuple[bool, bool, bool, Optional[float]]:
    """Independent exact small-state oracle for mandatory and AI schedules."""
    point.validate()
    all_fail = exact_certificate(
        _obligations(point.accepted_debts, point), point.capacity,
        now_ns=point.nrx_bound_ms * MS,
    )
    current = exact_certificate(
        _obligations(point.unresolved_debts, point), point.capacity,
        now_ns=point.decision_time_ms * MS,
    )
    mandatory_all_fail_safe = all_fail is not None
    current_mandatory_safe = current is not None
    # AI is admissible only inside a mode that was valid at initial all-fail
    # admission and remains valid for the current unresolved set.
    if not mandatory_all_fail_safe or not current_mandatory_safe:
        return mandatory_all_fail_safe, current_mandatory_safe, False, None
    if point.context_length is None or point.fault_class != "none":
        return mandatory_all_fail_safe, current_mandatory_safe, False, None
    ai_ms = point.control_bound_ms + AI_BOUNDS_MS[point.context_length]
    lease = SharedAILease(
        lease_id="c162-ai", owner_home="global",
        start_ns=point.decision_time_ms * MS,
        finish_ns=(point.decision_time_ms + ai_ms) * MS,
    )
    # The recovery oracle constrains leases only through their interference
    # with mandatory work.  C162 also requires the bounded AI unit itself to
    # finish by the request expiry guard, including when no debt remains.
    if lease.finish_ns > point.recovery_deadline_ms * MS:
        return mandatory_all_fail_safe, current_mandatory_safe, False, None
    schedule = exact_certificate(
        _obligations(point.unresolved_debts, point), point.capacity,
        leases=(lease,), now_ns=point.decision_time_ms * MS,
    )
    if schedule is None:
        return mandatory_all_fail_safe, current_mandatory_safe, False, None
    finish_ns = max(
        [lease.finish_ns] + [row.finish_ns for row in schedule]
    )
    return mandatory_all_fail_safe, current_mandatory_safe, True, finish_ns / MS


def qualification_reason(point: EnvelopePoint) -> Optional[str]:
    if point.lifecycle != "warm":
        return "lifecycle_bound_unqualified"
    if point.recovery_bound_ms != 25:
        return "recovery_bound_provenance_unqualified"
    if point.nrx_bound_ms != 45 or point.control_bound_ms != 5:
        return "timing_vector_unqualified"
    if point.topology not in QUALIFIED_TOPOLOGIES:
        return "topology_unqualified"
    if point.node_bound_status != "qualified":
        return "node_lifecycle_preflight_unqualified"
    if point.fault_class not in QUALIFIED_FAULTS:
        return "fault_class_unqualified"
    return None


def predict(point: EnvelopePoint) -> Prediction:
    point.validate()
    if point.resident_receivers_per_home >= 8:
        return Prediction("MI", "measured_receiver_residency_oom", False, False,
                          False, None, None)
    if 5 <= point.resident_receivers_per_home < 8:
        return Prediction("UQ", "receiver_residency_unqualified", False, False,
                          False, None, None)
    unqualified = qualification_reason(point)
    if unqualified is not None:
        return Prediction("UQ", unqualified, False, False, False, None, None)

    all_fail_finish = (
        point.nrx_bound_ms
        + ((point.accepted_debts + point.capacity - 1) // point.capacity)
        * point.recovery_bound_ms
    )
    current_finish = (
        max(point.decision_time_ms, point.nrx_bound_ms)
        + ((point.unresolved_debts + point.capacity - 1) // point.capacity)
        * point.recovery_bound_ms
    )
    # An empty obligation set is feasible at every observation time.  The
    # timestamp is retained for reporting, but it is not a missed recovery.
    all_fail_safe = (
        point.accepted_debts == 0
        or all_fail_finish <= point.recovery_deadline_ms
    )
    current_safe = (
        point.unresolved_debts == 0
        or current_finish <= point.recovery_deadline_ms
    )
    exact_all_fail, exact_current, exact_ai, exact_finish = exact_flags(point)
    if all_fail_safe != exact_all_fail or current_safe != exact_current:
        raise AssertionError("analytic mandatory classifier disagrees with exact oracle")
    if not all_fail_safe:
        return Prediction("MI", "initial_all_fail_admission_infeasible", False,
                          current_safe, False, all_fail_finish, None)
    if not current_safe:
        return Prediction("MI", "current_mandatory_recovery_infeasible", True,
                          False, False, current_finish, None)
    if point.fault_class != "none":
        return Prediction("QSN", "fault_quarantine_blocks_new_ai", True, True,
                          False, current_finish, None)
    if point.context_length is None:
        return Prediction("QSN", "no_ai_request", True, True, False,
                          current_finish, None)
    ai_finish = (
        max(point.decision_time_ms, point.nrx_bound_ms)
        + point.control_bound_ms + AI_BOUNDS_MS[point.context_length]
        + ((point.unresolved_debts + point.capacity - 1) // point.capacity)
        * point.recovery_bound_ms
    )
    ai_safe = ai_finish <= point.recovery_deadline_ms
    if ai_safe != exact_ai:
        raise AssertionError("analytic AI classifier disagrees with exact oracle")
    if ai_safe:
        return Prediction("QSU", "qualified_ai_and_mandatory_schedule", True,
                          True, True, ai_finish, exact_finish)
    return Prediction("QSN", "mandatory_safe_ai_class_does_not_fit", True,
                      True, False, ai_finish, None)
