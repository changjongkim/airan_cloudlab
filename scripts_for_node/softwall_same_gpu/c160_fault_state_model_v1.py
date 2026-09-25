#!/usr/bin/env python3.11
"""Pure state model for C160 integrated outcome/lease fault semantics."""

from __future__ import annotations

import dataclasses
from typing import FrozenSet, Iterable, Optional, Tuple


Key = Tuple[str, str]
CommitKey = Tuple[int, str]


@dataclasses.dataclass(frozen=True)
class Lease:
    lease_id: str
    epoch: int
    latest_start_ns: int
    stage: str = "committed"
    completion_marker: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.lease_id or self.epoch < 0 or self.latest_start_ns < 0:
            raise ValueError("invalid lease")
        if self.stage not in {
            "committed", "launched", "fenced", "nonlaunched_fenced"
        }:
            raise ValueError("invalid lease stage")
        if self.stage in {"fenced", "nonlaunched_fenced"}:
            if self.completion_marker != self.lease_id:
                raise ValueError("terminal physical state requires matching marker")
        elif self.completion_marker is not None:
            raise ValueError("unfenced lease cannot have a completion marker")

    @property
    def fence_confirmed(self) -> bool:
        return self.stage in {"fenced", "nonlaunched_fenced"}


@dataclasses.dataclass(frozen=True)
class State:
    epoch: int
    generation: int
    unresolved: FrozenSet[Key]
    applied_transactions: FrozenSet[str] = frozenset()
    lease: Optional[Lease] = None
    ai_quarantined: bool = False
    physical_launches: int = 0
    physical_fences: int = 0
    retired_leases: int = 0
    fenceless_retires: int = 0
    radio_commits: FrozenSet[CommitKey] = frozenset()

    def __post_init__(self) -> None:
        if (self.epoch < 0 or self.generation < 0 or self.physical_launches < 0
                or self.physical_fences < 0 or self.retired_leases < 0
                or self.fenceless_retires < 0):
            raise ValueError("negative state field")
        if len(self.unresolved) != len(set(self.unresolved)):
            raise ValueError("duplicate unresolved key")
        if self.physical_launches > 1:
            raise ValueError("single-token model permits at most one launch")
        if self.retired_leases > self.physical_fences:
            raise ValueError("lease retirement exceeds physical fence history")


@dataclasses.dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    state: State


def _unchanged(state: State, reason: str, accepted: bool = False) -> Decision:
    return Decision(accepted, reason, state)


def timely_successes(outcomes: Iterable[dict], epoch: int,
                     cutoff_ns: int) -> Tuple[Key, ...]:
    """Return only current-epoch success outcomes physically complete by cutoff."""
    values = []
    for row in outcomes:
        if int(row["epoch"]) != epoch:
            continue
        if int(row["completed_ns"]) > cutoff_ns:
            continue
        if row.get("success") is True:
            values.append(tuple(row["key"]))
    return tuple(values)


def apply_success_batch(state: State, *, event_epoch: int,
                        expected_generation: int,
                        success_keys: Iterable[Key],
                        transaction_id: str) -> Decision:
    """Atomically remove one common-cutoff success set."""
    if transaction_id in state.applied_transactions:
        return _unchanged(state, "duplicate_transaction", accepted=True)
    if event_epoch != state.epoch:
        return _unchanged(state, "stale_or_future_epoch")
    if expected_generation != state.generation:
        return _unchanged(state, "stale_generation")
    successes = tuple(tuple(key) for key in success_keys)
    if len(successes) != len(set(successes)):
        return _unchanged(state, "duplicate_success_in_batch")
    if not set(successes).issubset(state.unresolved):
        return _unchanged(state, "unknown_or_already_resolved_success")
    updated = dataclasses.replace(
        state,
        generation=state.generation + 1,
        unresolved=frozenset(state.unresolved - set(successes)),
        applied_transactions=state.applied_transactions | {transaction_id},
    )
    return Decision(True, "success_batch_applied", updated)


def commit_lease(state: State, *, expected_generation: int, lease_id: str,
                 latest_start_ns: int, transaction_id: str) -> Decision:
    if transaction_id in state.applied_transactions:
        return _unchanged(state, "duplicate_transaction", accepted=True)
    if expected_generation != state.generation:
        return _unchanged(state, "stale_generation")
    if state.ai_quarantined:
        return _unchanged(state, "ai_quarantined")
    if state.lease is not None:
        return _unchanged(state, "lease_already_held")
    lease = Lease(lease_id, state.epoch, latest_start_ns)
    updated = dataclasses.replace(
        state,
        generation=state.generation + 1,
        lease=lease,
        applied_transactions=state.applied_transactions | {transaction_id},
    )
    return Decision(True, "lease_committed", updated)


def physical_start(state: State, *, lease_id: str, accepted_ns: int) -> Decision:
    lease = state.lease
    if lease is None or lease.lease_id != lease_id:
        return _unchanged(state, "unknown_lease")
    if lease.stage != "committed":
        return _unchanged(state, "physical_start_already_decided", accepted=True)
    if accepted_ns > lease.latest_start_ns:
        nonlaunch = dataclasses.replace(
            lease, stage="nonlaunched_fenced", completion_marker=lease.lease_id
        )
        return Decision(True, "latest_start_expired_nonlaunch", dataclasses.replace(
            state, lease=nonlaunch, physical_fences=state.physical_fences + 1
        ))
    launched = dataclasses.replace(lease, stage="launched")
    return Decision(True, "physical_launch", dataclasses.replace(
        state, lease=launched, physical_launches=state.physical_launches + 1
    ))


def publish_completion_fence(state: State, *, lease_id: str) -> Decision:
    lease = state.lease
    if lease is None or lease.lease_id != lease_id:
        return _unchanged(state, "unknown_lease")
    if lease.stage == "fenced":
        return _unchanged(state, "duplicate_completion_fence", accepted=True)
    if lease.stage != "launched":
        return _unchanged(state, "completion_before_launch")
    fenced = dataclasses.replace(
        lease, stage="fenced", completion_marker=lease.lease_id
    )
    return Decision(True, "completion_fence_published", dataclasses.replace(
        state, lease=fenced, physical_fences=state.physical_fences + 1
    ))


def retire_lease(state: State, *, expected_generation: int,
                 lease_id: str, transaction_id: str) -> Decision:
    if transaction_id in state.applied_transactions:
        return _unchanged(state, "duplicate_transaction", accepted=True)
    if expected_generation != state.generation:
        return _unchanged(state, "stale_generation")
    lease = state.lease
    if lease is None or lease.lease_id != lease_id:
        return _unchanged(state, "unknown_lease")
    if not lease.fence_confirmed:
        return _unchanged(state, "physical_fence_required")
    updated = dataclasses.replace(
        state,
        generation=state.generation + 1,
        lease=None,
        retired_leases=state.retired_leases + 1,
        applied_transactions=state.applied_transactions | {transaction_id},
    )
    return Decision(True, "lease_retired", updated)


def handle_rpc_timeout(state: State, *, lease_id: str,
                       observed_marker: Optional[str],
                       transaction_id: str) -> Decision:
    """Quarantine AI and retire only a matching already-published fence."""
    if transaction_id in state.applied_transactions:
        return _unchanged(state, "duplicate_transaction", accepted=True)
    lease = state.lease
    if lease is None or lease.lease_id != lease_id:
        return _unchanged(state, "unknown_lease")
    transactions = state.applied_transactions | {transaction_id}
    if lease.fence_confirmed and observed_marker == lease.lease_id:
        updated = dataclasses.replace(
            state,
            generation=state.generation + 1,
            lease=None,
            ai_quarantined=True,
            retired_leases=state.retired_leases + 1,
            applied_transactions=transactions,
        )
        return Decision(True, "timeout_reconciled_from_matching_fence", updated)
    updated = dataclasses.replace(
        state, ai_quarantined=True, applied_transactions=transactions
    )
    return Decision(False, "timeout_ambiguous_lease_retained", updated)


def commit_radio(state: State, *, event_epoch: int, request_id: str,
                 expected_generation: int) -> Decision:
    key = (event_epoch, request_id)
    if event_epoch != state.epoch:
        return _unchanged(state, "stale_or_future_epoch")
    if expected_generation != state.generation:
        return _unchanged(state, "stale_generation")
    if key in state.radio_commits:
        return _unchanged(state, "duplicate_radio_commit", accepted=True)
    return Decision(True, "radio_committed", dataclasses.replace(
        state, radio_commits=state.radio_commits | {key}
    ))


def safety_invariants(state: State) -> dict[str, bool]:
    return {
        "single_physical_launch": state.physical_launches <= 1,
        "single_radio_commit_per_request": (
            len(state.radio_commits) == len(set(state.radio_commits))
        ),
        "every_retire_has_physical_fence": (
            state.fenceless_retires == 0
            and state.retired_leases <= state.physical_fences
        ),
        "quarantine_blocks_only_ai_state": (
            not state.ai_quarantined or state.generation >= 0
        ),
    }
