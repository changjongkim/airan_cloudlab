#!/usr/bin/env python3.11
"""Fail-closed durable reconciliation model for an ambiguous AI lease.

The controller may lose its channel after granting a lease while the worker
has zero, one, or a completed physical launch.  Reconnection is allowed to
release the lease only from a matching durable terminal record.  A prepared
record may be converted to a durable non-launch fence while the worker token
lock is quiesced.  An absent or launched record remains ambiguous.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Optional


STAGES = ("prepared", "launched", "fenced", "nonlaunch_fenced")
TERMINAL_STAGES = frozenset({"fenced", "nonlaunch_fenced"})


def digest_payload(value: dict) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(raw).hexdigest()


@dataclasses.dataclass(frozen=True)
class LeaseIdentity:
    token: str
    lifecycle_epoch: int
    payload_sha256: str
    worker_epoch: str

    def __post_init__(self) -> None:
        if (not self.token or self.lifecycle_epoch < 0
                or len(self.payload_sha256) != 64 or not self.worker_epoch):
            raise ValueError("invalid lease identity")


@dataclasses.dataclass(frozen=True)
class DurableRecord:
    identity: LeaseIdentity
    stage: str
    journal_seq: int
    physical_launch_count: int = 0
    physical_fence_count: int = 0

    def __post_init__(self) -> None:
        if self.stage not in STAGES or self.journal_seq <= 0:
            raise ValueError("invalid durable record")
        expected = {
            "prepared": (0, 0),
            "launched": (1, 0),
            "fenced": (1, 1),
            "nonlaunch_fenced": (0, 1),
        }[self.stage]
        if (self.physical_launch_count, self.physical_fence_count) != expected:
            raise ValueError("record counters do not match its stage")


@dataclasses.dataclass(frozen=True)
class ReconnectState:
    lifecycle_epoch: int
    generation: int = 0
    lease: Optional[LeaseIdentity] = None
    durable_record: Optional[DurableRecord] = None
    ai_quarantined: bool = False
    retired_tokens: frozenset[str] = frozenset()
    total_physical_launches: int = 0
    total_physical_fences: int = 0
    total_retirements: int = 0

    def __post_init__(self) -> None:
        if min(self.lifecycle_epoch, self.generation, self.total_physical_launches,
               self.total_physical_fences, self.total_retirements) < 0:
            raise ValueError("negative reconnect state field")
        if self.total_retirements > self.total_physical_fences:
            raise ValueError("retirement without a physical/nonlaunch fence")
        if self.durable_record is not None and self.lease is None:
            raise ValueError("active durable record requires an active lease")


@dataclasses.dataclass(frozen=True)
class Decision:
    accepted: bool
    reason: str
    state: ReconnectState


def _same_identity(left: LeaseIdentity, right: LeaseIdentity) -> bool:
    return left == right


def issue_lease(state: ReconnectState, identity: LeaseIdentity) -> Decision:
    if state.ai_quarantined:
        return Decision(False, "ai_quarantined", state)
    if state.lease is not None:
        return Decision(False, "lease_already_held", state)
    if identity.lifecycle_epoch != state.lifecycle_epoch:
        return Decision(False, "stale_lifecycle_epoch", state)
    if identity.token in state.retired_tokens:
        return Decision(False, "retired_token_reuse", state)
    return Decision(True, "lease_issued", dataclasses.replace(
        state, generation=state.generation + 1, lease=identity,
    ))


def worker_prepare(state: ReconnectState, identity: LeaseIdentity) -> Decision:
    if state.lease is None or not _same_identity(state.lease, identity):
        return Decision(False, "lease_identity_mismatch", state)
    record = state.durable_record
    if record is not None:
        if _same_identity(record.identity, identity):
            return Decision(True, "duplicate_prepare", state)
        return Decision(False, "durable_identity_conflict", state)
    prepared = DurableRecord(identity, "prepared", journal_seq=1)
    return Decision(True, "prepare_durable_before_ack", dataclasses.replace(
        state, durable_record=prepared,
    ))


def worker_launch(state: ReconnectState, identity: LeaseIdentity) -> Decision:
    record = state.durable_record
    if record is None or not _same_identity(record.identity, identity):
        return Decision(False, "durable_prepare_required", state)
    if record.stage in {"launched", "fenced"}:
        return Decision(True, "duplicate_launch_noop", state)
    if record.stage != "prepared":
        return Decision(False, "terminal_nonlaunch_cannot_launch", state)
    launched = DurableRecord(
        identity, "launched", journal_seq=record.journal_seq + 1,
        physical_launch_count=1,
    )
    return Decision(True, "launch_journaled_before_submit", dataclasses.replace(
        state, durable_record=launched,
        total_physical_launches=state.total_physical_launches + 1,
    ))


def worker_fence(state: ReconnectState, identity: LeaseIdentity) -> Decision:
    record = state.durable_record
    if record is None or not _same_identity(record.identity, identity):
        return Decision(False, "launched_record_required", state)
    if record.stage == "fenced":
        return Decision(True, "duplicate_fence_noop", state)
    if record.stage != "launched":
        return Decision(False, "physical_launch_required", state)
    fenced = DurableRecord(
        identity, "fenced", journal_seq=record.journal_seq + 1,
        physical_launch_count=1, physical_fence_count=1,
    )
    return Decision(True, "completion_fence_durable_before_reply", dataclasses.replace(
        state, durable_record=fenced,
        total_physical_fences=state.total_physical_fences + 1,
    ))


def lose_channel(state: ReconnectState) -> Decision:
    if state.lease is None:
        return Decision(False, "no_active_lease", state)
    if state.ai_quarantined:
        return Decision(True, "duplicate_channel_loss", state)
    return Decision(True, "ambiguous_channel_loss", dataclasses.replace(
        state, ai_quarantined=True,
    ))


def reconnect_and_reconcile(
    state: ReconnectState, *, observed: Optional[DurableRecord],
    worker_quiesced: bool,
) -> Decision:
    """Reconcile under the worker's per-token launch lock.

    `worker_quiesced` means the query and any prepared->nonlaunch transition
    run while no launcher can advance this token concurrently.  It does not
    prove that an already submitted kernel is complete.
    """
    identity = state.lease
    if identity is None:
        return Decision(False, "no_active_lease", state)
    if not state.ai_quarantined:
        return Decision(False, "reconnect_requires_quarantine", state)
    if not worker_quiesced:
        return Decision(False, "worker_token_lock_required", state)
    if observed is None:
        return Decision(False, "absent_record_is_ambiguous", state)
    if not _same_identity(identity, observed.identity):
        return Decision(False, "observed_identity_mismatch", state)
    local = state.durable_record
    if local is not None:
        if not _same_identity(local.identity, observed.identity):
            return Decision(False, "local_identity_mismatch", state)
        if observed.journal_seq < local.journal_seq:
            return Decision(False, "journal_rollback_detected", state)
        legal_successors = {
            "prepared": {"prepared", "launched", "fenced", "nonlaunch_fenced"},
            "launched": {"launched", "fenced"},
            "fenced": {"fenced"},
            "nonlaunch_fenced": {"nonlaunch_fenced"},
        }
        minimum_delta = {
            ("prepared", "prepared"): 0,
            ("prepared", "launched"): 1,
            ("prepared", "fenced"): 2,
            ("prepared", "nonlaunch_fenced"): 1,
            ("launched", "launched"): 0,
            ("launched", "fenced"): 1,
            ("fenced", "fenced"): 0,
            ("nonlaunch_fenced", "nonlaunch_fenced"): 0,
        }
        if observed.stage not in legal_successors[local.stage]:
            return Decision(False, "illegal_journal_stage_transition", state)
        if (observed.journal_seq - local.journal_seq
                < minimum_delta[(local.stage, observed.stage)]):
            return Decision(False, "journal_sequence_gap_too_small", state)
    if observed.stage == "prepared":
        terminal = DurableRecord(
            identity, "nonlaunch_fenced", observed.journal_seq + 1,
            physical_fence_count=1,
        )
        resolved = dataclasses.replace(
            state, durable_record=None, lease=None, ai_quarantined=False,
            generation=state.generation + 1,
            retired_tokens=state.retired_tokens | {identity.token},
            total_physical_fences=state.total_physical_fences + 1,
            total_retirements=state.total_retirements + 1,
        )
        # `terminal` construction validates the non-launch fence semantics;
        # the retired token remains in the controller's durable tombstone set.
        _ = terminal
        return Decision(True, "prepared_aborted_with_nonlaunch_fence", resolved)
    if observed.stage == "launched":
        updated = dataclasses.replace(state, durable_record=observed)
        return Decision(False, "launched_without_fence_retained", updated)
    if observed.stage in TERMINAL_STAGES:
        added_fence = 0
        if local is None or local.stage not in TERMINAL_STAGES:
            added_fence = 1
        resolved = dataclasses.replace(
            state, durable_record=None, lease=None, ai_quarantined=False,
            generation=state.generation + 1,
            retired_tokens=state.retired_tokens | {identity.token},
            total_physical_fences=state.total_physical_fences + added_fence,
            total_retirements=state.total_retirements + 1,
        )
        return Decision(True, "terminal_record_reconciled", resolved)
    raise AssertionError("unreachable durable stage")


def safety_invariants(state: ReconnectState) -> dict[str, bool]:
    record = state.durable_record
    return {
        "at_most_one_physical_launch_per_active_token": (
            record is None or record.physical_launch_count <= 1
        ),
        "retire_only_after_fence": (
            state.total_retirements <= state.total_physical_fences
        ),
        "quarantine_blocks_new_lease": (
            not state.ai_quarantined or state.lease is not None
        ),
        "retired_token_not_active": (
            state.lease is None or state.lease.token not in state.retired_tokens
        ),
    }
