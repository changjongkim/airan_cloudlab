#!/usr/bin/env python3.11
"""C161 phase-2 event-fault semantics shared by coordinator and analyzer."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import time
from pathlib import Path
from typing import Iterable

from c160_fault_state_model_v1 import State, apply_success_batch, commit_radio


ARMS = (
    "stale_duplicate_nrx",
    "post_fence_reply_delay",
    "pre_fence_channel_loss",
    "stale_duplicate_recovery",
)
PHASE2_CONTEXTS = (16, 128, 512)


def marker_paths(directory: Path, lease_id: str) -> tuple[Path, Path]:
    token = hashlib.sha256(lease_id.encode()).hexdigest()[:24]
    return (
        directory / f"{token}.launch.json",
        directory / f"{token}.complete.json",
    )


def parse_arm_order(text: str) -> tuple[str, ...]:
    arms = tuple(value for value in text.split(",") if value)
    if len(arms) != len(ARMS) or set(arms) != set(ARMS):
        raise ValueError(f"arm order must contain each phase-2 arm once: {ARMS}")
    return arms


def periodic_fault_due(sequence: int, injected: int, *, interval: int,
                       target: int) -> bool:
    if sequence <= 0 or injected < 0 or interval <= 0 or target <= 0:
        raise ValueError("invalid periodic fault arguments")
    return injected < target and sequence >= 1 + injected * interval


def _state_dict(state: State) -> dict:
    value = dataclasses.asdict(state)
    value["unresolved"] = sorted([list(key) for key in state.unresolved])
    value["applied_transactions"] = sorted(state.applied_transactions)
    value["radio_commits"] = sorted([list(key) for key in state.radio_commits])
    return value


def audit_nrx_batch(sequence: int, unresolved: Iterable[tuple[str, str]],
                    successes: Iterable[tuple[str, str]], *, inject: bool) -> dict:
    """Apply the physical current batch, then prove duplicate/stale no-op."""
    base = State(
        epoch=sequence,
        generation=0,
        unresolved=frozenset(tuple(key) for key in unresolved),
    )
    transaction_id = f"nrx-{sequence}"
    current = apply_success_batch(
        base,
        event_epoch=sequence,
        expected_generation=base.generation,
        success_keys=tuple(tuple(key) for key in successes),
        transaction_id=transaction_id,
    )
    if not current.accepted:
        raise RuntimeError(f"current NeuralRx outcome batch rejected: {current.reason}")
    result = {
        "current": {
            "accepted": current.accepted,
            "reason": current.reason,
            "before": _state_dict(base),
            "after": _state_dict(current.state),
        },
        "injected": inject,
        "duplicate": None,
        "stale": None,
    }
    if not inject:
        return result
    duplicate = apply_success_batch(
        current.state,
        event_epoch=sequence,
        expected_generation=current.state.generation,
        success_keys=tuple(tuple(key) for key in successes),
        transaction_id=transaction_id,
    )
    stale = apply_success_batch(
        current.state,
        event_epoch=max(0, sequence - 1),
        expected_generation=current.state.generation,
        success_keys=tuple(tuple(key) for key in successes),
        transaction_id=f"nrx-stale-{sequence}",
    )
    result["duplicate"] = {
        "accepted": duplicate.accepted,
        "reason": duplicate.reason,
        "state_unchanged": duplicate.state == current.state,
    }
    result["stale"] = {
        "accepted": stale.accepted,
        "reason": stale.reason,
        "state_unchanged": stale.state == current.state,
    }
    return result


def audit_recovery_response(sequence: int, request_id: str, *, inject: bool) -> dict:
    """Commit one physical response, then prove duplicate/stale event no-op."""
    base = State(epoch=sequence, generation=0, unresolved=frozenset())
    current = commit_radio(
        base,
        event_epoch=sequence,
        request_id=request_id,
        expected_generation=base.generation,
    )
    if not current.accepted:
        raise RuntimeError(f"current recovery response rejected: {current.reason}")
    result = {
        "current": {
            "accepted": current.accepted,
            "reason": current.reason,
            "before": _state_dict(base),
            "after": _state_dict(current.state),
        },
        "injected": inject,
        "duplicate": None,
        "stale": None,
    }
    if not inject:
        return result
    duplicate = commit_radio(
        current.state,
        event_epoch=sequence,
        request_id=request_id,
        expected_generation=current.state.generation,
    )
    stale = commit_radio(
        current.state,
        event_epoch=max(0, sequence - 1),
        request_id=request_id,
        expected_generation=current.state.generation,
    )
    result["duplicate"] = {
        "accepted": duplicate.accepted,
        "reason": duplicate.reason,
        "state_unchanged": duplicate.state == current.state,
    }
    result["stale"] = {
        "accepted": stale.accepted,
        "reason": stale.reason,
        "state_unchanged": stale.state == current.state,
    }
    return result


def wait_for_matching_marker(path: Path, lease_id: str, deadline_ns: int) -> dict | None:
    """Return only an atomically published marker for the exact lease."""
    while True:
        try:
            candidate = json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            candidate = None
        if candidate is not None and candidate.get("lease_id") == lease_id:
            return candidate
        if time.perf_counter_ns() > deadline_ns:
            return None
        time.sleep(0.0002)
