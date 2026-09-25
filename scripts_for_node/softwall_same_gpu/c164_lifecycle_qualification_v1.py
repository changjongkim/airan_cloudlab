#!/usr/bin/env python3.11
"""Lifecycle-scoped qualification tokens for optional NRx/AI admission."""

from __future__ import annotations

import dataclasses
import hashlib
import json
from typing import Mapping


LIFECYCLES = (
    "warm_persistent", "cold_first", "mps_restart_first",
    "worker_reconnect_first", "idle_30s_first", "idle_5m_first",
    "idle_30m_first", "gc_on", "qwen_reload_first",
)
REQUIRED_COMPONENTS = (
    "nrx", "recovery", "control", "ai16", "ai32", "ai64",
    "ai128", "ai256", "ai512",
)
REQUIRED_FLAGS = (
    "gpu_inventory_match", "mig_disabled", "mps_ready",
    "recovery_echo", "ipc_fence", "single_commit_canary",
)
STATUSES = ("qualifying", "qualified", "stale", "quarantined")


def canonical_digest(value: Mapping) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


@dataclasses.dataclass(frozen=True)
class QualificationProfile:
    mode_id: str
    lifecycle: str
    node_id: str
    hardware_fingerprint: str
    software_fingerprint: str
    placement_fingerprint: str
    bounds_ns: Mapping[str, int]
    minimum_samples: Mapping[str, int]
    max_idle_ns: int

    def __post_init__(self) -> None:
        if (not self.mode_id or self.lifecycle not in LIFECYCLES
                or not self.node_id or len(self.hardware_fingerprint) != 64
                or len(self.software_fingerprint) != 64
                or len(self.placement_fingerprint) != 64
                or self.max_idle_ns <= 0):
            raise ValueError("invalid qualification profile identity")
        if set(self.bounds_ns) != set(REQUIRED_COMPONENTS):
            raise ValueError("qualification profile has incomplete bounds")
        if set(self.minimum_samples) != set(REQUIRED_COMPONENTS):
            raise ValueError("qualification profile has incomplete sample targets")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in self.bounds_ns.values()):
            raise ValueError("bounds must be positive integer nanoseconds")
        if any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
               for value in self.minimum_samples.values()):
            raise ValueError("sample targets must be positive integers")

    @property
    def fingerprint(self) -> str:
        return canonical_digest(dataclasses.asdict(self))


@dataclasses.dataclass(frozen=True)
class QualificationState:
    epoch: int
    generation: int
    profile: QualificationProfile
    status: str = "qualifying"
    sample_counts: Mapping[str, int] = dataclasses.field(default_factory=dict)
    observed_max_ns: Mapping[str, int] = dataclasses.field(default_factory=dict)
    passed_flags: frozenset[str] = frozenset()
    last_activity_ns: int = 0
    reason: str = "new_mode_requires_preflight"

    def __post_init__(self) -> None:
        if (self.epoch < 0 or self.generation < 0 or self.status not in STATUSES
                or self.last_activity_ns < 0):
            raise ValueError("invalid qualification state")
        if not set(self.sample_counts).issubset(REQUIRED_COMPONENTS):
            raise ValueError("unknown sample component")
        if not set(self.observed_max_ns).issubset(REQUIRED_COMPONENTS):
            raise ValueError("unknown observed component")
        if not set(self.passed_flags).issubset(REQUIRED_FLAGS):
            raise ValueError("unknown preflight flag")


@dataclasses.dataclass(frozen=True)
class QualificationDecision:
    accepted: bool
    reason: str
    state: QualificationState
    token: str | None = None


def start_mode(profile: QualificationProfile, epoch: int,
               now_ns: int) -> QualificationState:
    if epoch < 0 or now_ns < 0:
        raise ValueError("invalid mode start")
    return QualificationState(
        epoch=epoch, generation=0, profile=profile,
        last_activity_ns=now_ns,
    )


def _token(state: QualificationState) -> str:
    return canonical_digest({
        "profile_fingerprint": state.profile.fingerprint,
        "epoch": state.epoch, "generation": state.generation,
        "status": state.status,
    })


def record_evidence(state: QualificationState, *, expected_generation: int,
                    counts: Mapping[str, int], maxima_ns: Mapping[str, int],
                    passed_flags: frozenset[str], now_ns: int
                    ) -> QualificationDecision:
    if expected_generation != state.generation:
        return QualificationDecision(False, "stale_generation", state)
    if state.status == "quarantined":
        return QualificationDecision(False, "mode_quarantined", state)
    if now_ns < state.last_activity_ns:
        return QualificationDecision(False, "nonmonotonic_evidence_time", state)
    if (set(counts) != set(maxima_ns)
            or not set(counts).issubset(REQUIRED_COMPONENTS)
            or not set(passed_flags).issubset(REQUIRED_FLAGS)
            or any(not isinstance(value, int) or isinstance(value, bool) or value <= 0
                   for value in counts.values())
            or any(not isinstance(value, int) or isinstance(value, bool) or value < 0
                   for value in maxima_ns.values())):
        return QualificationDecision(False, "invalid_evidence", state)
    merged_counts = dict(state.sample_counts)
    merged_maxima = dict(state.observed_max_ns)
    for component, count in counts.items():
        merged_counts[component] = merged_counts.get(component, 0) + count
        merged_maxima[component] = max(
            merged_maxima.get(component, 0), maxima_ns[component]
        )
    flags = state.passed_flags | passed_flags
    violations = {
        component for component, observed in merged_maxima.items()
        if observed > state.profile.bounds_ns[component]
    }
    if violations:
        updated = dataclasses.replace(
            state, generation=state.generation + 1, status="quarantined",
            sample_counts=merged_counts, observed_max_ns=merged_maxima,
            passed_flags=flags, last_activity_ns=now_ns,
            reason="component_bound_violation:" + ",".join(sorted(violations)),
        )
        return QualificationDecision(False, updated.reason, updated)
    complete = (
        all(merged_counts.get(name, 0) >= state.profile.minimum_samples[name]
            for name in REQUIRED_COMPONENTS)
        and flags == frozenset(REQUIRED_FLAGS)
    )
    updated = dataclasses.replace(
        state, generation=state.generation + 1,
        status="qualified" if complete else "qualifying",
        sample_counts=merged_counts, observed_max_ns=merged_maxima,
        passed_flags=flags, last_activity_ns=now_ns,
        reason="all_evidence_complete" if complete else "evidence_incomplete",
    )
    return QualificationDecision(
        complete, updated.reason, updated, _token(updated) if complete else None
    )


def check_optional_admission(state: QualificationState, *, token: str,
                             profile_fingerprint: str,
                             expected_epoch: int,
                             expected_generation: int,
                             now_ns: int) -> QualificationDecision:
    if state.status != "qualified":
        return QualificationDecision(False, "mode_not_qualified", state)
    if profile_fingerprint != state.profile.fingerprint:
        return QualificationDecision(False, "profile_fingerprint_mismatch", state)
    if expected_epoch != state.epoch:
        return QualificationDecision(False, "stale_lifecycle_epoch", state)
    if expected_generation != state.generation:
        return QualificationDecision(False, "stale_generation", state)
    if token != _token(state):
        return QualificationDecision(False, "qualification_token_mismatch", state)
    if now_ns < state.last_activity_ns:
        return QualificationDecision(False, "nonmonotonic_admission_time", state)
    if now_ns - state.last_activity_ns > state.profile.max_idle_ns:
        stale = dataclasses.replace(
            state, generation=state.generation + 1, status="stale",
            last_activity_ns=now_ns, reason="idle_requalification_required",
        )
        return QualificationDecision(False, stale.reason, stale)
    updated = dataclasses.replace(state, last_activity_ns=now_ns)
    return QualificationDecision(True, "qualified_optional_admission", updated, token)


def invalidate_for_restart(state: QualificationState, *, new_epoch: int,
                           now_ns: int, lifecycle: str,
                           new_profile: QualificationProfile
                           ) -> QualificationDecision:
    if new_epoch <= state.epoch or now_ns < state.last_activity_ns:
        return QualificationDecision(False, "invalid_restart_transition", state)
    if lifecycle != new_profile.lifecycle:
        return QualificationDecision(False, "restart_lifecycle_mismatch", state)
    restarted = start_mode(new_profile, new_epoch, now_ns)
    return QualificationDecision(True, "restart_invalidated_old_token", restarted)


def observe_runtime_bound(state: QualificationState, *, component: str,
                          observed_ns: int, now_ns: int
                          ) -> QualificationDecision:
    if (component not in REQUIRED_COMPONENTS or observed_ns < 0
            or now_ns < state.last_activity_ns):
        return QualificationDecision(False, "invalid_runtime_observation", state)
    if observed_ns <= state.profile.bounds_ns[component]:
        updated = dataclasses.replace(state, last_activity_ns=now_ns)
        return QualificationDecision(True, "runtime_bound_observed", updated,
                                     _token(updated) if updated.status == "qualified" else None)
    quarantined = dataclasses.replace(
        state, generation=state.generation + 1, status="quarantined",
        last_activity_ns=now_ns,
        reason=f"runtime_{component}_bound_violation",
    )
    return QualificationDecision(False, quarantined.reason, quarantined)
