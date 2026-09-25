#!/usr/bin/env python3
"""Exact small-state certificate for a GPU recovery pool shared by RAN homes.

The existing physical SoftWall modes keep one recovery calendar per home. This
module is a control-plane prototype for the non-separable case: mandatory
recoveries from several homes and external-AI leases contend for one shared
GPU calendar. It deliberately has no CUDA dependency and makes no physical or
WCET claim.
"""

from __future__ import annotations

import dataclasses
import threading
from typing import Dict, Iterable, Optional, Tuple


Key = Tuple[str, str]


@dataclasses.dataclass(frozen=True)
class RecoveryObligation:
    home_id: str
    request_id: str
    release_ns: int
    deadline_ns: int
    service_ns: int

    def __post_init__(self):
        if (not self.home_id or not self.request_id or self.release_ns < 0
                or self.deadline_ns <= self.release_ns or self.service_ns <= 0):
            raise ValueError("invalid recovery obligation")

    @property
    def key(self) -> Key:
        return self.home_id, self.request_id


@dataclasses.dataclass(frozen=True)
class RecoveryPlacement:
    home_id: str
    request_id: str
    lane: int
    start_ns: int
    finish_ns: int

    @property
    def key(self) -> Key:
        return self.home_id, self.request_id


@dataclasses.dataclass(frozen=True)
class SharedAILease:
    lease_id: str
    owner_home: str
    start_ns: int
    finish_ns: int

    def __post_init__(self):
        if (not self.lease_id or not self.owner_home or self.start_ns < 0
                or self.finish_ns <= self.start_ns):
            raise ValueError("invalid shared AI lease")


@dataclasses.dataclass(frozen=True)
class CertificateResult:
    accepted: bool
    reason: str
    generation: int
    placements: Tuple[RecoveryPlacement, ...]


def _advance_over_blackouts(start_ns, service_ns, leases):
    """Return the earliest non-overlapping start for a full-GPU AI blackout."""
    start = start_ns
    while True:
        conflict = next((lease for lease in leases
                         if start < lease.finish_ns
                         and lease.start_ns < start + service_ns), None)
        if conflict is None:
            return start
        start = conflict.finish_ns


def exact_certificate(obligations: Iterable[RecoveryObligation], capacity: int,
                      leases: Iterable[SharedAILease] = (), now_ns: int = 0
                      ) -> Optional[Tuple[RecoveryPlacement, ...]]:
    """Find a non-preemptive all-fail schedule by bounded exhaustive search.

    Recovery lanes are parallel. A shared AI lease conservatively blacks out
    the full GPU, so it overlaps no recovery lane. For a fixed job/lane order,
    earliest placement is dominant; enumerating every next job and lane is
    therefore exact for the supplied finite set.
    """
    jobs = tuple(sorted(obligations, key=lambda item: (
        item.deadline_ns, item.release_ns, item.home_id, item.request_id
    )))
    blackouts = tuple(sorted(leases, key=lambda item: (
        item.start_ns, item.finish_ns, item.lease_id
    )))
    if capacity <= 0 or now_ns < 0:
        raise ValueError("invalid certificate configuration")
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("duplicate recovery obligation")
    if len({lease.lease_id for lease in blackouts}) != len(blackouts):
        raise ValueError("duplicate AI lease")
    if any(left.finish_ns > right.start_ns
           for left, right in zip(blackouts, blackouts[1:])):
        return None
    lane_available = tuple(now_ns for _ in range(capacity))

    def search(remaining, available, placements):
        if not remaining:
            return tuple(sorted(placements, key=lambda item: item.key))
        for index, job in enumerate(remaining):
            rest = remaining[:index] + remaining[index + 1:]
            for lane in range(capacity):
                start = max(now_ns, job.release_ns, available[lane])
                start = _advance_over_blackouts(start, job.service_ns, blackouts)
                finish = start + job.service_ns
                if finish > job.deadline_ns:
                    continue
                next_available = list(available)
                next_available[lane] = finish
                result = search(
                    rest, tuple(next_available), placements + (
                        RecoveryPlacement(
                            job.home_id, job.request_id, lane, start, finish
                        ),
                    )
                )
                if result is not None:
                    return result
        return None

    return search(jobs, lane_available, ())


class SharedRecoveryCoordinator:
    """Generation-safe atomic coordinator for multiple logical RAN homes."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.generation = 0
        self._obligations: Dict[Key, RecoveryObligation] = {}
        self._leases: Dict[str, SharedAILease] = {}
        self._placements: Tuple[RecoveryPlacement, ...] = ()
        self._lock = threading.Lock()

    def _result(self, accepted, reason):
        return CertificateResult(
            accepted, reason, self.generation, self._placements
        )

    def _generation_ok(self, expected_generation):
        return expected_generation is None or expected_generation == self.generation

    def reserve_mandatory(self, obligation: RecoveryObligation,
                          expected_generation: Optional[int] = None,
                          now_ns: int = 0) -> CertificateResult:
        with self._lock:
            if not self._generation_ok(expected_generation):
                return self._result(False, "stale_generation")
            if obligation.key in self._obligations:
                return self._result(False, "duplicate_obligation")
            candidate = dict(self._obligations)
            candidate[obligation.key] = obligation
            schedule = exact_certificate(
                candidate.values(), self.capacity, self._leases.values(), now_ns
            )
            if schedule is None:
                return self._result(False, "global_all_fail_infeasible")
            self._obligations = candidate
            self._placements = schedule
            self.generation += 1
            return self._result(True, "reserved")

    def resolve_success(self, key: Key, expected_generation: Optional[int] = None,
                        now_ns: int = 0) -> CertificateResult:
        with self._lock:
            if not self._generation_ok(expected_generation):
                return self._result(False, "stale_generation")
            if key not in self._obligations:
                return self._result(False, "unknown_obligation")
            candidate = dict(self._obligations)
            del candidate[key]
            schedule = exact_certificate(
                candidate.values(), self.capacity, self._leases.values(), now_ns
            )
            if schedule is None:
                return self._result(False, "remaining_certificate_infeasible")
            self._obligations = candidate
            self._placements = schedule
            self.generation += 1
            return self._result(True, "success_released_credit")

    def replan_and_lease(self, lease: SharedAILease, expected_generation: int,
                         now_ns: int = 0) -> CertificateResult:
        """Publish the global replan and AI lease as one versioned update."""
        with self._lock:
            if not self._generation_ok(expected_generation):
                return self._result(False, "stale_generation")
            if lease.lease_id in self._leases:
                return self._result(False, "duplicate_lease")
            candidate_leases = dict(self._leases)
            candidate_leases[lease.lease_id] = lease
            schedule = exact_certificate(
                self._obligations.values(), self.capacity,
                candidate_leases.values(), now_ns,
            )
            if schedule is None:
                return self._result(False, "lease_breaks_global_certificate")
            self._leases = candidate_leases
            self._placements = schedule
            self.generation += 1
            return self._result(True, "lease_committed")

    def retire_lease(self, lease_id: str, gpu_fence_confirmed: bool,
                     expected_generation: Optional[int] = None,
                     now_ns: int = 0) -> CertificateResult:
        with self._lock:
            if not self._generation_ok(expected_generation):
                return self._result(False, "stale_generation")
            if not gpu_fence_confirmed:
                return self._result(False, "physical_fence_required")
            if lease_id not in self._leases:
                return self._result(False, "unknown_lease")
            candidate = dict(self._leases)
            del candidate[lease_id]
            schedule = exact_certificate(
                self._obligations.values(), self.capacity,
                candidate.values(), now_ns,
            )
            if schedule is None:
                return self._result(False, "remaining_certificate_infeasible")
            self._leases = candidate
            self._placements = schedule
            self.generation += 1
            return self._result(True, "lease_retired")

    def snapshot(self):
        with self._lock:
            return {
                "schema": "softwall-shared-recovery-certificate-v1",
                "generation": self.generation,
                "capacity": self.capacity,
                "obligations": [dataclasses.asdict(item) for item in sorted(
                    self._obligations.values(), key=lambda item: item.key
                )],
                "leases": [dataclasses.asdict(item) for item in sorted(
                    self._leases.values(), key=lambda item: item.lease_id
                )],
                "placements": [dataclasses.asdict(item)
                               for item in self._placements],
            }
