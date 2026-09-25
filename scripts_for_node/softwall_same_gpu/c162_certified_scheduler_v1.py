#!/usr/bin/env python3.11
"""Polynomial candidate scheduler with an independent executable verifier."""

from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable, Iterable

from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    RecoveryPlacement,
    SharedAILease,
    _advance_over_blackouts,
)


Order = Callable[[RecoveryObligation], tuple]


def _orders(now_ns: int) -> tuple[Order, ...]:
    return (
        lambda job: (job.deadline_ns, job.release_ns, job.service_ns, job.key),
        lambda job: (
            job.deadline_ns - max(now_ns, job.release_ns) - job.service_ns,
            job.deadline_ns, job.release_ns, job.key,
        ),
        lambda job: (job.release_ns, job.deadline_ns, job.service_ns, job.key),
        lambda job: (job.service_ns, job.deadline_ns, job.release_ns, job.key),
        lambda job: (-job.service_ns, job.deadline_ns, job.release_ns, job.key),
    )


def verify_certificate(
    obligations: Iterable[RecoveryObligation],
    placements: Iterable[RecoveryPlacement],
    capacity: int,
    leases: Iterable[SharedAILease] = (),
    now_ns: int = 0,
) -> tuple[bool, str]:
    jobs = tuple(obligations)
    rows = tuple(placements)
    blackouts = tuple(sorted(leases, key=lambda lease: (
        lease.start_ns, lease.finish_ns, lease.lease_id
    )))
    if capacity <= 0 or now_ns < 0:
        return False, "invalid_configuration"
    if len({job.key for job in jobs}) != len(jobs):
        return False, "duplicate_obligation"
    if len({row.key for row in rows}) != len(rows):
        return False, "duplicate_placement"
    if {job.key for job in jobs} != {row.key for row in rows}:
        return False, "placement_key_mismatch"
    if len({lease.lease_id for lease in blackouts}) != len(blackouts):
        return False, "duplicate_lease"
    if any(left.finish_ns > right.start_ns
           for left, right in zip(blackouts, blackouts[1:])):
        return False, "overlapping_ai_leases"
    job_by_key = {job.key: job for job in jobs}
    for row in rows:
        job = job_by_key[row.key]
        if not 0 <= row.lane < capacity:
            return False, "invalid_lane"
        if row.start_ns < max(now_ns, job.release_ns):
            return False, "release_or_now_violation"
        if row.finish_ns != row.start_ns + job.service_ns:
            return False, "service_mismatch"
        if row.finish_ns > job.deadline_ns:
            return False, "deadline_violation"
        if any(row.start_ns < lease.finish_ns
               and lease.start_ns < row.finish_ns for lease in blackouts):
            return False, "ai_overlap"
    for lane in range(capacity):
        lane_rows = sorted(
            (row for row in rows if row.lane == lane),
            key=lambda row: (row.start_ns, row.finish_ns, row.key),
        )
        if any(left.finish_ns > right.start_ns
               for left, right in zip(lane_rows, lane_rows[1:])):
            return False, "lane_overlap"
    return True, "verified"


def _list_schedule(jobs, capacity, blackouts, now_ns, order):
    available = [now_ns] * capacity
    rows = []
    for job in sorted(jobs, key=order):
        candidates = []
        for lane in range(capacity):
            start = max(now_ns, job.release_ns, available[lane])
            start = _advance_over_blackouts(start, job.service_ns, blackouts)
            candidates.append((start + job.service_ns, start, lane))
        finish, start, lane = min(candidates)
        if finish > job.deadline_ns:
            return None
        rows.append(RecoveryPlacement(
            job.home_id, job.request_id, lane, start, finish
        ))
        available[lane] = finish
    return tuple(sorted(rows, key=lambda row: row.key))


def certified_schedule(
    obligations: Iterable[RecoveryObligation],
    capacity: int,
    leases: Iterable[SharedAILease] = (),
    now_ns: int = 0,
) -> tuple[RecoveryPlacement, ...] | None:
    jobs = tuple(obligations)
    blackouts = tuple(sorted(leases, key=lambda lease: (
        lease.start_ns, lease.finish_ns, lease.lease_id
    )))
    if capacity <= 0 or now_ns < 0:
        raise ValueError("invalid scheduler configuration")
    if len({job.key for job in jobs}) != len(jobs):
        raise ValueError("duplicate recovery obligation")
    for order in _orders(now_ns):
        rows = _list_schedule(jobs, capacity, blackouts, now_ns, order)
        if rows is None:
            continue
        valid, _ = verify_certificate(jobs, rows, capacity, blackouts, now_ns)
        if valid:
            return rows
    return None


def certificate_digest(placements: Iterable[RecoveryPlacement]) -> str:
    payload = "\n".join(
        f"{row.home_id}\t{row.request_id}\t{row.lane}\t{row.start_ns}\t{row.finish_ns}"
        for row in sorted(placements, key=lambda item: item.key)
    ).encode()
    return hashlib.sha256(payload).hexdigest()

