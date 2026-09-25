#!/usr/bin/env python3.11
"""Map a validated production DU timing batch to QSU/QSN/MI/UQ."""

from __future__ import annotations

import dataclasses

from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedAILease,
    exact_certificate,
)
from c162_certified_scheduler_v1 import certified_schedule


MAX_EXACT_DEBTS = 10


@dataclasses.dataclass(frozen=True)
class ProductionPrediction:
    batch_id: str
    state: str
    reason: str
    mandatory_all_fail_safe: bool
    current_mandatory_safe: bool
    ai_safe: bool
    observed_ai_lease_accepted: bool
    decision_matches: bool
    all_fail_finish_ns: int | None
    current_finish_ns: int | None
    ai_and_recovery_finish_ns: int | None
    all_fail_certificate_method: str | None = None
    current_certificate_method: str | None = None
    ai_certificate_method: str | None = None


def _finish(schedule):
    if schedule is None:
        return None
    return max((row.finish_ns for row in schedule), default=None)


def _schedule(jobs, capacity, leases=(), now_ns=0):
    if len(jobs) <= MAX_EXACT_DEBTS:
        return exact_certificate(jobs, capacity, leases, now_ns), "exact"
    return certified_schedule(jobs, capacity, leases, now_ns), "certified"


def classify_batch(document: dict, batch_records: list[dict], decision: dict
                   ) -> ProductionPrediction:
    mode = document["mode"]
    batch_id = decision["batch_id"]
    observed = bool(decision["ai_lease_accepted"])
    if mode["lifecycle"] != "warm" or mode["node_bound_status"] != "qualified":
        return ProductionPrediction(
            batch_id, "UQ", "mode_provenance_unqualified", False, False,
            False, observed, observed is False, None, None, None,
        )
    record_by_id = {row["request_id"]: row for row in batch_records}
    mandatory_ids = tuple(decision["mandatory_request_ids"])
    unresolved_ids = tuple(decision["unresolved_request_ids"])
    capacity = int(mode["recovery_capacity"])
    recovery = int(mode["recovery_bound_ns"])
    nrx = int(mode["nrx_bound_ns"])
    guard = int(mode["guard_ns"])
    uncertainty = int(document["clock"]["calibration"]["max_error_ns"])

    def obligation(request_id):
        row = record_by_id[request_id]
        # An admitted optional NRx can defer conventional recovery only until
        # its qualified NRx bound.  A request for which NRx was not admitted
        # owes the conventional path immediately at release.
        recovery_release = int(row["release_ns"])
        if row["nrx_admitted"]:
            recovery_release += nrx
        return RecoveryObligation(
            home_id=str(row["home_id"]), request_id=request_id,
            release_ns=recovery_release,
            deadline_ns=int(row["expiry_ns"]) - guard - uncertainty,
            service_ns=recovery,
        )

    try:
        all_jobs = tuple(obligation(request_id) for request_id in mandatory_ids)
        current_jobs = tuple(obligation(request_id) for request_id in unresolved_ids)
    except ValueError:
        return ProductionPrediction(
            batch_id, "MI", "recovery_window_nonpositive", False, False,
            False, observed, observed is False, None, None, None,
        )
    all_now = min((job.release_ns for job in all_jobs), default=int(decision["decision_ns"]))
    all_fail, all_method = _schedule(all_jobs, capacity, now_ns=all_now)
    current, current_method = _schedule(
        current_jobs, capacity, now_ns=int(decision["decision_ns"])
    )
    all_safe = all_fail is not None
    current_safe = current is not None
    if not all_safe:
        if len(all_jobs) > MAX_EXACT_DEBTS:
            return ProductionPrediction(
                batch_id, "UQ", "large_all_fail_state_without_certificate",
                False, current_safe, False, observed, observed is False,
                None, _finish(current), None, all_method, current_method, None,
            )
        return ProductionPrediction(
            batch_id, "MI", "initial_all_fail_infeasible", False,
            current_safe, False, observed, observed is False,
            None, _finish(current), None, all_method, current_method, None,
        )
    if not current_safe:
        if len(current_jobs) > MAX_EXACT_DEBTS:
            return ProductionPrediction(
                batch_id, "UQ", "large_current_state_without_certificate",
                True, False, False, observed, observed is False,
                _finish(all_fail), None, None,
                all_method, current_method, None,
            )
        return ProductionPrediction(
            batch_id, "MI", "current_mandatory_infeasible", True, False,
            False, observed, observed is False, _finish(all_fail), None, None,
            all_method, current_method, None,
        )
    context = decision["ai_context_length"]
    if context is None:
        return ProductionPrediction(
            batch_id, "QSN", "no_ai_request", True, True, False,
            observed, observed is False, _finish(all_fail), _finish(current), None,
            all_method, current_method, None,
        )
    bounds = {int(key): int(value) for key, value in mode["ai_class_bounds_ns"].items()}
    ai_bound = bounds[context]
    lease = SharedAILease(
        lease_id=f"c163-{batch_id}", owner_home="global",
        start_ns=int(decision["decision_ns"]),
        finish_ns=(int(decision["decision_ns"])
                   + int(mode["control_bound_ns"]) + ai_bound),
    )
    ai_deadline = int(decision["ai_deadline_ns"])
    scheduled = None
    if lease.finish_ns <= ai_deadline:
        scheduled, ai_method = _schedule(
            current_jobs, capacity, (lease,), now_ns=int(decision["decision_ns"])
        )
    else:
        ai_method = "exact" if len(current_jobs) <= MAX_EXACT_DEBTS else "certified"
    ai_safe = scheduled is not None
    state = "QSU" if ai_safe else "QSN"
    reason = "qualified_ai_and_mandatory_schedule" if ai_safe else "ai_breaks_deadline_or_certificate"
    finish = max(
        [lease.finish_ns] + ([row.finish_ns for row in scheduled] if scheduled else [])
    ) if ai_safe else None
    return ProductionPrediction(
        batch_id, state, reason, True, True, ai_safe, observed,
        observed == ai_safe, _finish(all_fail), _finish(current), finish,
        all_method, current_method, ai_method,
    )


def classify_document(document: dict) -> list[ProductionPrediction]:
    records_by_batch = {}
    for record in document["records"]:
        records_by_batch.setdefault(record["batch_id"], []).append(record)
    return [
        classify_batch(document, records_by_batch[decision["batch_id"]], decision)
        for decision in document["decisions"]
    ]
