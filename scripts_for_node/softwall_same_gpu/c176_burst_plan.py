#!/usr/bin/env python3.11
"""C176 admission policies for bursty AI arrivals (pure Python, no CUDA).

Shared by the physical C176 coordinator and its CPU tests. See
c176_burst_coordinator.py for the policy definitions.
"""

from __future__ import annotations

from actual_nrx_batch_recovery_plan_v1 import obligation
from c159_q2_classes import CLASS_BOUNDS_MS
from integrated_shared_recovery_launch_plan_v1 import LAUNCH_CONTROL_BOUND_MS
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig
from shared_recovery_certificate_v1 import SharedAILease, SharedRecoveryCoordinator


MS = 1_000_000
POLICIES = ("backstop", "recovery_first", "idle_time")


def lease_ns(context_length: int) -> int:
    return (LAUNCH_CONTROL_BOUND_MS + CLASS_BOUNDS_MS[context_length]) * MS


def plan_period(policy: str, successes, now_ns: int, queue: list, config: IntegratedScenarioConfig,
                max_leases: int) -> dict:
    """All-fail admission, one atomic outcome batch, and the policy's leases.

    ``queue`` holds dicts with ``request_id``, ``context_length`` and
    ``deadline_rel_ns`` (the request deadline relative to this release). The
    function does not mutate the queue.
    """
    successes = tuple(tuple(key) for key in successes)
    admission = SharedRecoveryCoordinator(config.capacity)
    accepted = []
    for key in REQUEST_ORDER:
        if admission.reserve_mandatory(obligation(key, config),
                                       expected_generation=admission.generation).accepted:
            accepted.append(key)
    rejected = [key for key in REQUEST_ORDER if key not in accepted]
    unknown = [key for key in successes if key not in accepted]
    if unknown:
        raise ValueError(f"success is not an accepted obligation: {unknown}")
    unresolved = [key for key in accepted if key not in successes]
    coordinator = SharedRecoveryCoordinator(config.capacity)
    for key in unresolved:
        decision = coordinator.reserve_mandatory(obligation(key, config),
                                                 expected_generation=coordinator.generation, now_ns=now_ns)
        if not decision.accepted:
            raise RuntimeError(f"post-outcome certificate infeasible at current time: {key}")

    guard_ns = config.recovery_deadline_ns
    recovery_ns = config.conventional_bound_ms * MS
    start = max(config.ai_start_ns, now_ns)
    cursor = start + (len(unresolved) * recovery_ns if policy == "recovery_first" else 0)
    leases = []
    for request in queue:
        if len(leases) >= max_leases:
            break
        end = cursor + lease_ns(request["context_length"])
        if end > guard_ns or end > request["deadline_rel_ns"]:
            continue
        if policy == "backstop":
            before = coordinator.snapshot()
            result = coordinator.replan_and_lease(
                SharedAILease(lease_id=request["request_id"], owner_home="global", start_ns=cursor, finish_ns=end),
                expected_generation=coordinator.generation, now_ns=now_ns)
            if not result.accepted:
                if coordinator.snapshot() != before:
                    raise RuntimeError("rejected lease mutated the certificate")
                continue
        leases.append({"request_id": request["request_id"], "context_length": request["context_length"],
                       "start_ns": cursor, "finish_ns": end})
        cursor = end
    last_lease_end = leases[-1]["finish_ns"] if leases else start
    recoveries_ns = len(unresolved) * recovery_ns
    if policy == "recovery_first":
        bound_end = max(start + recoveries_ns, last_lease_end)
    else:
        bound_end = last_lease_end + recoveries_ns
    return {
        "coordinator": coordinator,
        "accepted_keys": accepted,
        "rejected_keys": rejected,
        "success_keys": list(successes),
        "unresolved_keys": unresolved,
        "leases": leases,
        # End of the last action if every lease and recovery took its bound.
        "bound_end_ns": bound_end,
        "contract_broken": policy == "idle_time" and bool(leases)
        and last_lease_end + len(unresolved) * recovery_ns > guard_ns,
    }
