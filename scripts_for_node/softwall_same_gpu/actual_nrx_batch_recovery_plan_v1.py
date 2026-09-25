#!/usr/bin/env python3.11
"""Atomic batch-outcome V17.1 plan for C159-Q2 late revalidation.

All-fail admission is still evaluated before outcomes. At the common cutoff,
the already-observed success set is removed in one atomic transition so a
nonexistent sequence of partially applied outcomes cannot cause false reject.
"""

from __future__ import annotations

import dataclasses

from integrated_shared_recovery_launch_plan_v1 import (
    LAUNCH_CONTROL_BOUND_MS,
    launch_lease,
)
from integrated_shared_recovery_plan_v1 import (
    LEASE_ID,
    REQUEST_ORDER,
    IntegratedScenarioConfig,
)
from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedRecoveryCoordinator,
    exact_certificate,
)


MS = 1_000_000


def obligation(key, config):
    return RecoveryObligation(
        home_id=key[0],
        request_id=key[1],
        release_ns=config.recovery_release_ns,
        deadline_ns=config.recovery_deadline_ns,
        service_ns=config.conventional_bound_ms * MS,
    )


def build_actual_outcome_batch_plan(
    success_keys,
    now_ns: int,
    config: IntegratedScenarioConfig | None = None,
) -> dict:
    """Reserve all-fail debt, atomically apply outcomes, and try one AI lease."""
    config = config or IntegratedScenarioConfig()
    config.validate()
    if now_ns < config.recovery_release_ns:
        raise ValueError("actual-outcome decision cannot precede the NRx cutoff")
    successes = tuple(tuple(key) for key in success_keys)
    if len(successes) != len(set(successes)):
        raise ValueError("duplicate actual NeuralRx success")

    local = {}
    for home in ("home0", "home1"):
        jobs = tuple(
            obligation(key, config) for key in REQUEST_ORDER if key[0] == home
        )
        local[home] = exact_certificate(jobs, config.capacity) is not None

    coordinator = SharedRecoveryCoordinator(config.capacity)
    reservations = []
    accepted = []
    for key in REQUEST_ORDER:
        before = coordinator.snapshot()
        decision = coordinator.reserve_mandatory(
            obligation(key, config),
            expected_generation=coordinator.generation,
        )
        after = coordinator.snapshot()
        reservations.append({
            "key": list(key),
            "accepted": decision.accepted,
            "reason": decision.reason,
            "generation": decision.generation,
            "state_unchanged_on_reject": decision.accepted or before == after,
        })
        if decision.accepted:
            accepted.append(key)

    unknown = tuple(key for key in successes if key not in accepted)
    if unknown:
        raise ValueError(f"success is not an accepted obligation: {unknown}")

    # The outcomes were all observed before this transaction. Reconstruct the
    # post-outcome coordinator from the unresolved accepted set at the actual
    # decision time instead of exposing artificial one-by-one intermediate
    # states. Initial all-fail admission and fifth-debt rejection above remain
    # authoritative and are reported separately.
    unresolved = tuple(key for key in accepted if key not in successes)
    post_outcome = SharedRecoveryCoordinator(config.capacity)
    for key in unresolved:
        decision = post_outcome.reserve_mandatory(
            obligation(key, config),
            expected_generation=post_outcome.generation,
            now_ns=now_ns,
        )
        if not decision.accepted:
            raise RuntimeError(
                f"post-outcome certificate infeasible at current time: {key}"
            )
    coordinator = post_outcome
    outcomes = [{
        "key": list(key),
        "accepted": True,
        "reason": "success_released_credit_in_atomic_batch",
        "generation": coordinator.generation,
    } for key in successes]

    candidate = launch_lease(config, now_ns)
    before_lease = coordinator.snapshot()
    lease = coordinator.replan_and_lease(
        candidate,
        expected_generation=coordinator.generation,
        now_ns=now_ns,
    )
    after_lease = coordinator.snapshot()
    if not lease.accepted and before_lease != after_lease:
        raise RuntimeError("rejected actual-outcome lease mutated coordinator")

    placements = tuple(sorted(
        (
            (row["home_id"], row["request_id"])
            for row in after_lease["placements"]
        ),
        key=lambda key: next(
            (
                row["start_ns"], row["lane"], row["home_id"], row["request_id"]
            )
            for row in after_lease["placements"]
            if (row["home_id"], row["request_id"]) == key
        ),
    ))
    return {
        "config": config,
        "coordinator": coordinator,
        "local_certificates": local,
        "reservation_decisions": reservations,
        "actual_success_outcomes": outcomes,
        "outcome_transition": {
            "kind": "atomic_observed_success_batch",
            "observed_successes": [list(key) for key in successes],
            "unresolved_obligations": [list(key) for key in unresolved],
            "applied_at_ns": now_ns,
        },
        "lease_decision": {
            "accepted": lease.accepted,
            "reason": lease.reason,
            "generation": lease.generation,
            "state_unchanged_on_reject": lease.accepted or before_lease == after_lease,
            "placements": [dataclasses.asdict(row) for row in lease.placements],
        },
        "ai_expected": lease.accepted,
        "lease_interval": dataclasses.asdict(candidate),
        "launch_control_bound_ms": LAUNCH_CONTROL_BOUND_MS,
        "launch_revalidation_now_ns": now_ns,
        "submitted_keys": REQUEST_ORDER,
        "accepted_keys": tuple(accepted),
        "rejected_keys": tuple(
            tuple(row["key"]) for row in reservations if not row["accepted"]
        ),
        "success_keys": successes,
        "physical_recovery_keys": placements,
        "lease_id": LEASE_ID,
    }
