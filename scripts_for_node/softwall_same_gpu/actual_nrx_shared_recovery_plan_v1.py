#!/usr/bin/env python3.11
"""Outcome-driven V17.1 plan for an actual TensorRT NeuralRx canary.

The controlled C154/C155 plan names successful requests in advance.  This
module accepts only outcomes observed from the physical NeuralRx path and
builds the same global certificate at the real launch decision time.
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


def build_actual_outcome_plan(
    success_keys,
    now_ns: int,
    config: IntegratedScenarioConfig | None = None,
) -> dict:
    """Reserve the 3+2 offer, apply observed successes, and try one AI lease."""
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

    outcomes = []
    for key in successes:
        decision = coordinator.resolve_success(
            key,
            expected_generation=coordinator.generation,
            now_ns=now_ns,
        )
        outcomes.append({
            "key": list(key),
            "accepted": decision.accepted,
            "reason": decision.reason,
            "generation": decision.generation,
        })
        if not decision.accepted:
            raise RuntimeError(f"actual success transition failed: {key}")

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

