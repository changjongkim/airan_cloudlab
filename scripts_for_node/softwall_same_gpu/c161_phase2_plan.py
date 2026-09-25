#!/usr/bin/env python3.11
"""C161 phase-2 plan wrapper with persistent AI quarantine."""

from __future__ import annotations

from actual_nrx_batch_recovery_plan_v1 import (
    build_actual_outcome_batch_plan,
    obligation,
)
from integrated_shared_recovery_launch_plan_v1 import LAUNCH_CONTROL_BOUND_MS
from integrated_shared_recovery_plan_v1 import (
    LEASE_ID,
    REQUEST_ORDER,
    IntegratedScenarioConfig,
)
from shared_recovery_certificate_v1 import (
    SharedRecoveryCoordinator,
    exact_certificate,
)


def build_phase2_plan(success_keys, now_ns: int,
                      config: IntegratedScenarioConfig, *, ai_enabled: bool) -> dict:
    """Build the normal atomic plan or a certificate-only quarantined plan."""
    if ai_enabled:
        plan = build_actual_outcome_batch_plan(success_keys, now_ns, config)
        plan["ai_quarantined"] = False
        return plan
    # Quarantine is an admission state, not a large artificial job. Build the
    # same all-fail and common-cutoff recovery certificate without constructing
    # any AI lease candidate. This remains correct when every debt succeeds;
    # using a large sentinel lease was false because an empty recovery set can
    # admit an arbitrarily long lease.
    successes = tuple(tuple(key) for key in success_keys)
    if len(successes) != len(set(successes)):
        raise ValueError("duplicate actual NeuralRx success")
    local = {}
    for home in ("home0", "home1"):
        jobs = tuple(obligation(key, config) for key in REQUEST_ORDER if key[0] == home)
        local[home] = exact_certificate(jobs, config.capacity) is not None
    all_fail = SharedRecoveryCoordinator(config.capacity)
    reservations = []
    accepted = []
    for key in REQUEST_ORDER:
        before = all_fail.snapshot()
        decision = all_fail.reserve_mandatory(
            obligation(key, config), expected_generation=all_fail.generation
        )
        after = all_fail.snapshot()
        reservations.append({
            "key": list(key), "accepted": decision.accepted,
            "reason": decision.reason, "generation": decision.generation,
            "state_unchanged_on_reject": decision.accepted or before == after,
        })
        if decision.accepted:
            accepted.append(key)
    unknown = tuple(key for key in successes if key not in accepted)
    if unknown:
        raise ValueError(f"success is not an accepted obligation: {unknown}")
    unresolved = tuple(key for key in accepted if key not in successes)
    coordinator = SharedRecoveryCoordinator(config.capacity)
    for key in unresolved:
        decision = coordinator.reserve_mandatory(
            obligation(key, config),
            expected_generation=coordinator.generation,
            now_ns=now_ns,
        )
        if not decision.accepted:
            raise RuntimeError(f"quarantined recovery certificate infeasible: {key}")
    snapshot = coordinator.snapshot()
    placements = tuple(
        (row["home_id"], row["request_id"])
        for row in sorted(snapshot["placements"], key=lambda row: (
            row["start_ns"], row["lane"], row["home_id"], row["request_id"]
        ))
    )
    return {
        "config": config,
        "coordinator": coordinator,
        "local_certificates": local,
        "reservation_decisions": reservations,
        "actual_success_outcomes": [{
            "key": list(key), "accepted": True,
            "reason": "success_released_credit_in_atomic_batch",
            "generation": coordinator.generation,
        } for key in successes],
        "outcome_transition": {
            "kind": "atomic_observed_success_batch",
            "observed_successes": [list(key) for key in successes],
            "unresolved_obligations": [list(key) for key in unresolved],
            "applied_at_ns": now_ns,
        },
        "lease_decision": {
            "accepted": False,
            "reason": "ai_quarantined_before_admission",
            "generation": coordinator.generation,
            "state_unchanged_on_reject": True,
            "placements": snapshot["placements"],
        },
        "ai_expected": False,
        "ai_quarantined": True,
        "lease_interval": None,
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
