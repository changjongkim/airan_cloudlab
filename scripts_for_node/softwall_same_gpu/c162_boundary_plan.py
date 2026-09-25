#!/usr/bin/env python3.11
"""Build one C162 outcome-injection boundary plan."""

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
from shared_recovery_certificate_v1 import SharedRecoveryCoordinator, exact_certificate


def build_boundary_plan(success_keys, now_ns: int,
                        config: IntegratedScenarioConfig, *, offer_ai: bool) -> dict:
    if offer_ai:
        plan = build_actual_outcome_batch_plan(success_keys, now_ns, config)
        plan["outcome_transition"]["kind"] = (
            "atomic_prespecified_failure_injection"
        )
        plan["ai_offered"] = True
        return plan

    successes = tuple(tuple(key) for key in success_keys)
    if len(successes) != len(set(successes)):
        raise ValueError("duplicate success key")
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
    if any(key not in accepted for key in successes):
        raise ValueError("success key was not admitted")
    unresolved = tuple(key for key in accepted if key not in successes)
    coordinator = SharedRecoveryCoordinator(config.capacity)
    for key in unresolved:
        decision = coordinator.reserve_mandatory(
            obligation(key, config), expected_generation=coordinator.generation,
            now_ns=now_ns,
        )
        if not decision.accepted:
            raise RuntimeError(f"mandatory boundary plan infeasible: {key}")
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
        "local_certificates": {
            home: exact_certificate(
                tuple(obligation(key, config) for key in REQUEST_ORDER if key[0] == home),
                config.capacity,
            ) is not None for home in ("home0", "home1")
        },
        "reservation_decisions": reservations,
        "actual_success_outcomes": [{
            "key": list(key), "accepted": True,
            "reason": "injected_outcome_success_in_atomic_batch",
            "generation": coordinator.generation,
        } for key in successes],
        "outcome_transition": {
            "kind": "atomic_prespecified_failure_injection",
            "observed_successes": [list(key) for key in successes],
            "unresolved_obligations": [list(key) for key in unresolved],
            "applied_at_ns": now_ns,
        },
        "lease_decision": {
            "accepted": False, "reason": "no_ai_offered",
            "generation": coordinator.generation,
            "state_unchanged_on_reject": True,
            "placements": snapshot["placements"],
        },
        "ai_expected": False,
        "ai_offered": False,
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
