#!/usr/bin/env python3
"""Prospective four-branch state machine for the C154/C155 holdout."""

from __future__ import annotations

import dataclasses

from integrated_shared_recovery_plan_v1 import (
    LEASE_ID,
    REQUEST_ORDER,
    IntegratedScenarioConfig,
    candidate_lease,
)
from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedRecoveryCoordinator,
    exact_certificate,
)


BRANCHES = ("all_fail", "conditional_open", "all_success", "overload")
ACCEPTED_KEYS = REQUEST_ORDER[:4]
BRANCH_CANDIDATES = {
    "all_fail": REQUEST_ORDER[:4],
    "conditional_open": REQUEST_ORDER,
    "all_success": REQUEST_ORDER[:4],
    "overload": REQUEST_ORDER,
}
BRANCH_SUCCESSES = {
    "all_fail": (),
    "conditional_open": REQUEST_ORDER[:2],
    "all_success": REQUEST_ORDER[:4],
    "overload": (),
}
BRANCH_AI_EXPECTED = {
    "all_fail": False,
    "conditional_open": True,
    "all_success": True,
    "overload": False,
}


def _obligation(key, config):
    return RecoveryObligation(
        home_id=key[0],
        request_id=key[1],
        release_ns=config.recovery_release_ns,
        deadline_ns=config.recovery_deadline_ns,
        service_ns=config.conventional_bound_ms * 1_000_000,
    )


def build_branch(branch: str, config: IntegratedScenarioConfig | None = None):
    if branch not in BRANCHES:
        raise ValueError(f"unknown holdout branch {branch}")
    config = config or IntegratedScenarioConfig()
    config.validate()
    candidates = BRANCH_CANDIDATES[branch]
    local = {}
    for home in ("home0", "home1"):
        jobs = tuple(_obligation(key, config) for key in candidates if key[0] == home)
        local[home] = exact_certificate(jobs, config.capacity) is not None

    coordinator = SharedRecoveryCoordinator(config.capacity)
    reservations = []
    for key in candidates:
        before = coordinator.snapshot()
        result = coordinator.reserve_mandatory(
            _obligation(key, config),
            expected_generation=coordinator.generation,
        )
        after = coordinator.snapshot()
        reservations.append({
            "key": list(key),
            "accepted": result.accepted,
            "reason": result.reason,
            "generation": result.generation,
            "state_unchanged_on_reject": result.accepted or before == after,
        })
    outcomes = []
    for key in BRANCH_SUCCESSES[branch]:
        result = coordinator.resolve_success(
            key,
            expected_generation=coordinator.generation,
            now_ns=config.recovery_release_ns,
        )
        outcomes.append({
            "key": list(key),
            "accepted": result.accepted,
            "reason": result.reason,
            "generation": result.generation,
        })
    before_lease = coordinator.snapshot()
    lease = coordinator.replan_and_lease(
        candidate_lease(config),
        expected_generation=coordinator.generation,
        now_ns=config.recovery_release_ns,
    )
    after_lease = coordinator.snapshot()
    ai_expected = BRANCH_AI_EXPECTED[branch]
    if lease.accepted != ai_expected:
        raise RuntimeError(
            f"branch {branch} lease={lease.accepted} expected={ai_expected}"
        )
    if not lease.accepted and before_lease != after_lease:
        raise RuntimeError("rejected lease mutated branch state")
    return {
        "branch": branch,
        "config": config,
        "coordinator": coordinator,
        "local_certificates": local,
        "reservation_decisions": reservations,
        "success_outcomes": outcomes,
        "lease_decision": {
            "accepted": lease.accepted,
            "reason": lease.reason,
            "generation": lease.generation,
            "state_unchanged_on_reject": lease.accepted or before_lease == after_lease,
            "placements": [dataclasses.asdict(row) for row in lease.placements],
        },
        "ai_expected": ai_expected,
        "physical_recovery_keys": tuple(
            (row["home_id"], row["request_id"])
            for row in sorted(
                after_lease["placements"],
                key=lambda row: (
                    row["start_ns"], row["lane"], row["home_id"], row["request_id"]
                ),
            )
        ),
        "submitted_keys": candidates,
        "success_keys": BRANCH_SUCCESSES[branch],
        "rejected_keys": tuple(
            tuple(row["key"]) for row in reservations if not row["accepted"]
        ),
        "lease_id": LEASE_ID,
    }

