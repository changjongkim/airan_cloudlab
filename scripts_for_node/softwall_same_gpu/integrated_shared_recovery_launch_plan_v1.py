#!/usr/bin/env python3.11
"""Launch-time-revalidated V17 plan for profiler and later physical modes."""

from __future__ import annotations

import dataclasses

from integrated_shared_recovery_holdout_plan_v1 import (
    BRANCHES,
    BRANCH_CANDIDATES,
    BRANCH_SUCCESSES,
)
from integrated_shared_recovery_plan_v1 import (
    LEASE_ID,
    IntegratedScenarioConfig,
)
from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedAILease,
    SharedRecoveryCoordinator,
    exact_certificate,
)


MS = 1_000_000
LAUNCH_CONTROL_BOUND_MS = 5


def obligation(key, config):
    return RecoveryObligation(
        home_id=key[0],
        request_id=key[1],
        release_ns=config.recovery_release_ns,
        deadline_ns=config.recovery_deadline_ns,
        service_ns=config.conventional_bound_ms * MS,
    )


def launch_lease(config, now_ns):
    start_ns = max(config.ai_start_ns, now_ns)
    return SharedAILease(
        lease_id=LEASE_ID,
        owner_home="global",
        start_ns=start_ns,
        finish_ns=(
            start_ns + (LAUNCH_CONTROL_BOUND_MS + config.ai_bound_ms) * MS
        ),
    )


def build_launch_branch(
    branch: str, now_ns: int,
    config: IntegratedScenarioConfig | None = None,
) -> dict:
    """Rebuild the full certificate at the physical AI launch decision time."""
    if branch not in BRANCHES:
        raise ValueError(f"unknown holdout branch {branch}")
    config = config or IntegratedScenarioConfig()
    config.validate()
    if now_ns < config.recovery_release_ns:
        raise ValueError("launch revalidation cannot precede outcome availability")
    candidates = BRANCH_CANDIDATES[branch]
    local = {}
    for home in ("home0", "home1"):
        jobs = tuple(
            obligation(key, config) for key in candidates if key[0] == home
        )
        local[home] = exact_certificate(jobs, config.capacity) is not None

    coordinator = SharedRecoveryCoordinator(config.capacity)
    reservations = []
    for key in candidates:
        before = coordinator.snapshot()
        result = coordinator.reserve_mandatory(
            obligation(key, config),
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
            now_ns=now_ns,
        )
        outcomes.append({
            "key": list(key),
            "accepted": result.accepted,
            "reason": result.reason,
            "generation": result.generation,
        })
        if not result.accepted:
            raise RuntimeError(f"success transition failed at launch: {key}")

    candidate = launch_lease(config, now_ns)
    before_lease = coordinator.snapshot()
    lease = coordinator.replan_and_lease(
        candidate,
        expected_generation=coordinator.generation,
        now_ns=now_ns,
    )
    after_lease = coordinator.snapshot()
    if not lease.accepted and before_lease != after_lease:
        raise RuntimeError("rejected launch-time lease mutated coordinator state")
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
        "ai_expected": lease.accepted,
        "lease_interval": dataclasses.asdict(candidate),
        "launch_control_bound_ms": LAUNCH_CONTROL_BOUND_MS,
        "launch_revalidation_now_ns": now_ns,
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
