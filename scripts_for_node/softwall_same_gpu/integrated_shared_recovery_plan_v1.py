#!/usr/bin/env python3
"""Frozen C153 mechanism scenario for integrated shared recovery.

The scenario deliberately separates the certificate decision from CUDA.  The
physical worker imports these exact transitions, so the CPU tests and GPU run
exercise one state machine rather than two hand-transcribed plans.
"""

from __future__ import annotations

import dataclasses
from typing import Iterable

from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedAILease,
    SharedRecoveryCoordinator,
    exact_certificate,
)


MS = 1_000_000


@dataclasses.dataclass(frozen=True)
class IntegratedScenarioConfig:
    period_ms: int = 180
    expiry_ms: int = 155
    nrx_bound_ms: int = 45
    conventional_bound_ms: int = 25
    ai_bound_ms: int = 35
    guard_ms: int = 2
    capacity: int = 1

    @property
    def recovery_release_ns(self) -> int:
        return self.nrx_bound_ms * MS

    @property
    def recovery_deadline_ns(self) -> int:
        return (self.expiry_ms - self.guard_ms) * MS

    @property
    def ai_start_ns(self) -> int:
        return self.nrx_bound_ms * MS

    @property
    def ai_finish_ns(self) -> int:
        return (self.nrx_bound_ms + self.ai_bound_ms) * MS

    def validate(self) -> None:
        values = (
            self.period_ms, self.expiry_ms, self.nrx_bound_ms,
            self.conventional_bound_ms, self.ai_bound_ms, self.guard_ms,
            self.capacity,
        )
        if any(value <= 0 for value in values):
            raise ValueError("scenario values must be positive")
        if self.recovery_deadline_ns <= self.recovery_release_ns:
            raise ValueError("recovery window must be positive")


REQUEST_ORDER = (
    ("home0", "h0-r0"),
    ("home0", "h0-r1"),
    ("home0", "h0-r2"),
    ("home1", "h1-r0"),
    ("home1", "h1-r1"),
)
SUCCESS_KEYS = (("home0", "h0-r0"), ("home0", "h0-r1"))
PHYSICAL_RECOVERY_KEYS = (("home0", "h0-r2"), ("home1", "h1-r0"))
REJECTED_KEY = ("home1", "h1-r1")
LEASE_ID = "c153-qwen-context64"


def obligations(config: IntegratedScenarioConfig) -> tuple[RecoveryObligation, ...]:
    config.validate()
    return tuple(
        RecoveryObligation(
            home_id=home,
            request_id=request,
            release_ns=config.recovery_release_ns,
            deadline_ns=config.recovery_deadline_ns,
            service_ns=config.conventional_bound_ms * MS,
        )
        for home, request in REQUEST_ORDER
    )


def local_certificates(config: IntegratedScenarioConfig) -> dict[str, bool]:
    items = obligations(config)
    homes = sorted({item.home_id for item in items})
    return {
        home: exact_certificate(
            (item for item in items if item.home_id == home),
            config.capacity,
        ) is not None
        for home in homes
    }


def reserve_global(
    config: IntegratedScenarioConfig,
) -> tuple[SharedRecoveryCoordinator, list[dict]]:
    coordinator = SharedRecoveryCoordinator(config.capacity)
    decisions = []
    for item in obligations(config):
        before = coordinator.snapshot()
        result = coordinator.reserve_mandatory(
            item,
            expected_generation=coordinator.generation,
        )
        after = coordinator.snapshot()
        decisions.append({
            "home_id": item.home_id,
            "request_id": item.request_id,
            "accepted": result.accepted,
            "reason": result.reason,
            "generation_before": before["generation"],
            "generation_after": after["generation"],
            "state_unchanged_on_reject": (
                result.accepted or before == after
            ),
        })
    return coordinator, decisions


def candidate_lease(config: IntegratedScenarioConfig) -> SharedAILease:
    return SharedAILease(
        lease_id=LEASE_ID,
        owner_home="global",
        start_ns=config.ai_start_ns,
        finish_ns=config.ai_finish_ns,
    )


def try_lease(
    coordinator: SharedRecoveryCoordinator,
    config: IntegratedScenarioConfig,
    now_ns: int,
):
    return coordinator.replan_and_lease(
        candidate_lease(config),
        expected_generation=coordinator.generation,
        now_ns=now_ns,
    )


def resolve_successes_and_lease(
    coordinator: SharedRecoveryCoordinator,
    config: IntegratedScenarioConfig,
) -> dict:
    outcomes = []
    after_first_probe = None
    for index, key in enumerate(SUCCESS_KEYS):
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
        if index == 0:
            before = coordinator.snapshot()
            probe = try_lease(
                coordinator, config, config.recovery_release_ns
            )
            after = coordinator.snapshot()
            after_first_probe = {
                "accepted": probe.accepted,
                "reason": probe.reason,
                "state_unchanged": before == after,
            }
    lease = try_lease(coordinator, config, config.recovery_release_ns)
    return {
        "success_outcomes": outcomes,
        "lease_after_one_success": after_first_probe,
        "lease_after_two_successes": {
            "accepted": lease.accepted,
            "reason": lease.reason,
            "generation": lease.generation,
            "placements": [dataclasses.asdict(item) for item in lease.placements],
        },
    }


def placement_keys(snapshot: dict) -> tuple[tuple[str, str], ...]:
    return tuple(
        (item["home_id"], item["request_id"])
        for item in sorted(
            snapshot["placements"],
            key=lambda item: (
                item["start_ns"], item["lane"], item["home_id"],
                item["request_id"],
            ),
        )
    )


def expected_physical_keys() -> tuple[tuple[str, str], ...]:
    return PHYSICAL_RECOVERY_KEYS


def all_keys(rows: Iterable[RecoveryObligation]) -> tuple[tuple[str, str], ...]:
    return tuple(item.key for item in rows)

