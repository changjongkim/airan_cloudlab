#!/usr/bin/env python3.11
"""Deterministic finite-state checks for the shared-recovery prototype.

This verifier intentionally checks only the control-plane scheduling model.
It does not execute CUDA, cuPHY, NeuralRx, IPC, or an external AI workload.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedAILease,
    SharedRecoveryCoordinator,
    exact_certificate,
)


MS = 1_000_000
QUANTUM_MS = 25
HORIZON_MS = 100


def obligation(home: str, index: int, release_ms: int = 0,
               deadline_ms: int = HORIZON_MS) -> RecoveryObligation:
    return RecoveryObligation(
        home_id=home,
        request_id=f"r{index}",
        release_ns=release_ms * MS,
        deadline_ns=deadline_ms * MS,
        service_ns=QUANTUM_MS * MS,
    )


def reference_slot_feasible(jobs, blackout_start_ms=None, capacity=1):
    """Independent discrete matching oracle for the 25 ms-grid campaign."""
    slots = tuple(
        (lane, start)
        for lane in range(capacity)
        for start in range(0, HORIZON_MS, QUANTUM_MS)
    )
    if blackout_start_ms is not None:
        slots = tuple(slot for slot in slots
                      if slot[1] != blackout_start_ms)

    ordered = sorted(jobs, key=lambda job: (
        job.deadline_ns, job.release_ns, job.home_id, job.request_id
    ))

    def search(index, free_slots):
        if index == len(ordered):
            return True
        job = ordered[index]
        for slot in free_slots:
            _, start_ms = slot
            start_ns = start_ms * MS
            finish_ns = (start_ms + QUANTUM_MS) * MS
            if start_ns < job.release_ns or finish_ns > job.deadline_ns:
                continue
            if search(index + 1, tuple(item for item in free_slots
                                       if item != slot)):
                return True
        return False

    return search(0, slots)


def model_feasible(jobs, blackout_start_ms=None, capacity=1):
    leases = ()
    if blackout_start_ms is not None:
        leases = (SharedAILease(
            "ai", "external", blackout_start_ms * MS,
            (blackout_start_ms + QUANTUM_MS) * MS,
        ),)
    return exact_certificate(jobs, capacity=capacity, leases=leases) is not None


def equal_deadline_grid():
    rows = []
    mismatch_count = 0
    locally_safe_globally_unsafe = 0
    conditional_open_count = 0
    for home0_count in range(5):
        for home1_count in range(5):
            jobs = (
                [obligation("h0", index) for index in range(home0_count)]
                + [obligation("h1", index) for index in range(home1_count)]
            )
            local0 = model_feasible(
                [job for job in jobs if job.home_id == "h0"]
            )
            local1 = model_feasible(
                [job for job in jobs if job.home_id == "h1"]
            )
            global_safe = model_feasible(jobs)
            analytic_safe = len(jobs) * QUANTUM_MS <= HORIZON_MS
            mismatch_count += global_safe != analytic_safe
            false_safe = local0 and local1 and not global_safe
            locally_safe_globally_unsafe += false_safe

            ai_before = model_feasible(jobs, blackout_start_ms=0)
            ai_after_one_success = None
            if jobs:
                ai_after_one_success = model_feasible(
                    jobs[1:], blackout_start_ms=0
                )
            conditional_open = (
                global_safe and not ai_before and ai_after_one_success is True
            )
            conditional_open_count += conditional_open
            rows.append({
                "home0_obligations": home0_count,
                "home1_obligations": home1_count,
                "local0_safe": local0,
                "local1_safe": local1,
                "global_safe": global_safe,
                "analytic_global_safe": analytic_safe,
                "local_safe_global_unsafe": false_safe,
                "ai_25ms_before_success": ai_before,
                "ai_25ms_after_one_success": ai_after_one_success,
                "conditional_ai_opened": conditional_open,
            })
    return {
        "state_count": len(rows),
        "analytic_mismatch_count": mismatch_count,
        "local_safe_global_unsafe_count": locally_safe_globally_unsafe,
        "conditional_ai_open_count": conditional_open_count,
        "rows": rows,
    }


def variable_window_grid():
    timing_options = ((0, 75), (0, 100), (25, 75), (25, 100))
    blackout_options = (None, 0, 25, 50, 75)
    state_count = 0
    mismatch_count = 0
    examples = []
    for capacity in (1, 2):
        for count in range(1, 5):
            for timings in itertools.product(timing_options, repeat=count):
                jobs = [
                    obligation(f"h{index % 2}", index, release, deadline)
                    for index, (release, deadline) in enumerate(timings)
                ]
                for blackout in blackout_options:
                    expected = reference_slot_feasible(
                        jobs, blackout, capacity
                    )
                    actual = model_feasible(jobs, blackout, capacity)
                    state_count += 1
                    if expected != actual:
                        mismatch_count += 1
                        if len(examples) < 10:
                            examples.append({
                                "capacity": capacity,
                                "timings_ms": timings,
                                "blackout_start_ms": blackout,
                                "reference": expected,
                                "model": actual,
                            })
    return {
        "state_count": state_count,
        "reference_mismatch_count": mismatch_count,
        "mismatch_examples": examples,
    }


def transaction_checks():
    coordinator = SharedRecoveryCoordinator(capacity=1)
    for home in ("h0", "h1"):
        for index in range(2):
            result = coordinator.reserve_mandatory(obligation(home, index))
            if not result.accepted:
                raise AssertionError(result)

    full_snapshot = coordinator.snapshot()
    rejected = coordinator.reserve_mandatory(obligation("h2", 0))
    rejected_update_atomic = (
        not rejected.accepted
        and rejected.reason == "global_all_fail_infeasible"
        and coordinator.snapshot() == full_snapshot
    )

    blocked = coordinator.replan_and_lease(
        SharedAILease("ai0", "h0", 0, QUANTUM_MS * MS),
        coordinator.generation,
    )
    blocked_state_unchanged = (
        not blocked.accepted
        and blocked.reason == "lease_breaks_global_certificate"
        and coordinator.snapshot() == full_snapshot
    )

    released = coordinator.resolve_success(
        ("h1", "r0"), coordinator.generation
    )
    leased = coordinator.replan_and_lease(
        SharedAILease("ai0", "h0", 0, QUANTUM_MS * MS),
        released.generation,
    )
    before_no_fence = coordinator.snapshot()
    no_fence = coordinator.retire_lease(
        "ai0", False, leased.generation
    )
    fence_required_and_atomic = (
        not no_fence.accepted
        and no_fence.reason == "physical_fence_required"
        and coordinator.snapshot() == before_no_fence
    )

    stale = coordinator.resolve_success(
        ("h0", "r0"), expected_generation=full_snapshot["generation"]
    )
    stale_generation_rejected = (
        not stale.accepted and stale.reason == "stale_generation"
    )
    return {
        "rejected_update_atomic": rejected_update_atomic,
        "blocked_lease_atomic": blocked_state_unchanged,
        "success_opens_ai_lease": released.accepted and leased.accepted,
        "physical_fence_required": fence_required_and_atomic,
        "stale_generation_rejected": stale_generation_rejected,
        "final_generation": coordinator.generation,
    }


def build_result():
    equal = equal_deadline_grid()
    variable = variable_window_grid()
    transactions = transaction_checks()
    gates = {
        "equal_grid_matches_analytic_capacity": (
            equal["analytic_mismatch_count"] == 0
        ),
        "nonseparable_counterexamples_exist": (
            equal["local_safe_global_unsafe_count"] > 0
        ),
        "conditional_ai_windows_exist": (
            equal["conditional_ai_open_count"] > 0
        ),
        "variable_grid_matches_independent_oracle": (
            variable["reference_mismatch_count"] == 0
        ),
        **transactions,
    }
    # final_generation is evidence rather than a boolean gate.
    final_generation = gates.pop("final_generation")
    return {
        "schema": "softwall-shared-recovery-model-v1",
        "status": "MODEL_PASS_PHYSICAL_UQ" if all(gates.values()) else "MODEL_FAIL",
        "claim_scope": (
            "finite-state control-plane validation only; no CUDA, cuPHY, "
            "NeuralRx, IPC, WCET, or physical multi-home claim"
        ),
        "configuration": {
            "homes": 2,
            "recovery_lanes": 1,
            "service_ms": QUANTUM_MS,
            "horizon_ms": HORIZON_MS,
            "ai_blackout_ms": QUANTUM_MS,
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "transaction_final_generation": final_generation,
        "equal_deadline_grid": equal,
        "variable_window_grid": variable,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_result()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "status": result["status"],
        "all_pass": result["all_pass"],
        "equal_states": result["equal_deadline_grid"]["state_count"],
        "variable_states": result["variable_window_grid"]["state_count"],
        "local_safe_global_unsafe": result["equal_deadline_grid"][
            "local_safe_global_unsafe_count"
        ],
    }, sort_keys=True))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
