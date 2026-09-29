#!/usr/bin/env python3.11
"""C178 early AI slots: AI that runs during the NeuralRx phase (pure Python, no CUDA).

At release every admitted TB still has a pending recovery, and the recovery
lane stays idle until the NeuralRx deadline. An early AI slot starts at
release and is committed only when the all-fail schedule of every admitted TB,
with the slot as an AI busy interval, stays executable: whatever the NeuralRx
outcomes, every recovery still ends before the recovery deadline.

On one GPU the slot shares the GPU with NeuralRx. A NeuralRx result that the
slot delays past the NeuralRx deadline only turns into a required recovery,
which the all-fail schedule already holds; the cost is NeuralRx value, not
radio safety.
"""

from __future__ import annotations

from actual_nrx_batch_recovery_plan_v1 import obligation
from c176_burst_plan import lease_ns
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig
from shared_recovery_certificate_v1 import SharedAILease, SharedRecoveryCoordinator


def plan_early(now_ns: int, queue: list, config: IntegratedScenarioConfig, max_leases: int,
               prefilter: bool = True) -> dict:
    """Early AI slots against the all-fail schedule of every admitted TB.

    ``now_ns`` is the decision time relative to release, and ``queue`` holds
    dicts with ``request_id``, ``context_length`` and ``deadline_rel_ns``. The
    function does not mutate the queue.

    With one recovery lane, every admitted TB shares one release and one
    recovery deadline, and an early slot starts before that release. The
    verifier then accepts a slot exactly when it ends by the recovery deadline
    minus the service of every admitted TB. ``prefilter`` skips the verifier for
    slots that end later, which keeps planning within the launch budget when a
    burst fills the queue; the verifier still decides every remaining slot.
    """
    coordinator = SharedRecoveryCoordinator(config.capacity)
    accepted = []
    for key in REQUEST_ORDER:
        if coordinator.reserve_mandatory(obligation(key, config),
                                         expected_generation=coordinator.generation).accepted:
            accepted.append(key)
    cursor = max(0, now_ns)
    latest_end = None
    if prefilter and config.capacity == 1:
        latest_end = config.recovery_deadline_ns - len(accepted) * config.conventional_bound_ms * 1_000_000
    leases = []
    for request in queue:
        if len(leases) >= max_leases:
            break
        end = cursor + lease_ns(request["context_length"])
        if end > request["deadline_rel_ns"]:
            continue
        if latest_end is not None and end > latest_end:
            continue
        before = coordinator.snapshot()
        result = coordinator.replan_and_lease(
            SharedAILease(lease_id=request["request_id"], owner_home="global", start_ns=cursor, finish_ns=end),
            expected_generation=coordinator.generation, now_ns=cursor)
        if not result.accepted:
            if coordinator.snapshot() != before:
                raise RuntimeError("rejected early slot mutated the all-fail schedule")
            continue
        leases.append({"request_id": request["request_id"], "context_length": request["context_length"],
                       "start_ns": cursor, "finish_ns": end, "early": True})
        cursor = end
    placements = coordinator.snapshot()["placements"]
    return {
        "coordinator": coordinator,
        "accepted_keys": accepted,
        "leases": leases,
        # End of the last recovery when every TB fails and every slot and
        # recovery takes its bound.
        "all_fail_end_ns": max((row["finish_ns"] for row in placements), default=0),
    }


def mandatory_infeasible_plan(successes, now_ns: int, config) -> dict:
    """Plan of an Infeasible period: no AI slot, every required recovery at once.

    The all-fail schedule of the unresolved TBs no longer fits before the
    recovery deadline at the current time, for example after a host stall.
    Algorithm 1 then admits no AI and runs every required recovery in deadline
    order; the period is recorded as Infeasible rather than as a scheduling
    decision.
    """
    successes = tuple(tuple(key) for key in successes)
    admission = SharedRecoveryCoordinator(config.capacity)
    accepted = [key for key in REQUEST_ORDER
                if admission.reserve_mandatory(obligation(key, config),
                                               expected_generation=admission.generation).accepted]
    unresolved = [key for key in accepted if key not in successes]
    start = max(config.recovery_release_ns, now_ns)
    return {
        "coordinator": admission,
        "accepted_keys": accepted,
        "rejected_keys": [key for key in REQUEST_ORDER if key not in accepted],
        "success_keys": list(successes),
        "unresolved_keys": unresolved,
        "leases": [],
        "bound_end_ns": start + len(unresolved) * config.conventional_bound_ms * 1_000_000,
        "contract_broken": False,
        "mandatory_infeasible": True,
    }
