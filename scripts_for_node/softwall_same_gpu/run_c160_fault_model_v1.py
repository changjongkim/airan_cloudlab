#!/usr/bin/env python3.11
"""Enumerate C160 stale/duplicate/fence/ACK state combinations."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c160_fault_state_model_v1 import (
    State,
    apply_success_batch,
    commit_lease,
    commit_radio,
    handle_rpc_timeout,
    physical_start,
    publish_completion_fence,
    retire_lease,
    safety_invariants,
)


KEYS = frozenset({("h0", "r0"), ("h0", "r1"), ("h1", "r0")})


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    initial = State(epoch=7, generation=3, unresolved=KEYS)
    checked = 0
    violations = []
    mutation_on_reject = []

    # Outcome epoch/generation/duplicate transaction product.
    for epoch in (6, 7, 8):
        for generation in (2, 3, 4):
            for preapplied in (False, True):
                state = initial
                transaction = "outcome"
                if preapplied:
                    state = apply_success_batch(
                        state, event_epoch=7, expected_generation=3,
                        success_keys=[("h0", "r0")],
                        transaction_id=transaction,
                    ).state
                before = state
                decision = apply_success_batch(
                    state, event_epoch=epoch, expected_generation=generation,
                    success_keys=[("h0", "r1")], transaction_id=transaction,
                )
                checked += 1
                expected_mutation = (
                    not preapplied and epoch == 7 and generation == 3
                )
                if (decision.state != before) != expected_mutation:
                    mutation_on_reject.append({
                        "kind": "outcome", "epoch": epoch,
                        "generation": generation, "preapplied": preapplied,
                        "reason": decision.reason,
                    })

    # Lease physical-start/fence/marker product.
    for late_start in (False, True):
        for publish_fence in (False, True):
            for marker in (None, "L", "wrong"):
                state = commit_lease(
                    initial, expected_generation=3, lease_id="L",
                    latest_start_ns=100, transaction_id="lease",
                ).state
                state = physical_start(
                    state, lease_id="L", accepted_ns=101 if late_start else 99
                ).state
                if publish_fence and state.lease.stage == "launched":
                    state = publish_completion_fence(state, lease_id="L").state
                decision = handle_rpc_timeout(
                    state, lease_id="L", observed_marker=marker,
                    transaction_id="timeout",
                )
                checked += 1
                if decision.state.lease is None:
                    if not state.lease.fence_confirmed or marker != "L":
                        violations.append({
                            "kind": "fenceless_retire", "late_start": late_start,
                            "publish_fence": publish_fence, "marker": marker,
                        })
                failed = [
                    name for name, passed in safety_invariants(decision.state).items()
                    if not passed
                ]
                if failed:
                    violations.append({"kind": "invariant", "failed": failed})

    # Retire generation and duplicate transaction product.
    state = commit_lease(
        initial, expected_generation=3, lease_id="L", latest_start_ns=100,
        transaction_id="lease",
    ).state
    state = physical_start(state, lease_id="L", accepted_ns=99).state
    state = publish_completion_fence(state, lease_id="L").state
    for generation in (state.generation - 1, state.generation,
                       state.generation + 1):
        before = state
        decision = retire_lease(
            state, expected_generation=generation, lease_id="L",
            transaction_id=f"retire-{generation}",
        )
        checked += 1
        if generation != state.generation and decision.state != before:
            mutation_on_reject.append({
                "kind": "retire", "generation": generation,
                "reason": decision.reason,
            })

    # Radio epoch/generation/duplicate product.
    for epoch in (6, 7, 8):
        for generation in (2, 3, 4):
            first = commit_radio(
                initial, event_epoch=epoch, request_id="r0",
                expected_generation=generation,
            )
            second = commit_radio(
                first.state, event_epoch=epoch, request_id="r0",
                expected_generation=generation,
            )
            checked += 2
            if not first.accepted and first.state != initial:
                mutation_on_reject.append({
                    "kind": "radio", "epoch": epoch,
                    "generation": generation,
                })
            if first.accepted and second.state != first.state:
                violations.append({
                    "kind": "duplicate_radio_commit", "epoch": epoch,
                    "generation": generation,
                })

    sources = [Path(__file__).resolve(), Path(__file__).with_name(
        "c160_fault_state_model_v1.py"
    ), Path(__file__).with_name("test_c160_fault_state_model_v1.py")]
    gates = {
        "bounded_state_product_nonempty": checked > 0,
        "zero_invariant_violation": not violations,
        "zero_rejected_transition_mutation": not mutation_on_reject,
    }
    value = {
        "schema": "softwall-c160-fault-state-model-v1",
        "status": "C160_MODEL_PASS" if all(gates.values()) else "C160_MODEL_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "summary": {
            "state_transition_cases": checked,
            "invariant_violations": len(violations),
            "rejected_transition_mutations": len(mutation_on_reject),
        },
        "violations": violations,
        "mutation_on_reject": mutation_on_reject,
        "scope": (
            "Finite pure-state audit of epoch/generation/idempotency and AI "
            "physical-fence lifecycle. It is not a GPU fault-injection result, "
            "distributed durability proof, or GPU/driver-hang model."
        ),
        "source_sha256": {
            str(path): sha256(path) for path in sources
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
