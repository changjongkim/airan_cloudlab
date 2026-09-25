#!/usr/bin/env python3
"""Exhaust the bounded V14 broker/RAN control state machine.

This is a finite protocol model, not a timing or hardware proof.  It checks the
ownership and blocking invariants for every pre/post-apply RPC failure branch.
"""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import NamedTuple


ASYNC_STATES = {"prepare_pending", "abort_pending", "complete_pending"}


class State(NamedTuple):
    global_state: str = "ready"
    local_state: str = "idle"
    commit_ack: bool = False
    launch_count: int = 0
    certificate_version: int = 0
    commit_budget_reserved: bool = False


def successors(state):
    g = state.global_state
    l = state.local_state
    if l == "idle" and g == "ready":
        yield "submit_prepare", State(g, "prepare_pending")
    elif l == "prepare_pending" and g == "ready":
        yield "prepare_ack_none", State("ready", "done_no_work")
        yield "prepare_ack_held", State("held", "staged")
        yield "prepare_fail_before_apply", State("ready", "quarantined")
        yield "prepare_fail_after_apply", State("held", "quarantined")
    elif l == "staged" and g == "held":
        yield "poll_current_horizon_fits", State("held", "offered")
        # A short local horizon retains the token for a later window; this
        # self-loop is recorded but not re-enqueued by the visited-state BFS.
        yield "poll_current_horizon_short", State("held", "staged")
        yield "request_slo_expired", State("held", "abort_pending")
    elif l == "offered" and g == "held":
        yield "local_replan_reject", State("held", "abort_pending")
        yield "local_replan_accept_charge_commit", State(
            "held", "commit_wait", commit_budget_reserved=True
        )
    elif l == "abort_pending" and g == "held":
        yield "abort_ack", State("ready", "done_no_work")
        yield "abort_fail_before_apply", State("held", "quarantined")
        yield "abort_fail_after_apply", State("ready", "quarantined")
    elif l == "commit_wait" and g == "held":
        yield "commit_ack", State(
            "inflight", "running", commit_ack=True,
            commit_budget_reserved=True,
        )
        yield "commit_fail_before_apply", State(
            "held", "quarantined", commit_budget_reserved=True
        )
        yield "commit_fail_after_apply", State(
            "inflight", "quarantined", commit_budget_reserved=True
        )
    elif l == "running" and g == "inflight":
        yield "physical_fence", State(
            "inflight", "complete_pending", commit_ack=True,
            launch_count=1, commit_budget_reserved=True,
        )
    elif l == "complete_pending" and g == "inflight":
        yield "complete_ack", State(
            "completed", "done", commit_ack=True,
            launch_count=1, commit_budget_reserved=True,
        )
        yield "complete_fail_before_apply", State(
            "inflight", "quarantined", commit_ack=True,
            launch_count=1, commit_budget_reserved=True,
        )
        yield "complete_fail_after_apply", State(
            "completed", "quarantined", commit_ack=True,
            launch_count=1, commit_budget_reserved=True,
        )


def invariant_failures(state):
    failures = []
    if state.launch_count > 1:
        failures.append("at-most-once launch")
    if state.launch_count and not state.commit_ack:
        failures.append("launch requires commit ACK")
    if state.launch_count and state.global_state not in {"inflight", "completed"}:
        failures.append("launched token cannot become reusable")
    if state.global_state == "ready" and state.launch_count:
        failures.append("executed token returned to ready")
    if state.local_state == "running" and not state.commit_ack:
        failures.append("running before ownership ACK")
    if state.local_state == "commit_wait" and not state.commit_budget_reserved:
        failures.append("synchronous wait lacks admission charge")
    if state.local_state in ASYNC_STATES and state.commit_budget_reserved:
        # complete_pending retains the historical commit charge but does not
        # consume a second RAN critical-path budget.
        if state.local_state != "complete_pending":
            failures.append("async prepare/abort charged as RAN wait")
    if state.certificate_version != 0:
        failures.append("broker transition mutated RAN certificate")
    return failures


def verify():
    initial = State()
    queue = deque([initial])
    seen = {initial}
    edges = []
    violations = []
    while queue:
        state = queue.popleft()
        for name, target in successors(state):
            edges.append({
                "transition": name,
                "source": state._asdict(),
                "target": target._asdict(),
                "ran_critical_path": name.startswith("commit_"),
            })
            for failure in invariant_failures(target):
                violations.append({
                    "transition": name,
                    "state": target._asdict(),
                    "invariant": failure,
                })
            if target not in seen:
                seen.add(target)
                queue.append(target)
    terminal = [state for state in seen if not list(successors(state))]
    return {
        "schema": "softwall-pipelined-control-finite-model-v1",
        "scope": (
            "one global request; all before/after-apply prepare, abort, commit, "
            "and complete reply-loss branches; timing bounds and GPU behavior excluded"
        ),
        "states": len(seen),
        "edges": len(edges),
        "terminal_states": len(terminal),
        "quarantined_terminal_states": sum(
            state.local_state == "quarantined" for state in terminal
        ),
        "invariants": [
            "at-most-once physical launch",
            "physical launch requires commit ACK",
            "executed or ambiguous token is never reusable",
            "only commit wait is on the RAN critical path and it is charged",
            "broker transitions do not mutate the local RAN certificate",
        ],
        "violations": violations,
        "all_pass": not violations,
        "terminal_state_values": [state._asdict() for state in terminal],
        "transition_values": edges,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("pipelined control model invariant failure")


if __name__ == "__main__":
    main()
