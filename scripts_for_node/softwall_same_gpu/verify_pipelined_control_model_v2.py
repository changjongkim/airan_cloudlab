#!/usr/bin/env python3
"""V15 finite audit: V14 fault branches plus single-token ownership."""

import argparse
import json
from collections import deque
from pathlib import Path
from typing import NamedTuple

from verify_pipelined_control_model import verify as verify_fault_protocol


class OwnershipState(NamedTuple):
    prepare_pending: bool = False
    staged: bool = False
    offered: bool = False
    broker_held_tokens: int = 0


def ownership_successors(state):
    if not state.prepare_pending and not state.staged and not state.offered:
        yield "poll_submit_prepare", OwnershipState(prepare_pending=True)
    if state.prepare_pending:
        yield "prepare_ack_none", OwnershipState()
        yield "prepare_ack_hold", OwnershipState(
            staged=True, broker_held_tokens=state.broker_held_tokens + 1
        )
    if state.staged:
        yield "short_horizon_retain", state
        yield "launch_window_offer", OwnershipState(
            offered=True, broker_held_tokens=state.broker_held_tokens
        )
        yield "request_expire_abort", OwnershipState()
    if state.offered:
        yield "local_reject_abort", OwnershipState()
        yield "commit_ack", OwnershipState()


def ownership_failures(state, transition):
    failures = []
    tracked = int(state.staged) + int(state.offered)
    if tracked > 1:
        failures.append("more than one client-tracked unlaunched token")
    if state.broker_held_tokens > 1:
        failures.append("more than one broker-held token")
    if state.broker_held_tokens != tracked:
        failures.append("broker-held token is not exactly client-tracked")
    if state.prepare_pending and tracked:
        failures.append("prepare pending behind staged/offered token")
    if transition == "poll_submit_prepare" and tracked:
        failures.append("poll submitted prepare while token already owned")
    return failures


def verify_single_token_ownership():
    initial = OwnershipState()
    queue = deque([initial])
    seen = {initial}
    edges = []
    violations = []
    while queue:
        source = queue.popleft()
        for transition, target in ownership_successors(source):
            edges.append({
                "transition": transition,
                "source": source._asdict(),
                "target": target._asdict(),
            })
            for failure in ownership_failures(target, transition):
                violations.append({
                    "transition": transition,
                    "state": target._asdict(),
                    "invariant": failure,
                })
            if target not in seen:
                seen.add(target)
                queue.append(target)
    return {
        "scope": (
            "one home and one unlaunched-token slot; repeated short-window "
            "polls, offer, abort, and commit; inflight-token lifecycle is "
            "covered by the fault protocol model"
        ),
        "states": len(seen),
        "edges": len(edges),
        "invariants": [
            "at most one broker-held unlaunched token per home",
            "every broker-held unlaunched token is tracked as staged or offered",
            "prepare is not pending while a staged or offered token exists",
        ],
        "violations": violations,
        "all_pass": not violations,
        "state_values": [state._asdict() for state in seen],
        "transition_values": edges,
    }


def verify():
    fault = verify_fault_protocol()
    ownership = verify_single_token_ownership()
    return {
        "schema": "softwall-pipelined-control-finite-model-v2",
        "fault_protocol_model": fault,
        "single_token_ownership_model": ownership,
        "all_pass": fault["all_pass"] and ownership["all_pass"],
        "claim_boundary": (
            "Finite protocol/ownership state audit. Timing, thread scheduling, "
            "broker implementation, and GPU behavior require separate tests."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("V15 pipelined control model invariant failure")


if __name__ == "__main__":
    main()
