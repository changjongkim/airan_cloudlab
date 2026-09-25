#!/usr/bin/env python3
"""Build V15 by demoting V14's uncovered token branch and adding its fix."""

import argparse
import copy
import json
from pathlib import Path


V14_ID = "sharded_home_g2_c8_broker_crash_pipelined_control_c142_c144"
V15_ID = "sharded_home_g2_c8_single_token_pipelined_control_c145_c146"
OWNERSHIP = "at-most-one-staged-or-offered-token-per-home"


def build(v14, node_results):
    value = copy.deepcopy(v14)
    value["schema"] = "softwall-sharded-home-envelope-grid-v15"
    qualified = (
        len(node_results) == 2
        and all(result.get("all_pass") for result in node_results)
        and len({tuple(result.get("nodes", [])) for result in node_results}) == 2
    )
    value["status"] = (
        "post-C146-single-token-pipelined-two-node-qualified"
        if qualified else "v15-single-token-physical-qualification-pending"
    )
    value["scope"] += (
        " Version 15 preserves V14 timing but adds an explicit one-unlaunched-"
        "token-per-home ownership contract. A deterministic two-request audit "
        "found that V14 could issue prepare behind a retained staged token; "
        "V15 suppresses that transition and exposes physical branch counters."
    )
    matches = [row for row in value["scenarios"] if row["id"] == V14_ID]
    if len(matches) != 1:
        raise ValueError("V14 grid must contain exactly one pipelined mode")
    old = matches[0]
    old["single_token_ownership_required"] = True
    old["single_token_ownership_qualified"] = False
    old["ownership_contract"] = OWNERSHIP
    old["evidence"].append(
        "Deterministic V14/V15 two-request audit reproduces one V14 untracked broker-held token; C142-C144 ended drained but did not gate this transition"
    )

    fixed = copy.deepcopy(old)
    fixed["id"] = V15_ID
    fixed["control_fault_model"] = (
        "broker_fail_stop_single_token_async_prepare_complete_sync_commit7"
    )
    fixed["single_token_ownership_qualified"] = qualified
    fixed["prepared_token_policy"] = (
        "retain across short horizon; never issue prepare while staged or "
        "offered token exists; abort only when request SLO cannot be met"
    )
    fixed["qualification_nodes"] = sorted(
        node for result in node_results for node in result.get("nodes", [])
    )
    fixed["hardware_generalization"] = (
        "same V15 contract qualified on two A100 nodes; cross-family and "
        "production d_MAC remain unqualified"
        if qualified else "V15 physical qualification incomplete"
    )
    fixed["evidence"] = [
        "V14/V15 deterministic two-request regression: V14 untracked held token 1, V15 0",
        "V15 finite audit composes the 16-state/20-edge fault protocol with a 4-state/8-edge single-token ownership model; violations 0",
    ]
    for result in node_results:
        totals = result.get("totals", {})
        fixed["evidence"].append(
            "{}: AI45 exchange {}, fault arms {}, radio {}, post-fault {}, suppressed prepares {}, maximum unlaunched token {}".format(
                ",".join(result.get("nodes", [])),
                totals.get("ai45_exchanges"),
                totals.get("fault_arms"),
                totals.get("radio_records"),
                totals.get("radio_after_fault"),
                totals.get("suppressed_prepare_due_owned_token"),
                totals.get("maximum_unlaunched_tokens"),
            )
        )
    value["scenarios"].append(fixed)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--node-results", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    v14 = json.loads(args.input.read_text())
    results = [json.loads(path.read_text()) for path in args.node_results]
    value = build(v14, results)
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
