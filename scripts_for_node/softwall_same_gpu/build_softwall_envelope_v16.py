#!/usr/bin/env python3
"""Build V16 by requiring physical coverage of all four broker transitions."""

import argparse
import copy
import json
from pathlib import Path


V15_ID = "sharded_home_g2_c8_single_token_pipelined_control_c145_c146"
V16_ID = "sharded_home_g2_c8_four_point_single_token_control_c145_c148"


def build(v15, abort_results):
    value = copy.deepcopy(v15)
    value["schema"] = "softwall-sharded-home-envelope-grid-v16"
    nodes = [node for result in abort_results for node in result.get("nodes", [])]
    qualified = (
        len(abort_results) == 2
        and all(result.get("all_pass") for result in abort_results)
        and len(set(nodes)) == 2
        and all(result.get("gates", {}).get("abort_fault_injected")
                for result in abort_results)
    )
    value["status"] = (
        "post-C148-four-point-two-node-qualified"
        if qualified else "v16-abort-physical-qualification-pending"
    )
    value["scope"] += (
        " Version 16 requires physical post-apply reply-loss coverage for "
        "prepare, abort, commit, and complete. V15 had two-node physical "
        "coverage for prepare/commit/complete but only finite-model and "
        "normal-path evidence for abort."
    )
    matches = [row for row in value["scenarios"] if row["id"] == V15_ID]
    if len(matches) != 1:
        raise ValueError("V15 grid must contain exactly one current mode")
    old = matches[0]
    old["four_point_control_fault_required"] = True
    old["four_point_control_fault_qualified"] = False
    old["evidence"].append(
        "C145-C146 physically injected prepare/commit/complete; abort post-apply reply-loss was not injected"
    )

    fixed = copy.deepcopy(old)
    fixed["id"] = V16_ID
    fixed["control_fault_model"] = (
        "broker_fail_stop_four_point_single_token_async_prepare_abort_complete_sync_commit7"
    )
    fixed["four_point_control_fault_qualified"] = qualified
    fixed["qualification_nodes"] = sorted(
        set(old.get("qualification_nodes", [])) | set(nodes)
    )
    fixed["hardware_generalization"] = (
        "all four post-apply reply-loss points have two independent A100-node "
        "physical arms; cross-family and production d_MAC remain unqualified"
        if qualified else "V16 abort physical qualification incomplete"
    )
    fixed["evidence"] = list(old["evidence"][:-1])
    fixed["evidence"].append(
        "C145-C146: prepare, commit, and complete post-apply fail-stop each passed on two independent A100 nodes"
    )
    for result in abort_results:
        totals = result.get("totals", {})
        fixed["evidence"].append(
            "{}: abort post-apply fail-stop, radio {}, post-fault {}, "
            "suppressed prepares {}, maximum unlaunched token {}, all gates {}".format(
                ",".join(result.get("nodes", [])),
                totals.get("radio_records"),
                totals.get("radio_after_fault"),
                totals.get("suppressed_prepare_due_owned_token"),
                totals.get("maximum_unlaunched_tokens"),
                result.get("all_pass"),
            )
        )
    value["scenarios"].append(fixed)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--abort-results", type=Path, nargs="*", default=[])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(
        json.loads(args.input.read_text()),
        [json.loads(path.read_text()) for path in args.abort_results],
    )
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
