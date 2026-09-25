#!/usr/bin/env python3
"""Build V14 by adding pipelined broker control to the V13 envelope."""

import argparse
import copy
import json
from pathlib import Path


V13_ID = "sharded_home_g2_c8_broker_crash_qualified_wall_bound_c139_c140"
V14_ID = "sharded_home_g2_c8_broker_crash_pipelined_control_c142_c144"


def build(v13):
    value = copy.deepcopy(v13)
    value["schema"] = "softwall-sharded-home-envelope-grid-v14"
    value["status"] = "post-C144-pipelined-control-two-node-qualified"
    value["scope"] += (
        " Version 14 moves prepare, abort, and complete to a separate control "
        "connection, revalidates prepared work against the current local "
        "certificate, and keeps only the launch-authorizing commit on the RAN "
        "critical path. C142-C144 qualify AI45+commit7+completion2=54 ms and "
        "the three post-apply fault points on two A100 nodes."
    )
    matches = [row for row in value["scenarios"] if row["id"] == V13_ID]
    if len(matches) != 1:
        raise ValueError("V13 must contain exactly one corrected synchronous mode")
    pipelined = copy.deepcopy(matches[0])
    pipelined["id"] = V14_ID
    pipelined["control_fault_model"] = (
        "broker_fail_stop_async_prepare_complete_sync_commit7"
    )
    pipelined["control_rpc_count"] = 1
    pipelined["control_transaction_budget_ms"] = 7
    pipelined["control_critical_operations"] = ["commit"]
    pipelined["control_deferred_operations"] = [
        "prepare", "abort", "complete",
    ]
    pipelined["prepared_token_policy"] = (
        "revalidate at poll; retain across a short local horizon; abort only "
        "when the request SLO cannot be met"
    )
    pipelined["completion_policy"] = (
        "retire local GPU lease at physical fence; keep global token inflight "
        "until asynchronous complete ACK"
    )
    pipelined["ai_bounds_ms"] = [35.0, 40.0, 45.0, 65.0, 75.0]
    pipelined["qualification_nodes"] = ["nid001361", "nid002049"]
    pipelined["hardware_generalization"] = (
        "same V14 contract qualified on two A100 nodes; cross-family and "
        "production d_MAC remain unqualified"
    )
    pipelined["evidence"] = [
        "The finite V14 protocol model exhausts 16 states and 20 transitions including every before/after-apply control failure branch with zero invariant violations",
        "C142 two AI45 arms on nid001361: 2,560 TB, 305 target branches, 9 exchange-only executions, zero safety/bound/credit violations",
        "The V14 transaction charges AI45 + synchronous launch commit7 + completion guard2 = 54 ms; static all-fail slack is 53 ms and conditional window is 58 ms",
        "C143 six fault arms on nid001361: 16,320 TB, 14,184 after fault, commit max 5.148988 ms, zero safety or duplicate violations",
        "C143 deferred prepare/abort/complete reached 9.614963 ms and exceeded 7 ms 50 times without blocking RAN recovery",
        "C144 independently requalified AI45 and all three fault points on nid002049: 6,080 TB and 3,730 after fault, all gates pass",
        "Prepared work is never launched without commit ACK and is revalidated against the current local horizon",
    ]
    value["scenarios"].append(pipelined)
    return value


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(json.loads(args.input.read_text()))
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
