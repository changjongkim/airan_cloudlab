#!/usr/bin/env python3
"""Build V13 by applying the C138-C140 control-bound correction to V12."""

import argparse
import copy
import json
from pathlib import Path


OLD_ID = "sharded_home_g2_c8_broker_crash_full_budget_control_matrix_c132_c134"
NEW_ID = "sharded_home_g2_c8_broker_crash_qualified_wall_bound_c139_c140"


def build(v12):
    value = copy.deepcopy(v12)
    value["schema"] = "softwall-sharded-home-envelope-grid-v13"
    value["status"] = "post-C140-socket-timeout-wall-bound-separated"
    value["scope"] += (
        " Version 13 rejects the former equality between the 5 ms socket timeout "
        "and an RPC wall-time bound after C138, separates a prospective 7 ms "
        "admission charge, and replaces the invalidated AI40 exchange-only class "
        "with the corrected AI35+control21+completion2 class validated by C140."
    )
    matches = [row for row in value["scenarios"] if row["id"] == OLD_ID]
    if len(matches) != 1:
        raise ValueError("V12 must contain exactly one historical full-budget mode")
    old = matches[0]
    old["control_fault_qualified"] = False
    old["evidence"].extend([
        "C138 prospectively rejected the 5 ms socket timeout as a wall-time bound: faulting prepare 5.205092 ms and concurrent complete 5.830741 ms",
        "The C132-C134 safety executions remain historical, but their 15 ms admission charge is superseded and this exact mode is UQ",
    ])
    corrected = copy.deepcopy(old)
    corrected["id"] = NEW_ID
    corrected["control_fault_model"] = (
        "broker_fail_stop_socket_timeout5_wall_bound7"
    )
    corrected["control_fault_qualified"] = True
    corrected["control_rpc_bound_ms"] = 7
    corrected["control_rpc_count"] = 3
    corrected["control_transaction_budget_ms"] = 21
    corrected["control_budget_accounted_in_admission"] = True
    corrected["qualification_nodes"] = ["nid001252"]
    corrected["hardware_generalization"] = (
        "one A100 node for corrected wall-bound and conditional-class "
        "qualification; independent-node and cross-family requalification pending"
    )
    corrected["evidence"] = [
        "C138 prospective first arm rejected equality between a 5 ms socket timeout and wall-time bound while all non-control safety gates passed",
        "C139 six prospective arms on nid001252: prepare/commit/complete each two, 16,320 TB, 14,919 TB after fault detection, all safety and duplicate gates zero",
        "C139 measured 2,020 attempted RPCs; 12 faulting returns exceeded the 5 ms socket timeout, maximum 5.653235 ms, and none exceeded the prospective 7 ms admission wall bound",
        "The corrected transaction charges AI35 + control21 + completion2 = 58 ms; static all-fail slack is 53 ms and the decision-time conditional window is 58 ms",
        "C140 two prospective arms on the same node: 2,560 TB, 309 target branches, 8 AI35 conditional exchanges, zero safety/bound/credit violations",
        "C140 static counterfactual rejected all 8 exchanges at margin -5 ms while the conditional contract accepted them at model margin 0 ms",
        "v13 preserves one-buffer CUDA-IPC ring depth and certificate-ordered recovery execution",
    ]
    value["scenarios"].append(corrected)
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
