#!/usr/bin/env python3
"""Combine C135 and C136 V12 mechanism qualification without hiding clusters."""

import argparse
import json
from pathlib import Path

from analyze_confirm136_v12_requalification import zero_failure_upper
from analyze_confirm135_static_counterfactual import analyze_arm


def controller_nodes(arm_results):
    nodes = set()
    for arm_path in arm_results:
        result = json.loads(Path(arm_path).read_text())
        for home in result["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            nodes.add(controller["host"])
    return nodes


def analyze(c135_path, c136_path):
    c135_path = Path(c135_path).resolve()
    c136_path = Path(c136_path).resolve()
    c135 = json.loads(c135_path.read_text())
    c136 = json.loads(c136_path.read_text())
    old_arm_paths = [arm["result"] for arm in c135["arms"]]
    old_events = [
        event
        for path in old_arm_paths
        for event in analyze_arm(path)["events"]
    ]
    nodes = controller_nodes(old_arm_paths) | set(c136["nodes"])
    arms = len(c135["arms"]) + c136["totals"]["arms"]
    radio = c135["totals"]["radio_records"] + c136["totals"]["radio_records"]
    home_releases = (
        sum(2 * json.loads(Path(arm["protocol"]).read_text())["iterations"]
            for arm in c135["arms"])
        + c136["totals"]["home_releases"]
    )
    exchanges = (
        c135["totals"]["ai40_candidate_exchanges"]
        + c136["totals"]["ai40_candidate_exchanges"]
    )
    branches = (
        c135["totals"]["candidate_branches"]
        + c136["totals"]["candidate_branches"]
    )
    atomic = c135["totals"]["atomic_exchanges"] + c136["totals"]["atomic_exchanges"]
    gates = {
        "both_campaigns_pass": c135["all_pass"] and c136["all_pass"],
        "two_distinct_nodes": len(nodes) == 2,
        "eight_arms": arms == 8,
        "every_arm_has_candidate_exchange": all(
            arm["candidate_evidence"]["exchanges"] > 0 for arm in c135["arms"]
        ) and all(
            arm["ai40_candidate_exchanges"] > 0 for arm in c136["arms"]
        ),
        "all_38_events_static_reject_conditional_accept": exchanges == 38
            and len(old_events) == c135["totals"]["ai40_candidate_exchanges"]
            and all(event["terms"]["static_margin_ms"] < 0
                    and event["terms"]["conditional_margin_ms"] >= 0
                    and all(event["checks"].values())
                    for event in old_events)
            and c136["gates"]["static_counterfactual_rejects_every_exchange"]
            and c136["gates"]["conditional_contract_accepts_every_exchange"],
    }
    return {
        "schema": "softwall-confirm135-136-combined-v12-qualification-v1",
        "inputs": {"c135": str(c135_path), "c136": str(c136_path)},
        "nodes": sorted(nodes),
        "totals": {
            "nodes": len(nodes),
            "arms": arms,
            "radio_records": radio,
            "home_releases": home_releases,
            "atomic_exchanges": atomic,
            "candidate_branches": branches,
            "ai40_candidate_exchanges": exchanges,
            "declared_safety_violations": 0,
        },
        "zero_failure_sensitivity": {
            "confidence": 0.95,
            "tb_level_iid_upper": zero_failure_upper(radio),
            "home_release_level_iid_upper": zero_failure_upper(home_releases),
            "arm_level_iid_upper": zero_failure_upper(arms),
            "node_level_iid_upper": zero_failure_upper(len(nodes)),
            "interpretation": (
                "The four units are alternative IID assumptions, not four "
                "simultaneous guarantees. Node-level evidence is only two "
                "A100 nodes and remains statistically weak; no result is WCET "
                "or cross-family evidence."
            ),
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Two-node, eight-arm, same-A100-family finite-sample mechanism "
            "qualification under one synthetic mode."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--c135", required=True)
    parser.add_argument("--c136", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = analyze(args.c135, args.c136)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(output)
    if not value["all_pass"]:
        raise SystemExit("combined C135/C136 gate failed")


if __name__ == "__main__":
    main()
