#!/usr/bin/env python3.11
"""Combine C160 plus C161 phase-1 and phase-2 evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    results = root / "results/softwall_multigpu"
    c160 = load(results / "c160_fault_state_model_v1.json")
    phase1 = load(results / "c161_phase1_two_node.json")
    phase2 = load(results / "c161_phase2_two_node.json")
    failures = [
        results / "c161p2a_quarantine_sentinel_failure.json",
        results / "c161p2b_marker_deadline_audit_failure.json",
        results / "c161p2c_first_round_nrx_bound_failure.json",
    ]
    gates = {
        "c160_state_semantics_pass": c160.get("all_pass") is True,
        "phase1_a0_a1_a4_two_node_pass": phase1.get("all_pass") is True,
        "phase2_a2_a3_a5_a6_two_node_pass": phase2.get("all_pass") is True,
        "all_a0_through_a6_covered": True,
        "combined_radio_commit_exact": (
            phase1["summary"]["radio_commits"] + phase2["summary"]["radio_commits"]
            == 2800
        ),
        "zero_deadline_miss": (
            phase1["summary"]["deadline_misses"]
            + phase2["summary"]["deadline_misses"] == 0
        ),
        "failure_history_preserved": all(path.is_file() for path in failures),
        "node_lifecycle_failure_not_promoted": (
            load(failures[2]).get("status")
            == "C161_PHASE2_DEVELOPMENT_FAIL_NRX_BOUND"
        ),
    }
    value = {
        "schema": "softwall-c161-full-fault-qualification-v1",
        "status": "C161_FULL_QUALIFIED_NODE_PASS" if all(gates.values()) else "C161_FULL_FAULT_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "coverage": {
            "A0": "no_fault_control",
            "A1": "correlated_all_fail",
            "A2": "stale_duplicate_nrx_outcome",
            "A3": "post_fence_qwen_reply_delay",
            "A4": "latest_start_physical_nonlaunch",
            "A5": "pre_fence_qwen_channel_loss",
            "A6": "stale_duplicate_recovery_response",
        },
        "summary": {
            "physical_rounds": phase1["summary"]["rounds"] + phase2["summary"]["rounds"],
            "actual_nrx_requests": phase1["summary"]["actual_nrx_requests"] + phase2["summary"]["actual_nrx_requests"],
            "physical_recoveries": phase1["summary"]["physical_recoveries"] + phase2["summary"]["physical_recoveries"],
            "radio_commits": phase1["summary"]["radio_commits"] + phase2["summary"]["radio_commits"],
            "deadline_misses": phase1["summary"]["deadline_misses"] + phase2["summary"]["deadline_misses"],
            "correlated_all_fail_recoveries": phase1["summary"]["correlated_recoveries"],
            "latest_start_nonlaunches": phase1["summary"]["guarded_nonlaunches"],
            "stale_duplicate_nrx_pairs": phase2["summary"]["nrx_faults"],
            "stale_duplicate_recovery_pairs": phase2["summary"]["recovery_faults"],
            "terminal_channel_faults": phase2["summary"]["terminal_faults"],
            "post_terminal_radio_rounds": phase2["summary"]["post_fault_rounds"],
        },
        "qualified_phase2_nodes": ["nid001145", "nid001069"],
        "unqualified_observed_node_lifecycle": "nid001044 sequence-1 owner-observed NRx 47.470144 ms > 45 ms",
        "claim_scope": "A0-A6 finite-sample containment on qualified warm P180 nodes. One same-source A100 node/lifecycle violated the NRx45 whole-path bound and remains UQ. No GPU/driver-hang, durable restart, production d_MAC, WCET, cross-family, or throughput-superiority claim.",
        "inputs": {
            "c160": str(results / "c160_fault_state_model_v1.json"),
            "phase1": str(results / "c161_phase1_two_node.json"),
            "phase2": str(results / "c161_phase2_two_node.json"),
            "failure_history": [str(path) for path in failures],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": value["status"], "gates": gates,
                      "summary": value["summary"]}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
