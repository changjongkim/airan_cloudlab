#!/usr/bin/env python3.11
"""Combine the passing C156 and independent C156b GPU timeline arms."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def seed_set(protocol: dict) -> set[int]:
    values = set()
    for arm in protocol["arms"]:
        for group in arm["seeds"].values():
            values.update(int(value) for value in group.values())
    return values


def combine(
    first_result: dict, second_result: dict,
    first_protocol: dict, second_protocol: dict,
) -> dict:
    summaries = (first_result["summary"], second_result["summary"])
    first_seeds = seed_set(first_protocol)
    second_seeds = seed_set(second_protocol)
    gates = {
        "both_timeline_campaigns_pass": (
            first_result["all_pass"] and second_result["all_pass"]
        ),
        "distinct_jobs_and_nodes": (
            first_result["job_id"] != second_result["job_id"]
            and first_result["node"] != second_result["node"]
        ),
        "second_protocol_excludes_first_node": (
            first_result["node"] in second_protocol["excluded_nodes"]
        ),
        "source_hashes_identical": (
            first_protocol["source_sha256"] == second_protocol["source_sha256"]
        ),
        "mode_and_profiler_identical": (
            first_protocol["mode"] == second_protocol["mode"]
            and first_protocol["profiler"] == second_protocol["profiler"]
        ),
        "seeds_disjoint": first_seeds.isdisjoint(second_seeds),
        "same_gate_set_all_true": (
            set(first_result["gates"]) == set(second_result["gates"])
            and all(first_result["gates"].values())
            and all(second_result["gates"].values())
        ),
        "combined_counts_and_safety": (
            sum(row["qwen_runtime_kernels"] for row in summaries) == 2448
            and sum(row["recovery_runtime_kernels"] for row in summaries) == 212
            and sum(row["worker_runtime_events"] for row in summaries) == 308
            and sum(row["nvtx_ranges"] for row in summaries) == 16
            and sum(row["physical_recoveries"] for row in summaries) == 4
            and sum(row["correct_home_commits"] for row in summaries) == 4
            and sum(row["deadline_misses"] for row in summaries) == 0
            and sum(row["uncovered_worker_events"] for row in summaries) == 0
            and sum(
                row["forbidden_qwen_recovery_kernel_overlap_ns"]
                for row in summaries
            ) == 0
        ),
    }
    return {
        "schema": "softwall-confirm156-two-node-gpu-timeline-v1",
        "status": (
            "TWO_NODE_V17_1_GPU_TIMELINE_SEMANTICS_PASS_CONTROLLED_OUTCOME"
            if all(gates.values()) else "TWO_NODE_V17_1_GPU_TIMELINE_GATE_FAIL"
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
        "jobs": [first_result["job_id"], second_result["job_id"]],
        "nodes": [first_result["node"], second_result["node"]],
        "totals": {
            key: sum(row[key] for row in summaries)
            for key in (
                "qwen_runtime_kernels", "recovery_runtime_kernels",
                "worker_runtime_events", "nvtx_ranges",
                "uncovered_worker_events",
                "forbidden_qwen_recovery_kernel_overlap_ns",
                "physical_recoveries", "correct_home_commits",
                "deadline_misses",
            )
        },
        "claim_scope": (
            "Two A100 nodes, disjoint seeds and identical V17.1 source reproduced "
            "the controlled conditional-open GPU timeline semantics. This is not "
            "an actual NeuralRx outcome, fault, WCET, throughput, cross-family or "
            "production d_MAC qualification."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first-result", type=Path, required=True)
    parser.add_argument("--second-result", type=Path, required=True)
    parser.add_argument("--first-protocol", type=Path, required=True)
    parser.add_argument("--second-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = combine(
        load(args.first_result), load(args.second_result),
        load(args.first_protocol), load(args.second_protocol),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "status": value["status"],
        "all_pass": value["all_pass"],
        "totals": value["totals"],
    }, indent=2, sort_keys=True))
    if not value["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
