#!/usr/bin/env python3.11
"""Combine the four C161 phase-2 arms from one physical node."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_c161_phase2_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    results = [load(Path(row["result"])) for row in protocol["arms"]]
    summaries = [row["summary"] for row in results]
    jobs = {str(row["job_id"]) for row in results}
    nodes = {row["node"] for row in results}
    gates = {
        "all_four_arms_pass": (
            len(results) == 4 and all(row.get("all_pass") is True for row in results)
        ),
        "arm_order_exact": (
            [row["arm"] for row in results] == list(protocol["arm_order"])
        ),
        "one_job_and_node": len(jobs) == len(nodes) == 1,
        "frozen_source_still_current": (
            source_hashes(args.scripts_root, args.task1_root) == protocol["source_sha256"]
        ),
        "sample_exact": (
            sum(row["rounds"] for row in summaries) == 260
            and sum(row["actual_nrx_requests"] for row in summaries) == 1040
            and sum(row["radio_commits"] for row in summaries) == 1040
        ),
        "event_fault_counts_exact": (
            sum(row["nrx_faults"] for row in summaries) == 10
            and sum(row["recovery_faults"] for row in summaries) == 10
        ),
        "terminal_faults_and_continuation": (
            sum(row["terminal_faults"] for row in summaries) == 2
            and sum(row["post_fault_rounds"] for row in summaries) >= 100
        ),
        "zero_deadline_miss": sum(row["deadline_misses"] for row in summaries) == 0,
    }
    artifacts = {args.protocol}
    for arm, result in zip(protocol["arms"], results):
        artifacts.add(Path(arm["result"]))
        artifacts.add(Path(arm["peer_spec"]))
        artifacts.update(Path(arm["completion_dir"]).glob("*.json"))
        artifacts.update(Path(protocol["arms"][0]["result"]).parent.glob(
            f"{arm['label']}*"
        ))
        artifacts.update(Path(arm["coordinator_output"]).parent.glob(
            f"{arm['label']}*"
        ))
    value = {
        "schema": "softwall-c161-phase2-campaign-result-v1",
        "status": "C161_PHASE2_CAMPAIGN_PASS" if all(gates.values()) else "C161_PHASE2_CAMPAIGN_FAIL",
        "all_pass": all(gates.values()),
        "campaign": protocol["campaign"],
        "job_id": next(iter(jobs)) if len(jobs) == 1 else sorted(jobs),
        "node": next(iter(nodes)) if len(nodes) == 1 else sorted(nodes),
        "seed_base": protocol["seed_base"],
        "arm_order": protocol["arm_order"],
        "gates": gates,
        "summary": {
            "rounds": sum(row["rounds"] for row in summaries),
            "actual_nrx_requests": sum(row["actual_nrx_requests"] for row in summaries),
            "actual_nrx_successes": sum(row["actual_nrx_successes"] for row in summaries),
            "physical_recoveries": sum(row["physical_recoveries"] for row in summaries),
            "qwen_worker_units": sum(row["qwen_worker_units"] for row in summaries),
            "nrx_faults": sum(row["nrx_faults"] for row in summaries),
            "recovery_faults": sum(row["recovery_faults"] for row in summaries),
            "terminal_faults": sum(row["terminal_faults"] for row in summaries),
            "post_fault_rounds": sum(row["post_fault_rounds"] for row in summaries),
            "radio_commits": sum(row["radio_commits"] for row in summaries),
            "deadline_misses": sum(row["deadline_misses"] for row in summaries),
            "maxima_ms": {
                key: max(row[key] for row in summaries)
                for key in ("nrx_max_ms", "recovery_max_ms", "qwen_gpu_max_ms", "commit_max_ms")
            },
        },
        "arms": [{
            "arm": row["arm"], "result": spec["result"], "summary": row["summary"]
        } for spec, row in zip(protocol["arms"], results)],
        "scope": protocol["scope"],
        "artifact_sha256": {
            str(path): sha256(path) for path in sorted(artifacts, key=str) if path.is_file()
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
