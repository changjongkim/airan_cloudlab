#!/usr/bin/env python3.11
"""Combine C161 phase-2 development and independent-node holdout."""

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
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--development-label", required=True)
    parser.add_argument("--holdout-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    results_root = root / "results/softwall_multigpu"
    scripts = root / "scripts_for_node/softwall_same_gpu"
    task1 = root / "scripts_for_node/task1"
    campaigns = []
    for label in (args.development_label, args.holdout_label):
        protocol_path = results_root / f"{label}_protocol.json"
        result_path = results_root / f"{label}_result.json"
        campaigns.append({
            "label": label, "protocol": load(protocol_path),
            "result": load(result_path), "paths": [protocol_path, result_path],
        })
    development, holdout = campaigns
    protocols = [row["protocol"] for row in campaigns]
    results = [row["result"] for row in campaigns]
    summaries = [row["summary"] for row in results]
    current = source_hashes(scripts, task1)
    gates = {
        "both_campaigns_pass": all(row.get("all_pass") is True for row in results),
        "frozen_source_equal_and_current": (
            current == protocols[0]["source_sha256"] == protocols[1]["source_sha256"]
        ),
        "development_then_holdout": (
            protocols[0]["campaign"] == "development"
            and protocols[1]["campaign"] == "holdout"
        ),
        "distinct_job_node_seed": (
            len({str(row["job_id"]) for row in results}) == 2
            and len({row["node"] for row in results}) == 2
            and len({row["seed_base"] for row in protocols}) == 2
        ),
        "development_node_excluded_from_holdout": (
            results[0]["node"] in protocols[1]["excluded_nodes"]
        ),
        "reverse_arm_order": (
            protocols[1]["arm_order"] == list(reversed(protocols[0]["arm_order"]))
        ),
        "combined_sample_exact": (
            sum(row["rounds"] for row in summaries) == 520
            and sum(row["actual_nrx_requests"] for row in summaries) == 2080
            and sum(row["radio_commits"] for row in summaries) == 2080
        ),
        "fault_coverage_exact": (
            sum(row["nrx_faults"] for row in summaries) == 20
            and sum(row["recovery_faults"] for row in summaries) == 20
            and sum(row["terminal_faults"] for row in summaries) == 4
            and sum(row["post_fault_rounds"] for row in summaries) >= 200
        ),
        "zero_deadline_miss": sum(row["deadline_misses"] for row in summaries) == 0,
    }
    artifacts = []
    for row in campaigns:
        artifacts.extend(row["paths"])
        artifacts.extend(results_root.glob(f"{row['label']}*"))
        for arm in row["protocol"]["arms"]:
            artifacts.extend(results_root.glob(f"{arm['label']}*"))
            artifacts.extend((results_root / "raw").glob(f"{arm['label']}*"))
            artifacts.extend(Path(arm["completion_dir"]).glob("*.json"))
    value = {
        "schema": "softwall-c161-phase2-two-node-v1",
        "status": "C161_PHASE2_TWO_NODE_PASS" if all(gates.values()) else "C161_PHASE2_TWO_NODE_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "campaigns": [{
            "label": row["label"], "campaign": row["protocol"]["campaign"],
            "job_id": row["result"]["job_id"], "node": row["result"]["node"],
            "seed_base": row["protocol"]["seed_base"],
            "arm_order": row["protocol"]["arm_order"], "summary": row["result"]["summary"],
        } for row in campaigns],
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
            "maxima_ms": {key: max(row["maxima_ms"][key] for row in summaries)
                          for key in summaries[0]["maxima_ms"]},
        },
        "scope": protocols[0]["scope"],
        "source_sha256": current,
        "artifact_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted(set(artifacts), key=str)
            if path.is_file() and path.is_relative_to(root)
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
