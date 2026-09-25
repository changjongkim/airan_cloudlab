#!/usr/bin/env python3.11
"""Combine C161 phase-1 development and independent-node holdout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_c161_phase1_protocol import source_hashes


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
    raw = results_root / "raw"
    scripts = root / "scripts_for_node/softwall_same_gpu"
    task1 = root / "scripts_for_node/task1"
    campaigns = []
    for label in (args.development_label, args.holdout_label):
        protocol_path = results_root / f"{label}_protocol.json"
        result_path = results_root / f"{label}_result.json"
        coordinator_path = raw / f"{label}_coordinator.json"
        protocol = load(protocol_path)
        result = load(result_path)
        coordinator = load(coordinator_path)
        campaigns.append({
            "label": label,
            "protocol": protocol,
            "result": result,
            "coordinator": coordinator,
            "paths": [protocol_path, result_path, coordinator_path],
        })
    development, holdout = campaigns
    protocols = [row["protocol"] for row in campaigns]
    results = [row["result"] for row in campaigns]
    current_hashes = source_hashes(scripts, task1)
    jobs = [str(row["job_id"]) for row in results]
    nodes = [row["node"] for row in results]
    seeds = [row["seed_base"] for row in protocols]
    summaries = [row["summary"] for row in results]
    expected_gates = {
        "actual_nrx_bound_and_transport", "atomic_batch_all_rounds",
        "complete_sample_and_arm_order", "correlated_all_fail_physically_recovered",
        "fault_arm_semantics", "latest_start_nonlaunch_exercised",
        "no_process_error", "node_inventory_and_exclusion",
        "nofault_qwen_exercised", "physical_p2p_ipc_lifecycle",
        "pre_staged_input_and_full_preflight", "qwen_bound_identity_and_guard",
        "recovery_bound_transport_and_contract", "single_timely_radio_commit",
        "source_hash_match",
    }
    gates = {
        "both_campaigns_pass_all_15_gates": all(
            result.get("all_pass") is True
            and set(result.get("gates", {})) == expected_gates
            and all(result["gates"].values())
            for result in results
        ),
        "frozen_source_equal_and_current": (
            current_hashes == protocols[0]["source_sha256"]
            == protocols[1]["source_sha256"]
        ),
        "development_then_holdout": (
            protocols[0]["campaign"] == "development"
            and protocols[1]["campaign"] == "holdout"
        ),
        "distinct_job_node_seed": (
            len(set(jobs)) == len(set(nodes)) == len(set(seeds)) == 2
        ),
        "development_node_excluded_from_holdout": (
            nodes[0] in protocols[1]["excluded_nodes"]
        ),
        "reverse_arm_order": (
            protocols[1]["fault_plan"]["arm_order"]
                == list(reversed(protocols[0]["fault_plan"]["arm_order"]))
        ),
        "same_mode_faults_and_sample": (
            protocols[0]["mode"] == protocols[1]["mode"]
            and protocols[0]["fault_plan"]["arm_length"]
                == protocols[1]["fault_plan"]["arm_length"] == 30
            and protocols[0]["fault_plan"]["fault_hold_ms"]
                == protocols[1]["fault_plan"]["fault_hold_ms"] == 10.0
        ),
        "combined_sample_exact": (
            sum(row["rounds"] for row in summaries) == 180
            and sum(row["actual_nrx_requests"] for row in summaries) == 720
            and sum(row["radio_commits"] for row in summaries) == 720
        ),
        "correlated_all_fail_240_recoveries": (
            sum(row["correlated_recoveries"] for row in summaries) == 240
        ),
        "guarded_nonlaunch_exercised_both_nodes": all(
            row["guarded_nonlaunches"] > 0 for row in summaries
        ),
        "nofault_qwen_exercised_both_nodes": all(
            row["qwen_units"] > 0 for row in summaries
        ),
        "zero_deadline_miss": sum(row["deadline_misses"] for row in summaries) == 0,
    }
    artifact_paths = []
    for row in campaigns:
        artifact_paths.extend(row["paths"])
        label = row["label"]
        artifact_paths.extend(results_root.glob(f"{label}_*"))
        artifact_paths.extend(raw.glob(f"{label}_*"))
    value = {
        "schema": "softwall-c161-phase1-two-node-v1",
        "status": (
            "C161_PHASE1_TWO_NODE_PASS"
            if all(gates.values()) else "C161_PHASE1_TWO_NODE_FAIL"
        ),
        "all_pass": all(gates.values()),
        "gates": gates,
        "campaigns": [{
            "label": row["label"],
            "campaign": row["protocol"]["campaign"],
            "job_id": row["result"]["job_id"],
            "node": row["result"]["node"],
            "seed_base": row["protocol"]["seed_base"],
            "arm_order": row["protocol"]["fault_plan"]["arm_order"],
            "summary": row["result"]["summary"],
        } for row in campaigns],
        "summary": {
            "rounds": sum(row["rounds"] for row in summaries),
            "actual_nrx_requests": sum(
                row["actual_nrx_requests"] for row in summaries
            ),
            "actual_nrx_successes": sum(
                row["actual_nrx_successes"] for row in summaries
            ),
            "effective_successes": sum(
                row["effective_successes"] for row in summaries
            ),
            "physical_recoveries": sum(
                row["physical_recoveries"] for row in summaries
            ),
            "correlated_recoveries": sum(
                row["correlated_recoveries"] for row in summaries
            ),
            "qwen_units": sum(row["qwen_units"] for row in summaries),
            "guarded_nonlaunches": sum(
                row["guarded_nonlaunches"] for row in summaries
            ),
            "radio_commits": sum(row["radio_commits"] for row in summaries),
            "deadline_misses": sum(row["deadline_misses"] for row in summaries),
            "maxima_ms": {
                key: max(row[key] for row in summaries)
                for key in (
                    "nrx_max_ms", "recovery_max_ms", "qwen_max_ms",
                    "commit_max_ms",
                )
            },
        },
        "scope": (
            "Two-node finite-sample C161 phase-1 validation of correlated "
            "outcome suppression and physical latest-start non-launch in the "
            "warm P180 actual-NeuralRx/shared-cuPHY path. Response loss, worker "
            "crash, missing completion fence, GPU/driver hang, cold lifecycle, "
            "production d_MAC and cross-family behavior remain unqualified."
        ),
        "source_sha256": current_hashes,
        "artifact_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted(set(artifact_paths), key=str) if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
