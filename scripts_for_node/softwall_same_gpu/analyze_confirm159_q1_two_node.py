#!/usr/bin/env python3.11
"""Combine the frozen C159-Q1 P180 development and independent-node holdout runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_confirm159_q1_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def summarize(values: list[float]) -> dict:
    ordered = sorted(values)
    at = lambda q: ordered[round((len(ordered) - 1) * q)] if ordered else None
    return {
        "count": len(ordered),
        "mean": sum(ordered) / len(ordered) if ordered else None,
        "p50": at(0.50),
        "p99": at(0.99),
        "max": ordered[-1] if ordered else None,
    }


def campaign_rows(peer_spec: dict, coordinator: dict, nrx: dict):
    owners = [load(Path(row["owner_output"])) for row in peer_spec["peers"]]
    owner_records = [row for owner in owners for row in owner["records"]]
    recoveries = coordinator["physical_recoveries"]
    qwen = [row["qwen"] for row in coordinator["rounds"] if row["qwen"]]
    return (
        owners, owner_records, recoveries, qwen, nrx["records"],
        coordinator.get("recovery_preflight", []),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--development-label", required=True)
    parser.add_argument("--holdout-label", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.project_root.resolve()
    scripts = root / "scripts_for_node" / "softwall_same_gpu"
    task1 = root / "scripts_for_node" / "task1"
    results = root / "results" / "softwall_multigpu"
    raw = results / "raw"

    campaigns = []
    for label in (args.development_label, args.holdout_label):
        protocol_path = results / f"{label}_protocol.json"
        result_path = results / f"{label}_result.json"
        spec_path = results / f"{label}_peer_spec.json"
        coordinator_path = raw / f"{label}_coordinator.json"
        nrx_path = raw / f"{label}_nrx_worker.json"
        protocol = load(protocol_path)
        result = load(result_path)
        spec = load(spec_path)
        coordinator = load(coordinator_path)
        nrx = load(nrx_path)
        (
            owners, owner_records, recoveries, qwen, nrx_records,
            recovery_preflight,
        ) = campaign_rows(spec, coordinator, nrx)
        campaigns.append({
            "label": label,
            "protocol_path": protocol_path,
            "result_path": result_path,
            "spec_path": spec_path,
            "coordinator_path": coordinator_path,
            "nrx_path": nrx_path,
            "protocol": protocol,
            "result": result,
            "coordinator": coordinator,
            "nrx": nrx,
            "owners": owners,
            "owner_records": owner_records,
            "recoveries": recoveries,
            "qwen": qwen,
            "nrx_records": nrx_records,
            "recovery_preflight": recovery_preflight,
        })

    development, holdout = campaigns
    all_owner = sum((row["owner_records"] for row in campaigns), [])
    all_recovery = sum((row["recoveries"] for row in campaigns), [])
    all_qwen = sum((row["qwen"] for row in campaigns), [])
    all_nrx_worker = sum((row["nrx_records"] for row in campaigns), [])
    all_preflight = sum((row["recovery_preflight"] for row in campaigns), [])
    current_hashes = source_hashes(scripts, task1)
    source_equal = (
        current_hashes == development["protocol"]["source_sha256"]
        == holdout["protocol"]["source_sha256"]
    )
    nodes = [row["result"]["node"] for row in campaigns]
    jobs = [str(row["result"]["job_id"]) for row in campaigns]
    seeds = [row["protocol"]["seed_base"] for row in campaigns]

    gates = {
        "both_campaigns_pass": all(row["result"]["all_pass"] for row in campaigns),
        "frozen_source_equal_and_current": source_equal,
        "development_then_holdout": (
            development["protocol"]["campaign"] == "development"
            and holdout["protocol"]["campaign"] == "holdout"
        ),
        "distinct_job_node_seed": (
            len(set(jobs)) == len(set(nodes)) == len(set(seeds)) == 2
        ),
        "development_node_excluded_from_holdout": (
            nodes[0] in holdout["protocol"]["excluded_nodes"]
        ),
        "same_p180_mode_and_workload": (
            development["protocol"]["mode"] == holdout["protocol"]["mode"]
            and development["protocol"]["mode"]["period_ms"] == 180.0
            and development["protocol"]["mode"]["expiry_ms"] == 155
            and development["protocol"]["accepted_physical_keys"]
                == holdout["protocol"]["accepted_physical_keys"]
        ),
        "combined_sample_exactly_2000": len(all_owner) == 2000,
        "full_recovery_preflight_both_nodes": (
            len(all_preflight) == 8
            and all(row.get("backward_echo_equal") is True
                    for row in all_preflight)
        ),
        "zero_bound_deadline_contract_lifecycle_violation": (
            not any(row.get("deadline_miss") for row in all_owner)
            and all(row.get("nrx_input_roundtrip_equal") is True for row in all_owner)
            and all(row.get("backward_echo_equal") is True for row in all_nrx_worker)
            and all(row.get("backward_echo_equal") is True for row in all_recovery)
            and all(row.get("recovery_contract_valid") is True
                    for row in all_owner
                    if row.get("commit_source") == "shared_conventional")
            and all(row["nrx_release_to_complete_ms"] <= 45 for row in all_owner)
            and all((row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
                    <= 25 for row in all_recovery)
        ),
    }
    artifact_paths = []
    for row in campaigns:
        artifact_paths.extend([
            row["protocol_path"], row["result_path"], row["spec_path"],
            row["coordinator_path"], row["nrx_path"],
        ])
        artifact_paths.extend(Path(peer["owner_output"])
                              for peer in load(row["spec_path"])["peers"])

    value = {
        "schema": "softwall-confirm159-q1-two-node-result-v1",
        "status": (
            "C159_Q1_TWO_NODE_P180_PASS"
            if all(gates.values()) else "C159_Q1_TWO_NODE_P180_FAIL"
        ),
        "all_pass": all(gates.values()),
        "gates": gates,
        "campaigns": [{
            "label": row["label"],
            "job_id": row["result"]["job_id"],
            "node": row["result"]["node"],
            "campaign": row["protocol"]["campaign"],
            "seed_base": row["protocol"]["seed_base"],
            "summary": row["result"]["summary"],
        } for row in campaigns],
        "summary": {
            "rounds": sum(row["coordinator"]["completed_rounds"] for row in campaigns),
            "actual_nrx_requests": len(all_owner),
            "actual_nrx_successes": sum(row["timely_success"] for row in all_owner),
            "physical_recoveries": len(all_recovery),
            "qwen_units": len(all_qwen),
            "radio_commits": sum(row["commit_count"] for row in all_owner),
            "deadline_misses": sum(row["deadline_miss"] for row in all_owner),
            "input_roundtrip_violations": sum(
                row.get("nrx_input_roundtrip_equal") is not True for row in all_owner
            ) + sum(
                row.get("recovery_input_roundtrip_equal") is not True
                for row in all_owner
                if row.get("commit_source") == "shared_conventional"
            ),
            "response_echo_violations": sum(
                row.get("backward_echo_equal") is not True for row in all_nrx_worker
            ) + sum(
                row.get("backward_echo_equal") is not True for row in all_recovery
            ),
            "recovery_contract_violations": sum(
                row.get("recovery_contract_valid") is not True
                for row in all_owner
                if row.get("commit_source") == "shared_conventional"
            ),
            "local_shared_class_disagreements": sum(
                row.get("recovery_local_class_equal") is False
                for row in all_owner
                if row.get("commit_source") == "shared_conventional"
            ),
            "raw_zero_normalized_failures": sum(
                row.get("raw_response_crc") == 0
                and row.get("serialized_crc_status") == 1
                for row in all_recovery
            ),
            "recovery_preflight_path_ms": summarize([
                row["path_ms"] for row in all_preflight
            ]),
            "nrx_release_to_complete_ms": summarize([
                row["nrx_release_to_complete_ms"] for row in all_owner
            ]),
            "recovery_path_ms": summarize([
                (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
                for row in all_recovery
            ]),
            "qwen_execution_ms": summarize([
                row["execution_ms"] for row in all_qwen
            ]),
            "radio_release_to_commit_ms": summarize([
                row["release_to_commit_ms"] for row in all_owner
            ]),
        },
        "claim_scope": (
            "Two-node finite-sample qualification of the warm synthetic "
            "P180/D155 actual-NeuralRx transition mode. The 180 ms period "
            "leaves a 25 ms post-expiry bank-copy interval; all expensive "
            "input/oracle construction and the full P2P/window/cuPHY/response "
            "cold preflight complete before readiness. The radio contract "
            "remains D155 with NRx45, conventional25, AI35. This is not WCET, "
            "cold/long-idle, production d_MAC, integrated fault, variable-"
            "context trace throughput, or cross-family evidence. Pilot and "
            "failed development samples are excluded."
        ),
        "source_sha256": current_hashes,
        "artifact_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted(set(artifact_paths), key=str) if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
