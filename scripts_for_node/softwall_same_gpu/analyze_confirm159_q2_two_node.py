#!/usr/bin/env python3.11
"""Combine the frozen C159-Q2 development and independent-node holdout runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_confirm159_q2_protocol import source_hashes


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


def campaign(root: Path, label: str) -> dict:
    results = root / "results" / "softwall_multigpu"
    raw = results / "raw"
    paths = {
        "protocol": results / f"{label}_protocol.json",
        "result": results / f"{label}_result.json",
        "peer_spec": results / f"{label}_peer_spec.json",
        "coordinator": raw / f"{label}_coordinator.json",
        "nrx": raw / f"{label}_nrx_worker.json",
        "qwen": raw / f"{label}_qwen.json",
        "inventory": raw / f"{label}_gpu_inventory.csv",
    }
    value = {key: load(path) for key, path in paths.items() if path.suffix == ".json"}
    value["paths"] = paths
    value["owners"] = [
        load(Path(peer["owner_output"])) for peer in value["peer_spec"]["peers"]
    ]
    value["owner_records"] = [
        record for owner in value["owners"] for record in owner.get("records", [])
    ]
    return value


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
    campaigns = [
        campaign(root, args.development_label),
        campaign(root, args.holdout_label),
    ]
    development, holdout = campaigns
    protocols = [row["protocol"] for row in campaigns]
    results = [row["result"] for row in campaigns]
    coordinators = [row["coordinator"] for row in campaigns]
    owner_records = [
        record for row in campaigns for record in row["owner_records"]
    ]
    recovery_rows = [
        item for row in coordinators for item in row.get("physical_recoveries", [])
    ]
    qwen_rows = [
        item for row in coordinators for item in row.get("rounds", [])
        if item.get("qwen") is not None
    ]
    nrx_rows = [
        item for row in campaigns for item in row["nrx"].get("records", [])
    ]
    contexts = tuple(int(value) for value in protocols[0]["ai_offer"]["order"])
    bounds = {
        int(key): float(value)
        for key, value in protocols[0]["mode"]["ai_class_bounds_ms"].items()
    }
    offered = {
        context: sum(
            int(round_row.get("context_length", -1)) == context
            for coordinator in coordinators
            for round_row in coordinator.get("rounds", [])
        )
        for context in contexts
    }
    executed = {
        context: sum(int(row["context_length"]) == context for row in qwen_rows)
        for context in contexts
    }
    certificate_rejected = {
        context: sum(
            int(round_row.get("context_length", -1)) == context
            and round_row.get("lease_accepted") is False
            for coordinator in coordinators
            for round_row in coordinator.get("rounds", [])
        )
        for context in contexts
    }
    launch_guard_rejected = {
        context: sum(
            int(round_row.get("context_length", -1)) == context
            and round_row.get("lease_accepted") is True
            and round_row.get("qwen") is None
            and round_row.get("qwen_attempt", {}).get("launched") is False
            for coordinator in coordinators
            for round_row in coordinator.get("rounds", [])
        )
        for context in contexts
    }
    current_hashes = source_hashes(scripts, task1)
    source_equal = current_hashes == protocols[0]["source_sha256"] == protocols[1]["source_sha256"]
    jobs = [str(row["job_id"]) for row in results]
    nodes = [row["node"] for row in results]
    seeds = [row["seed_base"] for row in protocols]
    required_result_gates = {
        "atomic_observed_outcome_batch",
        "atomic_qwen_lease",
        "bounded_revalidation_and_physical_start_guard",
        "certificate_recovery_conformance",
        "complete_request_sample",
        "global_certificate_each_round",
        "nrx_bound",
        "single_timely_radio_commit",
        "variable_context_class_qualification",
    }
    result_gates_present = all(
        all(result.get("gates", {}).get(gate) is True for gate in required_result_gates)
        for result in results
    )
    atomic_batch_raw = all(
        round_row.get("outcome_transition", {}).get("kind")
            == "atomic_observed_success_batch"
        for coordinator in coordinators
        for round_row in coordinator.get("rounds", [])
    )
    gates = {
        "both_campaigns_pass_all_frozen_gates": (
            all(result.get("all_pass") is True for result in results)
            and result_gates_present
        ),
        "frozen_source_equal_and_current": source_equal,
        "development_then_holdout": (
            protocols[0]["campaign"] == "development"
            and protocols[1]["campaign"] == "holdout"
        ),
        "distinct_job_node_seed": (
            len(set(jobs)) == len(set(nodes)) == len(set(seeds)) == 2
        ),
        "development_node_excluded_from_holdout": (
            nodes[0] in protocols[1].get("excluded_nodes", [])
        ),
        "same_p180_variable_context_mode": (
            protocols[0]["mode"] == protocols[1]["mode"]
            and protocols[0]["mode"]["period_ms"] == 180.0
            and protocols[0]["mode"]["expiry_ms"] == 155
            and protocols[0]["ai_offer"] == protocols[1]["ai_offer"]
        ),
        "combined_sample_exactly_1200_rounds_4800_nrx": (
            sum(row.get("completed_rounds", 0) for row in coordinators) == 1200
            and len(owner_records) == 4800
            and len(nrx_rows) == 4800
        ),
        "each_class_offered_200_and_qualified_per_node": (
            all(offered[context] == 200 for context in contexts)
            and all(
                int(result["summary"]["qwen_executed_by_class"][str(context)])
                    >= int(protocol["ai_offer"]["minimum_executed_per_class"])
                for result, protocol in zip(results, protocols)
                for context in contexts
            )
        ),
        "all_physical_qwen_within_class_bound": all(
            float(row["qwen"]["execution_ms"])
                <= bounds[int(row["context_length"])]
            for row in qwen_rows
        ),
        "bounded_revalidation_activated_both_nodes": all(
            result["summary"].get("launch_revalidation_retry_rounds", 0) >= 1
            and result["summary"].get("launch_revalidation_attempts", {}).get("max", 0) <= 3
            for result in results
        ),
        "atomic_observed_outcome_batch_all_rounds": atomic_batch_raw,
        "zero_deadline_transport_recovery_or_start_guard_violation": (
            not any(record.get("deadline_miss") for record in owner_records)
            and all(record.get("nrx_input_roundtrip_equal") is True
                    for record in owner_records)
            and all(record.get("backward_echo_equal") is True for record in nrx_rows)
            and all(row.get("backward_echo_equal") is True for row in recovery_rows)
            and all(
                record.get("recovery_contract_valid") is True
                and record.get("recovery_input_roundtrip_equal") is True
                for record in owner_records
                if record.get("commit_source") == "shared_conventional"
            )
            and all(
                int(row["qwen"]["worker_accepted_ns"])
                    <= int(row["qwen"]["latest_start_ns"])
                for row in qwen_rows
            )
        ),
        "burstgpt_holdout_still_unmaterialized": all(
            protocol.get("prespec", {}).get("holdout_materialized") is False
            for protocol in protocols
        ),
    }
    artifact_paths: set[Path] = set()
    for row in campaigns:
        artifact_paths.update(row["paths"].values())
        artifact_paths.update(Path(peer["owner_output"])
                              for peer in row["peer_spec"]["peers"])

    qwen_execution_by_class = {
        str(context): summarize([
            float(row["qwen"]["execution_ms"])
            for row in qwen_rows if int(row["context_length"]) == context
        ])
        for context in contexts
    }
    value = {
        "schema": "softwall-confirm159-q2-two-node-result-v1",
        "status": (
            "C159_Q2_TWO_NODE_VARIABLE_P180_PASS"
            if all(gates.values()) else "C159_Q2_TWO_NODE_VARIABLE_P180_FAIL"
        ),
        "all_pass": all(gates.values()),
        "gates": gates,
        "campaigns": [
            {
                "label": protocol["label"],
                "campaign": protocol["campaign"],
                "job_id": result["job_id"],
                "node": result["node"],
                "seed_base": protocol["seed_base"],
                "summary": result["summary"],
            }
            for protocol, result in zip(protocols, results)
        ],
        "summary": {
            "rounds": sum(row.get("completed_rounds", 0) for row in coordinators),
            "actual_nrx_requests": len(owner_records),
            "actual_nrx_successes": sum(
                bool(record.get("timely_success")) for record in owner_records
            ),
            "physical_recoveries": len(recovery_rows),
            "qwen_units": len(qwen_rows),
            "radio_commits": sum(record.get("commit_count", 0)
                                 for record in owner_records),
            "deadline_misses": sum(bool(record.get("deadline_miss"))
                                   for record in owner_records),
            "input_roundtrip_violations": sum(
                record.get("nrx_input_roundtrip_equal") is not True
                for record in owner_records
            ) + sum(
                record.get("recovery_input_roundtrip_equal") is not True
                for record in owner_records
                if record.get("commit_source") == "shared_conventional"
            ),
            "response_echo_violations": sum(
                record.get("backward_echo_equal") is not True for record in nrx_rows
            ) + sum(
                row.get("backward_echo_equal") is not True for row in recovery_rows
            ),
            "recovery_contract_violations": sum(
                record.get("recovery_contract_valid") is not True
                for record in owner_records
                if record.get("commit_source") == "shared_conventional"
            ),
            "launch_revalidation_retry_rounds": sum(
                len(round_row.get("launch_revalidation_attempts", [])) > 1
                for coordinator in coordinators
                for round_row in coordinator.get("rounds", [])
            ),
            "qwen_offered_by_class": {str(key): value for key, value in offered.items()},
            "qwen_executed_by_class": {str(key): value for key, value in executed.items()},
            "qwen_certificate_rejected_by_class": {
                str(key): value for key, value in certificate_rejected.items()
            },
            "qwen_launch_guard_rejected_by_class": {
                str(key): value for key, value in launch_guard_rejected.items()
            },
            "qwen_execution_ms_by_class": qwen_execution_by_class,
            "nrx_release_to_complete_ms": summarize([
                float(record["nrx_release_to_complete_ms"])
                for record in owner_records
            ]),
            "recovery_path_ms": summarize([
                (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
                for row in recovery_rows
            ]),
            "radio_release_to_commit_ms": summarize([
                float(record["release_to_commit_ms"]) for record in owner_records
            ]),
        },
        "claim_scope": (
            "Two-node finite-sample qualification of six variable Qwen prefill "
            "classes under the warm synthetic P180/D155 actual-NeuralRx mode. "
            "It validates class-specific bounded admission, bounded launch-time "
            "revalidation, a physical latest-start gate, and an atomic batch "
            "transition for outcomes observed at one cutoff. It is not an online "
            "BurstGPT performance result, WCET proof, cold/long-idle qualification, "
            "production d_MAC result, or integrated fault qualification."
        ),
        "source_sha256": current_hashes,
        "artifact_sha256": {
            str(path.relative_to(root)): sha256(path)
            for path in sorted(artifact_paths, key=str) if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"],
        "gates": gates,
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
