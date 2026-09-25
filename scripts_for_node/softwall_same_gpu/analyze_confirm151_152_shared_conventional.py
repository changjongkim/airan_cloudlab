#!/usr/bin/env python3.11
"""Combine the two frozen C151/C152 shared-conventional physical canaries."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path):
    return json.loads(path.read_text(encoding="utf-8"))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.result) != 2:
        parser.error("exactly two independent result files are required")

    results = [load(path) for path in args.result]
    protocols = [load(Path(value["protocol"])) for value in results]
    workers = [load(Path(value["worker"])) for value in results]
    owners = [
        [load(Path(path)) for path in value["owners"]]
        for value in results
    ]
    artifact_hashes_valid = all(
        sha256(Path(path)) == digest
        for value in results
        for path, digest in value["artifact_sha256"].items()
    )
    source_maps = [value["source_sha256"] for value in protocols]
    nodes = [value["host"] for value in workers]
    expected_per_node = [value["iterations"] * 2 for value in protocols]
    gates = {
        "two_results": len(results) == 2,
        "both_individual_pass": all(value["all_pass"] for value in results),
        "nine_gates_each": all(
            len(value["gates"]) == 9 and all(value["gates"].values())
            for value in results
        ),
        "artifact_hashes_valid": artifact_hashes_valid,
        "identical_frozen_sources": source_maps[0] == source_maps[1],
        "independent_nodes": len(set(nodes)) == 2,
        "independent_workload_seeds": (
            protocols[0]["workload"]["receiver_seeds"]
            != protocols[1]["workload"]["receiver_seeds"]
            and protocols[0]["workload"]["channel_seed_bases"]
            != protocols[1]["workload"]["channel_seed_bases"]
        ),
        "worker_counts": all(
            worker["completed_units"] == expected
            and worker["correct_units"] == expected
            for worker, expected in zip(workers, expected_per_node)
        ),
        "owner_counts_and_integrity": all(
            owner["completed_units"] == protocol["iterations"]
            and owner["decode_errors"] == 0
            and owner["termination_acknowledged"]
            and owner["error"] is None
            for home_owners, protocol in zip(owners, protocols)
            for owner in home_owners
        ),
        "global_order_and_lifecycle": all(
            worker["global_order_exact"]
            and worker["ipc_handles_closed_before_ack"]
            and worker["error"] is None
            for worker in workers
        ),
    }
    result = {
        "schema": "softwall-confirm151-152-shared-conventional-qualification-v1",
        "status": (
            "TWO_NODE_PHYSICAL_PATH_PASS_V17_REMAINS_UQ"
            if all(gates.values()) else "FAIL"
        ),
        "individual_results": [str(path) for path in args.result],
        "nodes": nodes,
        "gates": gates,
        "all_pass": all(gates.values()),
        "totals": {
            "home_units": sum(
                owner["completed_units"]
                for home_owners in owners for owner in home_owners
            ),
            "worker_units": sum(
                worker["completed_units"] for worker in workers
            ),
            "correct_units": sum(
                worker["correct_units"] for worker in workers
            ),
            "decode_errors": sum(
                owner["decode_errors"]
                for home_owners in owners for owner in home_owners
            ),
        },
        "per_node_timing": [
            {
                "node": worker["host"],
                "conventional_gpu_ms": worker["conventional_gpu_ms"],
                "worker_path_ms": worker["worker_path_ms"],
                "forward_gpu_us": worker["forward_gpu_us"],
                "backward_gpu_us": worker["backward_gpu_us"],
            }
            for worker in workers
        ],
        "claim_scope": (
            "two-node finite-sample physical qualification of the shared "
            "conventional P2P data path and deterministic global execution "
            "order; no integrated all-fail admission, external-AI lease, "
            "deadline service bound, WCET, or V17 QSU claim"
        ),
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "nodes": nodes,
        "totals": result["totals"],
        "gates": gates,
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
