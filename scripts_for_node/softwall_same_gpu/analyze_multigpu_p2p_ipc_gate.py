#!/usr/bin/env python3
"""Strict analyzer for the cross-process CUDA-IPC plus P2P gate."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--owner", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    owner = json.loads(args.owner.read_text(encoding="utf-8"))
    worker = json.loads(args.worker.read_text(encoding="utf-8"))
    expected = int(protocol["iterations"])
    timing_warmup = min(int(protocol.get("timing_warmup_units", 20)), expected)
    source_hashes = protocol["source_sha256"]
    observed_hashes = owner["source_sha256"]
    gates = {
        "protocol_hash_matches_owner": source_hashes == observed_hashes,
        "protocol_hash_matches_worker": source_hashes == worker["source_sha256"],
        "iteration_count": owner["completed_units"] == worker["completed_units"] == expected,
        "owner_integrity": owner["integrity_errors"] == 0,
        "worker_integrity": worker["integrity_errors"] == 0,
        "owner_sequence": bool(owner["sequences_contiguous"]),
        "worker_sequence": bool(worker["sequences_contiguous"]),
        "termination_lifecycle": bool(owner["termination_acknowledged"])
        and bool(worker["ipc_handles_closed_before_ack"]),
        "bidirectional_peer_access": all(worker["peer_access"].values()),
        "payload_contract": owner["forward_bytes"] == worker["forward_bytes"] == 1_415_232
        and owner["backward_bytes"] == worker["backward_bytes"] == 314_496,
        "no_recorded_error": owner["error"] is None and worker["error"] is None,
    }
    result = {
        "schema": "softwall-multigpu-p2p-ipc-gate-v1",
        "protocol": str(args.protocol),
        "protocol_sha256": sha256(args.protocol),
        "owner": str(args.owner),
        "owner_sha256": sha256(args.owner),
        "worker": str(args.worker),
        "worker_sha256": sha256(args.worker),
        "iterations": expected,
        "payload": {
            "forward_bytes": owner["forward_bytes"],
            "backward_bytes": owner["backward_bytes"],
        },
        "timing_us": {
            "round_trip": owner["round_trip_us"],
            "forward_gpu": worker["forward_gpu_us"],
            "forward_host": worker["forward_host_us"],
            "backward_gpu": worker["backward_gpu_us"],
            "backward_host": worker["backward_host_us"],
        },
        "timing_warmup_units": timing_warmup,
        "cold_first_unit_us": {
            "round_trip": owner["records"][0]["round_trip_us"] if owner["records"] else None,
            "forward_gpu": worker["records"][0]["forward_gpu_us"] if worker["records"] else None,
            "backward_gpu": worker["records"][0]["backward_gpu_us"] if worker["records"] else None,
        },
        "steady_timing_us": {
            "round_trip": summarize([
                row["round_trip_us"] for row in owner["records"][timing_warmup:]
            ]),
            "forward_gpu": summarize([
                row["forward_gpu_us"] for row in worker["records"][timing_warmup:]
            ]),
            "forward_host": summarize([
                row["forward_host_us"] for row in worker["records"][timing_warmup:]
            ]),
            "backward_gpu": summarize([
                row["backward_gpu_us"] for row in worker["records"][timing_warmup:]
            ]),
            "backward_host": summarize([
                row["backward_host_us"] for row in worker["records"][timing_warmup:]
            ]),
        },
        "analyzer_sha256": sha256(Path(__file__).resolve()),
        "gates": gates,
        "all_pass": all(gates.values()),
        "scope": (
            "cross-process CUDA IPC mapping plus bidirectional full-GPU P2P "
            "transport integrity; no NeuralRx service, MPS co-run, or deadline claim"
        ),
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2))
    if not result["all_pass"]:
        raise SystemExit("multi-GPU P2P IPC gate failed")


if __name__ == "__main__":
    main()
