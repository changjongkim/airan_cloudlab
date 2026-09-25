#!/usr/bin/env python3.11
"""Gate the C149 shared conventional-worker physical data-path canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab")
SOURCE_MAP = {
    "/softwall/shared_conventional_owner.py":
        "scripts_for_node/softwall_same_gpu/shared_conventional_owner.py",
    "/softwall/shared_conventional_worker.py":
        "scripts_for_node/softwall_same_gpu/shared_conventional_worker.py",
    "/softwall/multigpu_p2p_ipc_gate.py":
        "scripts_for_node/softwall_same_gpu/multigpu_p2p_ipc_gate.py",
    "/softwall/dual_receiver_phy.py":
        "scripts_for_node/softwall_same_gpu/dual_receiver_phy.py",
    "/softwall_task1/isca_v2/cuda_ipc_channel.py":
        "scripts_for_node/task1/isca_v2/cuda_ipc_channel.py",
}


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--owner0", type=Path, required=True)
    parser.add_argument("--owner1", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    owners = [
        json.loads(args.owner0.read_text(encoding="utf-8")),
        json.loads(args.owner1.read_text(encoding="utf-8")),
    ]
    worker = json.loads(args.worker.read_text(encoding="utf-8"))
    inventory = args.inventory.read_text(encoding="utf-8")
    expected = protocol["iterations"]
    source_ok = all(
        sha256(ROOT / SOURCE_MAP[key]) == value
        for key, value in protocol["source_sha256"].items()
    )
    workload = protocol["workload"]
    configuration_match = all(
        owner["receiver_seed"] == workload["receiver_seeds"][home]
        and owner["channel_seed_base"]
        == workload["channel_seed_bases"][home]
        and owner["snr_db"] == workload["snr_db"]
        for home, owner in enumerate(owners)
    )
    gates = {
        "source_hash": source_ok,
        "configuration_match": configuration_match,
        "three_a100_mig_off": (
            inventory.count("A100-SXM4-40GB") >= 3
            and "Enabled" not in inventory
        ),
        "owner_counts": all(
            value["completed_units"] == expected for value in owners
        ),
        "owner_integrity": all(
            value["decode_errors"] == 0
            and value["sequences_contiguous"]
            and value["termination_acknowledged"]
            and value["error"] is None
            for value in owners
        ),
        "shared_worker_count": (
            worker["completed_units"] == 2 * expected
            and worker["correct_units"] == 2 * expected
        ),
        "global_order_exact": worker["global_order_exact"],
        "peer_access": all(
            value == 1 for value in worker["peer_access"].values()
        ),
        "ipc_lifecycle": (
            worker["ipc_handles_closed_before_ack"]
            and worker["error"] is None
        ),
    }
    artifacts = (
        args.protocol, args.owner0, args.owner1, args.worker, args.inventory
    )
    result = {
        "schema": "softwall-confirm149-shared-conventional-canary-v1",
        "status": (
            "PHYSICAL_PATH_PASS_V17_REMAINS_UQ"
            if all(gates.values()) else "FAIL"
        ),
        "protocol": str(args.protocol),
        "owners": [str(args.owner0), str(args.owner1)],
        "worker": str(args.worker),
        "inventory": str(args.inventory),
        "gates": gates,
        "all_pass": all(gates.values()),
        "summary": {
            "home_units": [value["completed_units"] for value in owners],
            "worker_units": worker["completed_units"],
            "worker_path_ms": worker["worker_path_ms"],
            "conventional_gpu_ms": worker["conventional_gpu_ms"],
            "forward_gpu_us": worker["forward_gpu_us"],
            "backward_gpu_us": worker["backward_gpu_us"],
        },
        "scope": protocol["scope"],
        "artifact_sha256": {
            str(path): sha256(path) for path in artifacts
        },
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": result["status"],
        "gates": gates,
        "summary": result["summary"],
    }, indent=2, sort_keys=True))
    raise SystemExit(0 if result["all_pass"] else 1)


if __name__ == "__main__":
    main()
