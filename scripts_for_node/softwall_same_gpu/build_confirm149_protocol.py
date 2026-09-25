#!/usr/bin/env python3.11
"""Freeze the C149 shared conventional-worker canary protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--receiver-seed0", type=int, required=True)
    parser.add_argument("--receiver-seed1", type=int, required=True)
    parser.add_argument("--channel-seed0", type=int, required=True)
    parser.add_argument("--channel-seed1", type=int, required=True)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--owner-source", type=Path, required=True)
    parser.add_argument("--worker-source", type=Path, required=True)
    parser.add_argument("--p2p-source", type=Path, required=True)
    parser.add_argument("--phy-source", type=Path, required=True)
    parser.add_argument("--ipc-source", type=Path, required=True)
    args = parser.parse_args()
    sources = {
        "/softwall/shared_conventional_owner.py": args.owner_source,
        "/softwall/shared_conventional_worker.py": args.worker_source,
        "/softwall/multigpu_p2p_ipc_gate.py": args.p2p_source,
        "/softwall/dual_receiver_phy.py": args.phy_source,
        "/softwall_task1/isca_v2/cuda_ipc_channel.py": args.ipc_source,
    }
    value = {
        "schema": "softwall-v17-shared-conventional-canary-protocol-v1",
        "status": "frozen-before-run",
        "label": args.label,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "placement": {
            "home0_source": "GPU0",
            "home1_source": "GPU1",
            "shared_conventional_worker": "GPU2",
        },
        "global_execution_order": "(sequence, home_id)",
        "transport": (
            "cross-process CUDA IPC handles plus NVLink P2P; payload remains "
            "in GPU memory"
        ),
        "workload": {
            "receiver_seeds": [args.receiver_seed0, args.receiver_seed1],
            "channel_seed_bases": [args.channel_seed0, args.channel_seed1],
            "snr_db": args.snr_db,
            "noise_reference": "pre_fading",
        },
        "scope": (
            "physical shared conventional data-path and global-order canary "
            "only; no deadline, all-fail admission, AI lease, WCET, or V17 "
            "QSU claim"
        ),
        "source_sha256": {
            key: sha256(path) for key, path in sources.items()
        },
    }
    temporary = args.output.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
