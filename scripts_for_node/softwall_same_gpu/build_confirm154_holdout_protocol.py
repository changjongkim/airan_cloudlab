#!/usr/bin/env python3.11
"""Freeze a four-branch C154/C155 integrated holdout campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrated_shared_recovery_holdout_plan_v1 import BRANCHES
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


SCRIPT_SOURCES = (
    "integrated_shared_recovery_plan_v1.py",
    "integrated_shared_recovery_holdout_plan_v1.py",
    "integrated_shared_recovery_owner.py",
    "integrated_shared_recovery_worker.py",
    "integrated_shared_recovery_holdout_worker.py",
    "shared_recovery_certificate_v1.py",
    "shared_conventional_owner.py",
    "shared_conventional_worker.py",
    "multigpu_p2p_ipc_gate.py",
    "dual_receiver_phy.py",
    "trace_qwen_worker.py",
    "build_confirm154_holdout_protocol.py",
    "build_confirm154_peer_spec.py",
    "analyze_confirm154_holdout_arm.py",
    "analyze_confirm154_holdout_campaign.py",
    "run_confirm154_integrated_holdout.sh",
)
TASK1_SOURCES = ("isca_v2/cuda_ipc_channel.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in SCRIPT_SOURCES}
    paths.update({f"task1/{name}": task1_root / name for name in TASK1_SOURCES})
    return {key: sha256(path) for key, path in paths.items()}


def parse_order(text: str) -> tuple[str, ...]:
    order = tuple(value.strip() for value in text.split(",") if value.strip())
    if len(order) != len(BRANCHES) or set(order) != set(BRANCHES):
        raise argparse.ArgumentTypeError(
            f"branch order must contain exactly {','.join(BRANCHES)}"
        )
    return order


def arm_seeds(seed_base: int, branch_index: int) -> dict:
    receivers = {}
    channels = {}
    for request_index, key in enumerate(REQUEST_ORDER):
        name = "/".join(key)
        receivers[name] = seed_base + branch_index * 10_000 + request_index * 100 + 11
        channels[name] = seed_base + branch_index * 10_000 + request_index * 100 + 51
    return {"receiver": receivers, "channel": channels}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--branch-order", type=parse_order, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--excluded-nodes", required=True)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    args = parser.parse_args()
    config = IntegratedScenarioConfig()
    arms = []
    for index, branch in enumerate(args.branch_order):
        arms.append({
            "arm_index": index,
            "branch": branch,
            "seeds": arm_seeds(args.seed_base, index),
        })
    value = {
        "schema": "softwall-confirm154-integrated-holdout-protocol-v1",
        "status": "frozen-before-run",
        "label": args.label,
        "branch_order": list(args.branch_order),
        "excluded_nodes": sorted({
            value for value in args.excluded_nodes.split(",") if value
        }),
        "mode": {
            "period_ms": config.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "ai_bound_ms": config.ai_bound_ms,
            "guard_ms": config.guard_ms,
            "capacity": config.capacity,
            "qwen_context_length": 64,
            "qwen_model": "Qwen/Qwen2.5-1.5B",
            "snr_db": args.snr_db,
            "warmup": args.warmup,
            "release_lead_ms": args.release_lead_ms,
            "release_semantics": "t_IQ_ready_after_synthetic_channel_preparation",
        },
        "placement": {
            "home0_owners": "GPU0",
            "home1_owners": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS 80/20",
            "GPU3": "unused in controlled-outcome holdout",
        },
        "arms": arms,
        "outcome_source": "prospectively controlled branch injection",
        "required_branches": list(BRANCHES),
        "scope": (
            "Prospective same-source four-branch integrated holdout. Passing "
            "qualifies controlled outcome semantics on one additional A100 node; "
            "actual NeuralRx output, fault matrix, WCET, production d_MAC and "
            "throughput remain outside this campaign."
        ),
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()

