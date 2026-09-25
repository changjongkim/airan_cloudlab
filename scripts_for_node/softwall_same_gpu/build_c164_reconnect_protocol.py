#!/usr/bin/env python3.11
"""Freeze one C164 physical reconnect campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = (
    "c161_phase2_qwen_worker.py", "c164_reconnect_state_model_v1.py",
    "c164_reconnect_qwen_worker.py", "c164_reconnect_client.py",
    "c164_mandatory_continuity_runner.py",
    "build_c164_reconnect_protocol.py", "analyze_c164_reconnect.py",
    "test_c164_reconnect_physical.py", "run_c164_reconnect.sh",
    "dual_receiver_phy.py", "softwall_phy.py",
)
TASK1_SOURCES = ("nrx_trt_direct.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in SOURCES}
    paths.update({f"task1/{name}": task1_root / name for name in TASK1_SOURCES})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(missing)
    return {key: sha256(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign", choices=("development", "holdout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--excluded-nodes", default="")
    args = parser.parse_args()
    if args.tokens <= 0 or args.tokens % 4:
        parser.error("token count must be a positive multiple of four")
    value = {
        "schema": "softwall-c164-reconnect-protocol-v1",
        "status": "frozen-before-run", "label": args.label,
        "campaign": args.campaign, "seed": args.seed, "tokens": args.tokens,
        "worker_epoch": f"{args.label}-worker",
        "lifecycle_epoch": args.seed,
        "excluded_nodes": sorted({x for x in args.excluded_nodes.split(",") if x}),
        "mode": {
            "cells": 4, "period_ms": 180, "deadline_ms": 155,
            "component_bound_ms": 25, "mps_caps": {"mandatory": 80, "qwen": 20},
            "contexts": [128, 512], "ai_bounds_ms": {"128": 40, "512": 75},
            "faults": ["drop_after_prepare", "drop_after_fence"],
            "expected_per_fault": args.tokens // 2,
            "expected_per_context_fault": args.tokens // 4,
            "post_fence_launch_alignment": "10 ms before the next mandatory release",
        },
        "contract": {
            "same_worker_epoch": True,
            "prepare_loss": "durable prepared record becomes nonlaunch_fenced under reconnect token lock",
            "post_fence_loss": "matching durable fenced record retires the lease exactly once",
            "overlap_gate": "post-fence Qwen launches are aligned before mandatory releases and the launch-call-to-fence wall interval must overlap mandatory execution; this is not a new kernel-overlap trace",
            "mandatory": "all four conventional cells <=25 ms and release <=155 ms",
            "claim_scope": "same-process worker channel reconnect subset; worker process replacement, production d_MAC and WCET remain UQ",
        },
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
