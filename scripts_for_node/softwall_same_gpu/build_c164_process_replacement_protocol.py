#!/usr/bin/env python3.11
"""Freeze a physical MPS worker-process replacement campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = (
    "c161_phase2_qwen_worker.py", "c164_reconnect_state_model_v1.py",
    "c164_process_replacement_model_v1.py",
    "c164_process_replacement_qwen_worker.py",
    "c164_process_replacement_client.py", "c164_mps_client_control.py",
    "c164_process_exit_marker.py", "build_c164_quiescence_certificate.py",
    "c164_recover_replacement_journal.py",
    "c164_mandatory_continuity_runner.py",
    "build_c164_process_replacement_protocol.py",
    "analyze_c164_process_replacement.py",
    "test_c164_process_replacement_physical.py",
    "run_c164_process_replacement.sh", "dual_receiver_phy.py", "softwall_phy.py",
)
TASK1_SOURCES = ("nrx_trt_direct.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts: Path, task1: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts / name for name in SOURCES}
    paths.update({f"task1/{name}": task1 / name for name in TASK1_SOURCES})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing: raise FileNotFoundError(missing)
    return {key: sha256(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--socket-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign", choices=("development", "holdout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=8)
    parser.add_argument("--excluded-nodes", default="")
    args = parser.parse_args()
    if args.episodes <= 0 or args.episodes % 4:
        parser.error("episodes must be a positive multiple of four")
    episode_rows = []
    for index in range(args.episodes):
        number = index + 1
        fault = "drop_after_prepare" if index % 2 == 0 else "drop_after_launch"
        context = 128 if (index // 2) % 2 == 0 else 512
        prefix = f"{args.label}_replacement_{number:03d}"
        old_epoch = f"{args.label}-w{index:03d}"
        new_epoch = f"{args.label}-w{index + 1:03d}"
        episode_rows.append({
            "episode": number, "fault_mode": fault, "context_length": context,
            "token": f"{args.label}-token-{number:03d}",
            "old_worker_epoch": old_epoch, "new_worker_epoch": new_epoch,
            "socket": str(args.socket_dir / f"p{index:03d}.sock"),
            "ready": str(args.raw_dir / f"{prefix}_ready.json"),
            "worker_output": str(args.raw_dir / f"{prefix}_worker.json"),
            "worker_log": str(args.raw_dir / f"{prefix}_worker.log"),
            "client": str(args.raw_dir / f"{prefix}_client.json"),
            "journal": str(args.state_dir / f"journal_{number:03d}.json"),
            "before": str(args.raw_dir / f"{prefix}_mps_before.json"),
            "termination": str(args.raw_dir / f"{prefix}_mps_terminate.json"),
            "exit": str(args.raw_dir / f"{prefix}_process_exit.json"),
            "after": str(args.raw_dir / f"{prefix}_mps_after.json"),
            "certificate": str(args.raw_dir / f"{prefix}_quiescence_certificate.json"),
        })
    value = {
        "schema": "softwall-c164-process-replacement-protocol-v1",
        "status": "frozen-before-run", "label": args.label,
        "campaign": args.campaign, "seed": args.seed,
        "lifecycle_epoch": args.seed, "episodes": episode_rows,
        "excluded_nodes": sorted({x for x in args.excluded_nodes.split(",") if x}),
        "final_worker": {
            "epoch": f"{args.label}-w{args.episodes:03d}",
            "socket": str(args.socket_dir / "pfinal.sock"),
            "ready": str(args.raw_dir / f"{args.label}_final_worker_ready.json"),
            "output": str(args.raw_dir / f"{args.label}_final_worker_output.json"),
            "log": str(args.raw_dir / f"{args.label}_final_worker.log"),
            "journal": str(args.state_dir / "final_unused_journal.json"),
        },
        "mode": {
            "placement": "four-cell mandatory cuPHY and replaceable Qwen worker share physical GPU2 under MPS 80/20",
            "cells": 4, "period_ms": 180, "deadline_ms": 155,
            "component_bound_ms": 25, "episodes": args.episodes,
            "faults": ["drop_after_prepare", "drop_after_launch"],
            "contexts": [128, 512], "expected_per_pair": args.episodes // 4,
        },
        "contract": {
            "quiescence_primitive": "Legacy MPS v2 terminate_client must return 0 before SIGKILL",
            "required_proof": "target present before; terminate_client=0; process exited; target absent after; same MPS server and mandatory client survive",
            "recovery": "prepared->nonlaunch_fenced; launched->quiescence_fenced; no replay of old token",
            "claim_scope": "MPS-assisted worker process replacement subset; arbitrary GPU/driver failure, production d_MAC and WCET remain UQ",
        },
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
