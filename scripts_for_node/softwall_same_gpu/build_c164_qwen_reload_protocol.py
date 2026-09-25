#!/usr/bin/env python3.11
"""Freeze the C164 Qwen-reload mandatory-continuity campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


SOURCES = (
    "c159_q2_qwen_worker.py",
    "dual_receiver_phy.py",
    "softwall_phy.py",
    "c164_mandatory_continuity_runner.py",
    "c164_qwen_reload_episode.py",
    "build_c164_qwen_reload_protocol.py",
    "analyze_c164_qwen_reload.py",
    "test_c164_qwen_reload.py",
    "run_c164_qwen_reload.sh",
)
TASK1_SOURCES = ("nrx_trt_direct.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in SOURCES}
    paths.update({f"task1/{name}": task1_root / name for name in TASK1_SOURCES})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing C164 reload source: {missing}")
    return {key: sha256(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign", choices=("development", "holdout"), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=30)
    parser.add_argument("--excluded-nodes", default="")
    parser.add_argument("--quiet-gap-s", type=float, default=1.0)
    args = parser.parse_args()
    if args.episodes <= 0 or args.quiet_gap_s <= 0:
        parser.error("episode count and quiet gap must be positive")
    events = [
        str(args.raw_dir / f"{args.label}_reload_episode_{index:03d}.json")
        for index in range(1, args.episodes + 1)
    ]
    value = {
        "schema": "softwall-c164-qwen-reload-protocol-v1",
        "status": "frozen-before-run",
        "label": args.label, "campaign": args.campaign,
        "seed": args.seed, "episodes": args.episodes,
        "excluded_nodes": sorted({item for item in args.excluded_nodes.split(",") if item}),
        "mode": {
            "lifecycle": "qwen_reload_first",
            "placement": "four-cell mandatory cuPHY and Qwen2.5-1.5B share physical GPU2 under MPS 80/20",
            "cells": 4, "period_ms": 180, "deadline_ms": 155,
            "component_bound_ms": 25, "mandatory_warmup": 20,
            "qwen_contexts": [16, 32, 64, 128, 256, 512],
            "qwen_warmup_per_context": 3,
            "quiet_gap_s": args.quiet_gap_s,
        },
        "contract": {
            "optional_admission_during_reload": 0,
            "mandatory_rule": "all four conventional results must be correct; every cell <=25 ms and every release <=155 ms",
            "reload_rule": "each fresh Qwen process loads and warms every frozen context while mandatory releases continue, then exits before the next episode",
            "claim_scope": "mandatory continuity during Qwen reload; optional-work availability and WCET are not claimed",
        },
        "event_paths": events,
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"label": args.label, "episodes": args.episodes,
                      "source_sha256": value["source_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
