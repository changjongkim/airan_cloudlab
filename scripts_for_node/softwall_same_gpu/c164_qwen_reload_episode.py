#!/usr/bin/env python3.11
"""Freeze one Qwen unload/reload interval and its worker evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode", type=int, required=True)
    parser.add_argument("--launch-ns", type=int, required=True)
    parser.add_argument("--ready-ns", type=int, required=True)
    parser.add_argument("--exit-ns", type=int, required=True)
    parser.add_argument("--qwen-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    qwen = json.loads(args.qwen_output.read_text())
    chronology = 0 < args.launch_ns < args.ready_ns < args.exit_ns
    value = {
        "schema": "softwall-c164-qwen-reload-episode-v1",
        "episode": args.episode,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "clock": "time.perf_counter_ns",
        "launch_ns": args.launch_ns,
        "ready_ns": args.ready_ns,
        "exit_ns": args.exit_ns,
        "load_and_warmup_ms": (args.ready_ns - args.launch_ns) / 1e6,
        "ready_to_exit_ms": (args.exit_ns - args.ready_ns) / 1e6,
        "qwen_output": str(args.qwen_output),
        "qwen_output_sha256": sha256(args.qwen_output),
        "qwen": qwen,
        "all_pass": (
            chronology
            and qwen.get("host") == platform.node()
            and qwen.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
            and qwen.get("warmup_per_length") == 3
            and len(qwen.get("warmup", [])) == 18
            and qwen.get("completed_units") == 0
            and not qwen.get("rejected")
            and qwen.get("mps_active_thread_percentage") == "20"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"episode": args.episode,
                      "load_and_warmup_ms": value["load_and_warmup_ms"],
                      "all_pass": value["all_pass"]}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
