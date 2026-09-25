#!/usr/bin/env python3.11
"""Bind an MPS control/server identity snapshot to a completed CUDA probe."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_pids(value: str) -> list[int]:
    return sorted({int(item) for item in value.replace(",", " ").split() if item})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=("before_restart", "after_restart"), required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--control-pids", default="")
    parser.add_argument("--server-pids", default="")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    probe = json.loads(args.probe.read_text())
    pipe = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"])
    control = pipe / "control"
    stat = control.stat()
    snapshot_ns = time.perf_counter_ns()
    value = {
        "schema": "softwall-c164-mps-epoch-snapshot-v1",
        "phase": args.phase,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "clock": "time.perf_counter_ns",
        "snapshot_ns": snapshot_ns,
        "pipe_directory": str(pipe),
        "control_socket": {
            "path": str(control), "inode": stat.st_ino,
            "device": stat.st_dev, "mtime_ns": stat.st_mtime_ns,
        },
        "control_pids": parse_pids(args.control_pids),
        "server_pids": parse_pids(args.server_pids),
        "probe_path": str(args.probe),
        "probe_sha256": sha256(args.probe),
        "probe": probe,
        "all_pass": (
            probe.get("all_pass") is True
            and probe.get("host") == platform.node()
            and probe.get("slurm_job_id") == os.environ.get("SLURM_JOB_ID")
            and probe.get("completed_ns", snapshot_ns + 1) <= snapshot_ns
            and len(probe.get("devices", [])) == 4
            and bool(parse_pids(args.control_pids))
            and bool(parse_pids(args.server_pids))
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
