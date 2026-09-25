#!/usr/bin/env python3.11
"""Snapshot and safely terminate one Legacy MPS v2 client."""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def control(command: str) -> dict:
    completed = subprocess.run(
        ["nvidia-cuda-mps-control"], input=command + "\n",
        text=True, capture_output=True, check=False,
    )
    return {"command": command, "returncode": completed.returncode,
            "stdout": completed.stdout, "stderr": completed.stderr}


def parse_clients(raw: str) -> list[dict]:
    clients = []
    for line in raw.splitlines():
        fields = line.split(maxsplit=5)
        if len(fields) < 5 or not all(re.fullmatch(r"\d+", fields[i])
                                      for i in (0, 1, 2)):
            continue
        clients.append({
            "pid": int(fields[0]), "id": int(fields[1]),
            "server_pid": int(fields[2]), "device": fields[3],
            "namespace": fields[4],
            "command": fields[5] if len(fields) > 5 else "",
        })
    return clients


def snapshot(target_pid: int) -> dict:
    observed_ns = time.perf_counter_ns()
    result = control("ps")
    clients = parse_clients(result["stdout"])
    pipe = Path(os.environ["CUDA_MPS_PIPE_DIRECTORY"])
    stat = (pipe / "control").stat()
    return {
        "schema": "softwall-c164-mps-client-snapshot-v1",
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "clock": "time.perf_counter_ns", "observed_ns": observed_ns,
        "control_socket": {"path": str(pipe / "control"),
                           "inode": stat.st_ino, "device": stat.st_dev},
        "control_result": result, "clients": clients,
        "server_pids": sorted({row["server_pid"] for row in clients}),
        "target_match": [row for row in clients if row["pid"] == target_pid],
        "mandatory_match": [row for row in clients if row["pid"] != target_pid],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="operation", required=True)
    snap = sub.add_parser("snapshot")
    snap.add_argument("--target-pid", type=int, required=True)
    snap.add_argument("--output", type=Path, required=True)
    terminate = sub.add_parser("terminate")
    terminate.add_argument("--before", type=Path, required=True)
    terminate.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.operation == "snapshot":
        value = snapshot(args.target_pid)
    else:
        before = json.loads(args.before.read_text())
        matches = before.get("target_match", [])
        if len(matches) != 1:
            raise RuntimeError(f"expected exactly one target MPS client: {matches}")
        target = matches[0]
        started_ns = time.perf_counter_ns()
        result = control(
            f"terminate_client {target['server_pid']} {target['pid']}"
        )
        returned_ns = time.perf_counter_ns()
        lines = [line.strip() for line in result["stdout"].splitlines()
                 if line.strip()]
        result_code = int(lines[-1]) if lines and re.fullmatch(r"-?\d+", lines[-1]) else None
        value = {
            "schema": "softwall-c164-mps-terminate-client-v1",
            "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "clock": "time.perf_counter_ns", "started_ns": started_ns,
            "returned_ns": returned_ns, "target": target,
            "control_result": result, "result_code": result_code,
            "all_pass": result["returncode"] == 0 and result_code == 0,
        }
    atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if args.operation == "snapshot" or value["all_pass"] else 1)


if __name__ == "__main__":
    main()
