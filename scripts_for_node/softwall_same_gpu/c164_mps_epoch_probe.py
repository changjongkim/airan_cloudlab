#!/usr/bin/env python3.11
"""Create one real CUDA client epoch on every visible MPS GPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started_ns = time.perf_counter_ns()
    count = cp.cuda.runtime.getDeviceCount()
    devices = []
    for index in range(count):
        with cp.cuda.Device(index):
            value = cp.arange(4096, dtype=cp.float32)
            observed = float(cp.sum(value * cp.float32(2)).item())
            cp.cuda.get_current_stream().synchronize()
            properties = cp.cuda.runtime.getDeviceProperties(index)
            raw_name = properties.get("name", b"")
            name = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
            devices.append({
                "logical_device": index,
                "name": name,
                "probe_value": observed,
                "expected_value": float(4095 * 4096),
                "value_match": observed == float(4095 * 4096),
            })
    completed_ns = time.perf_counter_ns()
    value = {
        "schema": "softwall-c164-mps-epoch-probe-v1",
        "epoch": args.epoch,
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pid": os.getpid(),
        "clock": "time.perf_counter_ns",
        "started_ns": started_ns,
        "completed_ns": completed_ns,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "mps_pipe_directory": os.environ.get("CUDA_MPS_PIPE_DIRECTORY"),
        "devices": devices,
        "all_pass": count == 4 and all(row["value_match"] for row in devices),
    }
    atomic_json(args.output, value)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
