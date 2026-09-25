#!/usr/bin/env python3.11
"""Record the ready-schedule to first-release idle interval on one clock."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schedule", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timeout-s", type=float, default=600.0)
    parser.add_argument("--required-idle-s", type=float, required=True)
    args = parser.parse_args()
    if args.timeout_s <= 0 or args.required_idle_s <= 0:
        parser.error("timeouts must be positive")
    deadline = time.monotonic() + args.timeout_s
    schedule = None
    observed_ns = None
    while time.monotonic() < deadline:
        try:
            raw = args.schedule.read_text()
            candidate = json.loads(raw)
            if candidate.get("first_release_wall_ns") is not None:
                schedule = candidate
                observed_ns = time.perf_counter_ns()
                break
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(.002)
    if schedule is None or observed_ns is None:
        raise TimeoutError("schedule was not published before observer timeout")
    first_release_ns = int(schedule["first_release_wall_ns"])
    gap_ns = first_release_ns - observed_ns
    result = {
        "schema": "softwall-c164-schedule-observer-v1",
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "clock": "time.perf_counter_ns",
        "schedule_observed_ns": observed_ns,
        "first_release_wall_ns": first_release_ns,
        "observed_idle_ns": gap_ns,
        "required_idle_ns": round(args.required_idle_s * 1e9),
        "idle_gate": gap_ns >= round(args.required_idle_s * 1e9),
        "schedule": schedule,
    }
    atomic_json(args.output, result)
    print(json.dumps(result, indent=2, sort_keys=True))
    raise SystemExit(0 if result["idle_gate"] else 1)


if __name__ == "__main__":
    main()
