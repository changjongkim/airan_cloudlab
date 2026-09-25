#!/usr/bin/env python3.11
"""Record host-process disappearance after a successful MPS termination."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path


def alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--client-pid", type=int, required=True)
    parser.add_argument("--wrapper-pid", type=int, required=True)
    parser.add_argument("--wrapper-wait-status", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = {
        "schema": "softwall-c164-process-exit-marker-v1",
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "clock": "time.perf_counter_ns", "observed_ns": time.perf_counter_ns(),
        "client_pid": args.client_pid, "wrapper_pid": args.wrapper_pid,
        "wrapper_wait_status": args.wrapper_wait_status,
        "client_alive": alive(args.client_pid),
        "wrapper_alive": alive(args.wrapper_pid),
    }
    value["all_pass"] = not value["client_alive"] and not value["wrapper_alive"]
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
