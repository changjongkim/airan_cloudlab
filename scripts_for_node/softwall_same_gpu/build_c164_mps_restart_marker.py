#!/usr/bin/env python3.11
"""Prove one quiescent MPS daemon teardown and fresh daemon epoch."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build(pre: dict, post: dict, *, stop_started_ns: int,
          stop_completed_ns: int, control_absent: bool,
          server_absent: bool) -> dict:
    pre_control = set(pre.get("control_pids", []))
    post_control = set(post.get("control_pids", []))
    pre_server = set(pre.get("server_pids", []))
    post_server = set(post.get("server_pids", []))
    checks = {
        "both_cuda_epoch_probes_pass": pre.get("all_pass") is True and post.get("all_pass") is True,
        "same_allocation_clock_domain": (
            pre.get("host") == post.get("host")
            and pre.get("slurm_job_id") == post.get("slurm_job_id")
            and pre.get("clock") == post.get("clock") == "time.perf_counter_ns"
        ),
        "daemon_fully_absent_between_epochs": control_absent and server_absent,
        "strict_restart_chronology": (
            0 < pre.get("snapshot_ns", 0) < stop_started_ns
            < stop_completed_ns < post.get("snapshot_ns", 0)
        ),
        "control_epoch_identity_changed": (
            bool(pre_control) and bool(post_control)
            and pre_control.isdisjoint(post_control)
            and pre.get("control_socket", {}).get("inode")
                != post.get("control_socket", {}).get("inode")
        ),
        "server_epoch_identity_changed": (
            bool(pre_server) and bool(post_server)
            and pre_server.isdisjoint(post_server)
        ),
    }
    return {
        "schema": "softwall-c164-mps-restart-marker-v1",
        "status": "MPS_RESTART_PROVEN" if all(checks.values()) else "MPS_RESTART_NOT_PROVEN",
        "all_pass": all(checks.values()),
        "checks": checks,
        "clock": "time.perf_counter_ns",
        "stop_started_ns": stop_started_ns,
        "stop_completed_ns": stop_completed_ns,
        "control_absent_after_stop": control_absent,
        "server_absent_after_stop": server_absent,
        "before": pre,
        "after": post,
        "created_ns": time.perf_counter_ns(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--stop-started-ns", type=int, required=True)
    parser.add_argument("--stop-completed-ns", type=int, required=True)
    parser.add_argument("--control-absent", choices=("true", "false"), required=True)
    parser.add_argument("--server-absent", choices=("true", "false"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = build(
        load(args.before), load(args.after),
        stop_started_ns=args.stop_started_ns,
        stop_completed_ns=args.stop_completed_ns,
        control_absent=args.control_absent == "true",
        server_absent=args.server_absent == "true",
    )
    value["artifact_sha256"] = {
        "before": sha256(args.before), "after": sha256(args.after),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": value["status"], "checks": value["checks"]}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
