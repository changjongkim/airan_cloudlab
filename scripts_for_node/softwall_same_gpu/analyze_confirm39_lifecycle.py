#!/usr/bin/env python3
"""Audit the complete, serial Confirm39 lifecycle comparison against its frozen protocol."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def interval(data: dict) -> tuple[int, int]:
    records = data["records"]
    if len(records) != data["iterations"] or not records:
        raise ValueError("missing release records")
    if [row["index"] for row in records] != list(range(len(records))):
        raise ValueError("release indices are not contiguous")
    starts = [row["release_ns"] for row in records]
    if starts != sorted(starts) or len(set(starts)) != len(starts):
        raise ValueError("release timestamps are not strictly increasing")
    end = max(
        row["release_ns"] + round(row["response_ms"] * 1_000_000)
        for row in records
    )
    return starts[0], end


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    sham = protocol["sham_retirement"]
    maintenance = protocol["maintenance_requalification"]
    names = []
    phases = []
    checks: dict[str, bool] = {}
    expected_host = None

    def register(label: str, data: dict, n: int, period: int, deadline: int) -> None:
        nonlocal expected_host
        host = data["host"]
        expected_host = host if expected_host is None else expected_host
        checks[f"{label}_identity"] = (
            data.get("slurm_job_id") == args.job
            and host == expected_host
            and data["iterations"] == n
            and data["period_ms"] == period
            and data["deadline_ms"] == deadline
        )
        phases.append((label, *interval(data)))

    requal_name = f"confirm39_requalification_job{args.job}.json"
    requal = load(args.raw / requal_name)
    names.append(requal_name)
    register("requalification", requal, 200, 60, 35)
    checks["requalification_zero_miss"] = requal["deadline_misses"] == 0

    sham_misses = 0
    for round_id in range(1, sham["rounds"] + 1):
        for cap in sham["caps"]:
            label = f"sham_r{round_id}_cap{cap}"
            prefix = f"{sham['campaign']}_r{round_id}_cap{cap}_job{args.job}"
            ran_name, worker_name = prefix + "_ran.json", prefix + "_worker.json"
            ran, worker = load(args.raw / ran_name), load(args.raw / worker_name)
            names.extend((ran_name, worker_name))
            register(label, ran, sham["releases_per_run"], sham["period_ms"], sham["deadline_ms"])
            checks[f"{label}_worker_identity"] = (
                worker.get("slurm_job_id") == args.job
                and worker["host"] == expected_host
                and int(worker["mps_active_thread_percentage"]) == cap
            )
            checks[f"{label}_no_optional_work"] = (
                ran["retire_before_first_release"]
                and ran["endpoint_retired"]
                and ran["endpoint_retired_at_release"] == -1
                and ran["stop_response"] == {"ok": True, "stopped": True}
                and ran["injected_overruns"] == 0
                and ran["timeouts_detected"] == 0
                and worker["completed_units"] == 0
                and not ran["outstanding_at_end"]
            )
            checks[f"{label}_radio"] = (
                ran["correct_releases"] == sham["releases_per_run"]
                and ran["deadline_misses"] == 0
            )
            sham_misses += ran["deadline_misses"]

    post_misses = 0
    min_quiet_s = float("inf")
    for round_id in range(1, maintenance["rounds"] + 1):
        for cap in maintenance["caps"]:
            label = f"maintenance_r{round_id}_cap{cap}"
            prefix = f"{maintenance['campaign']}_r{round_id}_cap{cap}_job{args.job}"
            pre_name = prefix + "_pre_ran.json"
            worker_name = prefix + "_worker.json"
            post_name = prefix + "_post_ran.json"
            pre, worker, post = (load(args.raw / name) for name in (pre_name, worker_name, post_name))
            names.extend((pre_name, worker_name, post_name))
            register(label + "_pre", pre, maintenance["fault_phase_releases_per_run"], maintenance["period_ms"], maintenance["deadline_ms"])
            register(label + "_post", post, maintenance["post_maintenance_releases_per_run"], maintenance["period_ms"], maintenance["deadline_ms"])
            pre_interval = interval(pre)
            post_interval = interval(post)
            quiet_s = (post_interval[0] - pre_interval[1]) / 1_000_000_000
            min_quiet_s = min(min_quiet_s, quiet_s)
            checks[f"{label}_worker_identity"] = (
                worker.get("slurm_job_id") == args.job
                and worker["host"] == expected_host
                and int(worker["mps_active_thread_percentage"]) == cap
            )
            checks[f"{label}_fault_drained"] = (
                pre["injected_overruns"] == 1
                and pre["timeouts_detected"] == 1
                and pre["responses_drained"] == 1
                and worker["completed_units"] == 1
                and not pre["outstanding_at_end"]
            )
            checks[f"{label}_quiet_time"] = quiet_s >= maintenance["quiet_seconds_after_client_exit"]
            checks[f"{label}_post_radio"] = (
                post["correct_releases"] == maintenance["post_maintenance_releases_per_run"]
                and post["deadline_misses"] == 0
            )
            post_misses += post["deadline_misses"]

    # A previous campaign was invalid because two release windows overlapped.
    # Audit the intended serial order as well as the existence of all 16 runs.
    checks["exact_artifact_count"] = len(names) == 1 + 2 * sham["rounds"] * len(sham["caps"]) + 3 * maintenance["rounds"] * len(maintenance["caps"])
    checks["no_reused_artifact_names"] = len(names) == len(set(names))
    checks["serial_release_intervals"] = all(
        earlier[2] < later[1] for earlier, later in zip(phases, phases[1:])
    )
    result = {
        "schema": "softwall-confirm39-lifecycle-audit-v1",
        "protocol": args.protocol.name,
        "job": args.job,
        "host": expected_host,
        "sham_releases": sham["rounds"] * len(sham["caps"]) * sham["releases_per_run"],
        "sham_misses": sham_misses,
        "post_maintenance_releases": maintenance["rounds"] * len(maintenance["caps"]) * maintenance["post_maintenance_releases_per_run"],
        "post_maintenance_misses": post_misses,
        "minimum_pre_to_post_quiet_s": min_quiet_s,
        "checks": checks,
        "all_frozen_gates_pass": all(checks.values()),
    }
    lines = [
        "# Confirm39 clean lifecycle audit", "",
        f"- Job/node: {args.job} / {expected_host}",
        f"- Sham: {result['sham_releases']} releases, {sham_misses} misses",
        f"- Post-maintenance: {result['post_maintenance_releases']} releases, {post_misses} misses",
        f"- Minimum observed pre/post quiet interval: {min_quiet_s:.3f} s",
        f"- Complete, same-node, serial frozen gates pass: {result['all_frozen_gates_pass']}",
        "", "## Failed checks", "",
    ]
    lines.extend(f"- {name}" for name, passed in checks.items() if not passed)
    if all(checks.values()):
        lines.append("- None")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    if not result["all_frozen_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
