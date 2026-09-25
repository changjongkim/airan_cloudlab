#!/usr/bin/env python3.11
"""Audit request-owned PHY / endpoint-owned IPC cross-routing canary."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = root / "results/softwall_same_gpu"
    raw = result / "raw"
    protocol = read(result / "confirm59_queueaware_canary_protocol.json")
    prefix = f"confirm59_queueaware_job{args.job}"
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{cell}.json") for cell in range(2)]
    ai = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    records = ran["records"]
    gates = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    gates["fixed_setup"] = (
        ran["iterations"] == protocol["iterations"]
        and ran["cells"] == 2
        and ran["period_ms"] == protocol["period_ms"]
        and ran["deadline_ms"] == protocol["deadline_ms"]
        and ran["channel_mode"] == "clean"
        and ran["routing_policy"] == "shortest_queue"
        and ran["alternate_admission_order"] is True
        and ran["gate_mode"] == "always"
        and len(records) == 2 * protocol["iterations"]
        and requal["iterations"] == 200 and requal["deadline_misses"] == 0
    )
    gates["same_node_job"] = (
        len({item["host"] for item in (ran, *workers, ai, requal)}) == 1
        and all(str(item["slurm_job_id"]) == args.job for item in (ran, *workers, ai, requal))
        and all(40 <= worker["visible_sm_count"] <= 44 for worker in workers)
        and int(ai["mps_active_thread_percentage"]) == 20
    )
    cross_route = sum(row["endpoint_id"] != f"nrx{row['cell']}" for row in records)
    gates["all_requests_cross_routed_as_planned"] = (
        cross_route == protocol["expected_cross_routed_cells"]
        and all(
            row["endpoint_id"] == f"nrx{1 - row['cell'] if row['index'] % 2 else row['cell']}"
            for row in records
        )
    )
    gates["timing_and_correctness"] = (
        ran["correct_cells"] == len(records)
        and ran["deadline_misses"] == 0
        and ran["nrx_bound_violations"] == 0
        and ran["conv_path_bound_violations"] == 0
        and ran["admission_rejections"] == 0
        and ran["fallbacks"] == 0
        and all(
            row["nrx_commit"] and row["correct"]
            and row["feature_begin_ns"] >= row["release_ns"]
            and row["commit_return_ns"] <= row["release_ns"] + round(protocol["deadline_ms"] * 1e6)
            for row in records
        )
    )
    gates["resource_integrity"] = (
        ran["background_units"] == ai["completed_units"] > 0
        and ran["background_budget_violations"] == 0
        and ran["background_release_crossings"] == 0
        and not ran["background_faults"] and not ran["endpoint_faults"]
        and ran["fallback_calendar_final"]["outstanding"] == 0
        and all(item["outstanding"] == 0 for item in ran["endpoint_final"].values())
        and all(worker["completed_units"] == ran["warmup"] + 1 + protocol["iterations"]
                for worker in workers)
    )
    report = {
        "schema": "softwall-confirm59-queueaware-canary-v1",
        "job": args.job,
        "cross_routed_cells": cross_route,
        "correct_cells": ran["correct_cells"],
        "deadline_misses": ran["deadline_misses"],
        "background_units": ran["background_units"],
        "gates": gates,
        "all_pass": all(gates.values()),
        "interpretation": "Cross-routing transport and logical queue-aware API canary with idle queues; no loaded queue-aware policy gain, joint AI scheduling, or production deadline claim.",
    }
    output = result / f"confirm59_queueaware_canary_job{args.job}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
