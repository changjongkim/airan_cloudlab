#!/usr/bin/env python3.11
"""Audit one frozen physical canary of atomic replan plus AI lease."""

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
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm73_atomic_lease_protocol.json")
    prefix = protocol["prefix"]
    raw = base / "raw"
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{cell}.json") for cell in range(2)]
    background = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    radio = ran["records"]
    pre_ai = [row for row in ran["background_records"]
              if row.get("phase") == "before_nrx_observation"]
    joint_ai = [row for row in ran["background_records"]
                if row.get("phase") == "after_nrx_before_recovery"]
    admitted_actions = [row for row in ran["recovery_decisions"]
                        if row["ai_lease"]]
    gates = {
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        ),
        "contract": (
            ran["iterations"] == protocol["iterations"]
            and ran["period_ms"] == protocol["period_ms"]
            and ran["deadline_ms"] == protocol["deadline_ms"]
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and ran["gate_mode"] == "low_threshold"
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["ai_during_nrx"] == "async_one"
            and ran["ai_aware_recovery"] == "joint"
            and ran["early_mandatory"] == "on"
            and ran["ai_deadline_ms"] == protocol["ai_deadline_ms"]
            and len(radio) == 2 * protocol["iterations"]
            and len(ran["recovery_decisions"]) == protocol["iterations"]
            and requal["iterations"] == 200 and requal["deadline_misses"] == 0
        ),
        "provenance": (
            len({x["host"] for x in (ran, *workers, background, requal)}) == 1
            and all(str(x["slurm_job_id"]) == protocol["job"]
                    for x in (ran, *workers, background, requal))
            and all(40 <= worker["visible_sm_count"] <= 44 for worker in workers)
            and int(background["mps_active_thread_percentage"]) == 20
        ),
        "safety": (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["pre_radio_ai_guard_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and not ran["background_faults"] and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
            and all(x["outstanding"] == 0 for x in ran["endpoint_final"].values())
            and ran["background_units"] == background["completed_units"]
            and all(row["commit_kind"] in ("nrx", "conventional") for row in radio)
        ),
        "joint_physical_path": (
            len(joint_ai) >= protocol["min_joint_ai"]
            and len(pre_ai) == ran["pre_radio_ai_units"]
            and len(joint_ai) == ran["ai_before_recovery_units"]
            and len(joint_ai) == ran["joint_lease_retired_count"]
            and len(joint_ai) == len(admitted_actions)
            and len(joint_ai) == ran["runtime_metrics"].get("joint_admission_admitted", 0)
            and len(joint_ai) == ran["runtime_metrics"].get("joint_lease_retired", 0)
            and all(row["joint_lease_retired"] and row["ai_completed_before_recovery"]
                    for row in admitted_actions)
            and all(row["returned_ns"] + protocol["ai_guard_ns"]
                    <= row["earliest_fallback_start_ns"]
                    and row["returned_ns"]
                    <= radio[2 * row["release_index"]]["release_ns"]
                    + round(protocol["ai_deadline_ms"] * 1e6)
                    for row in joint_ai)
        ),
    }
    report = {
        "schema": "softwall-confirm73-atomic-lease-physical-canary-v1",
        "job": protocol["job"], "gates": gates,
        "all_pass": all(gates.values()),
        "correct_cells": ran["correct_cells"],
        "joint_ai_units": len(joint_ai),
        "joint_retime_count": ran["recovery_retime_count"],
        "pre_nrx_ai_units": len(pre_ai),
        "background_units": ran["background_units"],
        "interpretation": "One physical success-path canary of atomic fallback-calendar replan plus bounded AI lease, retired after the AI worker's synchronized GPU response. Fixed low-feature gate, synthetic P150/D130 and homogeneous AI; no PHY-value joint policy, strong baseline superiority, fault continuation, mode-specific worst-case bound, or production MAC expiry claim.",
    }
    path = base / f"confirm73_atomic_lease_job{protocol['job']}.json"
    path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [key for key, ok in gates.items() if not ok],
                      "joint_ai_units": len(joint_ai),
                      "joint_retime_count": ran["recovery_retime_count"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
