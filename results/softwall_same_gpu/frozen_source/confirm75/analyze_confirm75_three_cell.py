#!/usr/bin/env python3.11
"""Frozen audit of three-cell all-fail and two-endpoint physical bring-up."""

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
    protocol = read(base / "confirm75_three_cell_protocol.json")
    raw = base / "raw"
    prefix = protocol["prefix"]
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{endpoint}.json")
               for endpoint in range(2)]
    background = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    records = ran["records"]
    groups = [records[3 * index:3 * index + 3]
              for index in range(protocol["iterations"])]
    two_admitted = sum(sum(row["admitted"] for row in group) == 2
                       for group in groups)
    cell2_routed = sum(row["cell"] == 2 and row["admitted"]
                       and row["endpoint_id"] in ("nrx0", "nrx1")
                       for row in records)
    three_conventional = sum(
        all(row["commit_kind"] == "conventional" for row in group)
        for group in groups
    )
    correlated_all_fail = sum(
        sum(row["admitted"] for row in group) == 2
        and all(row["forced_nrx_failure"] for row in group if row["admitted"])
        and all(row["commit_kind"] == "conventional" for row in group)
        for group in groups
    )
    gates = {
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        ),
        "contract": (
            ran["iterations"] == protocol["iterations"]
            and ran["cells"] == 3
            and ran["period_ms"] == protocol["period_ms"]
            and ran["deadline_ms"] == protocol["deadline_ms"]
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and ran["gate_mode"] == "low_threshold"
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["early_mandatory"] == "on"
            and ran["ai_during_nrx"] == "async_one"
            and ran["ai_aware_recovery"] == "joint"
            and ran["inject_correlated_failure_every"]
                == protocol["inject_correlated_failure_every"]
            and len(records) == 3 * protocol["iterations"]
            and len(ran["recovery_decisions"]) == protocol["iterations"]
            and ran["runtime_metrics"].get("mandatory_reserved")
                == 3 * protocol["iterations"]
            and requal["iterations"] == 200 and requal["deadline_misses"] == 0
            and all(row["channel_seed"]
                    == protocol["channel_seed_base"] + row["index"] * 3 + row["cell"]
                    for row in records)
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
            and all(row["commit_kind"] in ("nrx", "conventional") for row in records)
        ),
        "two_endpoint_route": (
            two_admitted >= protocol["min_two_admitted_releases"]
            and cell2_routed >= protocol["min_cell2_routed"]
            and all(sum(row["admitted"] for row in group) <= 2 for group in groups)
        ),
        "correlated_all_fail": (
            correlated_all_fail >= protocol["min_correlated_all_fail_releases"]
            and three_conventional >= correlated_all_fail
            and all(row["forced_nrx_failure"]
                    == (row["admitted"] and row["index"]
                        % protocol["inject_correlated_failure_every"] == 0)
                    for row in records)
        ),
    }
    report = {
        "schema": "softwall-confirm75-three-cell-two-endpoint-canary-v1",
        "job": protocol["job"], "gates": gates,
        "all_pass": all(gates.values()),
        "correct_cells": ran["correct_cells"],
        "background_units": ran["background_units"],
        "two_admitted_releases": two_admitted,
        "cell2_routed": cell2_routed,
        "three_conventional_releases": three_conventional,
        "correlated_all_fail_releases": correlated_all_fail,
        "joint_ai_units": ran["ai_before_recovery_units"],
        "joint_retimes": ran["recovery_retime_count"],
        "interpretation": "Three simultaneous radio TBs, two cap40 NeuralRx endpoints, one cap20 AI client and one conservative conventional recovery lane on one GPU. Injected branch outcome forces simultaneous fallback for both admitted NeuralRx in qualifying releases. Synthetic P150/D130 and sample bounds only; no actual GPU WCET or production MAC expiry. Fixed low gate, so not a PHY-value joint-policy or strong-baseline throughput comparison.",
    }
    output = base / f"confirm75_three_cell_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_pass": report["all_pass"],
        "failed": [key for key, ok in gates.items() if not ok],
        "correlated_all_fail_releases": correlated_all_fail,
        "cell2_routed": cell2_routed,
        "joint_ai_units": ran["ai_before_recovery_units"],
    }, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
