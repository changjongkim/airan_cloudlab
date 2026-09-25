#!/usr/bin/env python3.11
"""Audit an actual B_conv=8 ms four-cell calendar after Confirm82 qualification."""

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
    protocol = read(base / "confirm83_conv8_fourcell_protocol.json")
    prior = read(base / "confirm82_bound_campaign_job58738952.json")
    prefix = protocol["prefix"]
    raw = base / "raw"
    ran = read(raw / f"{prefix}_controller.json")
    workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
    background = read(raw / f"{prefix}_background.json")
    requal = read(raw / f"{prefix}_requalification.json")
    records = ran["records"]
    groups = [records[4 * i:4 * i + 4] for i in range(protocol["iterations"])]
    nrx = [r["nrx_response_ms"] for r in records if r["nrx_response_ms"] is not None]
    conv = [r["conventional_host_path_ms"] for r in records
            if r["conventional_host_path_ms"] is not None]
    joint = [r for r in ran["background_records"]
             if r.get("phase") == "after_nrx_before_recovery"]
    all_fail = sum(
        sum(r["admitted"] for r in group) == 2
        and all(r["forced_nrx_failure"] for r in group if r["admitted"])
        and all(r["commit_kind"] == "conventional" for r in group)
        for group in groups
    )
    two_admitted = sum(sum(r["admitted"] for r in group) == 2 for group in groups)
    gates = {
        "confirm82_go_condition": (
            prior["all_current_contract_gates_pass"]
            and prior["candidate_gates"]["warm_conv_8ms_observed"]
        ),
        "source_hashes": all(
            hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
            for name, digest in protocol["source_sha256_before_run"].items()
        ),
        "contract": (
            ran["schema"] == "softwall-four-cell-two-endpoint-v1"
            and ran["iterations"] == protocol["iterations"]
            and ran["cells"] == 4
            and ran["period_ms"] == 180
            and ran["deadline_ms"] == 155
            and ran["nrx_bound_ms"] == 50
            and ran["conv_bound_ms"] == 8
            and ran["ai_budget_ms"] == 15
            and ran["ai_rpc_timeout_ms"] == 12
            and ran["ai_deadline_ms"] == 50
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["inject_correlated_failure_every"] == 5
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and len(records) == 4 * protocol["iterations"]
            and len(ran["recovery_decisions"]) == protocol["iterations"]
            and ran["runtime_metrics"].get("mandatory_reserved") == len(records)
            and all(r["channel_seed"] == protocol["channel_seed_base"]
                    + r["index"] * 4 + r["cell"] for r in records)
            and requal["iterations"] == 200
            and requal["conv_bound_ms"] == 25
            and requal["deadline_misses"] == 0
        ),
        "provenance": (
            len({x["host"] for x in (ran, *workers, background, requal)}) == 1
            and ran["host"] == protocol["node"]
            and all(str(x["slurm_job_id"]) == protocol["job"]
                    for x in (ran, *workers, background, requal))
            and all(40 <= x["visible_sm_count"] <= 44 for x in workers)
            and int(background["mps_active_thread_percentage"]) == 20
        ),
        "sample_safety": (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and ran["pre_radio_ai_guard_violations"] == 0
            and not ran["background_faults"]
            and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
            and all(x["outstanding"] == 0 for x in ran["endpoint_final"].values())
            and ran["background_units"] == background["completed_units"]
            and all(r["commit_kind"] in ("nrx", "conventional") for r in records)
            and max(conv) <= 8
            and max(nrx) <= 50
        ),
        "physical_coverage": (
            two_admitted >= protocol["min_two_admitted"]
            and all_fail >= protocol["min_injected_all_fail"]
            and len(joint) >= protocol["min_joint_ai"]
            and len(joint) == ran["joint_lease_retired_count"]
            and len(joint) == ran["runtime_metrics"].get("joint_admission_admitted", 0)
            and all(r["returned_ns"] + protocol["ai_guard_ns"]
                    <= r["earliest_fallback_start_ns"] for r in joint)
        ),
    }
    report = {
        "schema": "softwall-confirm83-actual-conv8-calendar-v1",
        "job": protocol["job"], "prefix": prefix,
        "gates": gates, "all_pass": all(gates.values()),
        "nrx_paths": len(nrx), "nrx_max_ms": max(nrx),
        "conventional_host_paths": len(conv), "conventional_host_max_ms": max(conv),
        "conventional_over_8_ms": sum(x > 8 for x in conv),
        "requalification_max_gpu_ms": max(r["gpu_ms"] for r in requal["records"]),
        "two_admitted_releases": two_admitted,
        "injected_all_fail_releases": all_fail,
        "joint_ai_units": len(joint),
        "background_units": ran["background_units"],
        "deadline_misses": ran["deadline_misses"],
        "interpretation": "An actual B_conv8 four-cell prewarmed MPS calendar, conditional on Confirm82. A passing sample is neither WCET nor evidence for a 12-cell contract or joint-policy advantage. Requalification uses B_conv25 and is a separate mode.",
    }
    output = base / f"confirm83_conv8_fourcell_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [k for k, v in gates.items() if not v],
                      "nrx_max_ms": report["nrx_max_ms"],
                      "conventional_host_max_ms": report["conventional_host_max_ms"],
                      "joint_ai_units": len(joint)}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
