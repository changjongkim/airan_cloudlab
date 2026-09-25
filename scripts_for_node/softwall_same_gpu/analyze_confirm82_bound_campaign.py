#!/usr/bin/env python3.11
"""Frozen audit of two long persistent four-cell service-bound samples."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(fraction * len(ordered)) - 1)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    raw = base / "raw"
    protocol = read(base / "confirm82_bound_campaign_protocol.json")
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms = []
    for arm in protocol["arms"]:
        prefix = arm["prefix"]
        ran = read(raw / f"{prefix}_controller.json")
        workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
        background = read(raw / f"{prefix}_background.json")
        requal = read(raw / f"{prefix}_requalification.json")
        records = ran["records"]
        nrx = [row["nrx_response_ms"] for row in records
               if row["nrx_response_ms"] is not None]
        conv = [row["conventional_host_path_ms"] for row in records
                if row["conventional_host_path_ms"] is not None]
        ai = [row["execution_ms"] for row in ran["background_records"]]
        nrx_tail = [row for row in records
                    if row["nrx_response_ms"] is not None
                    and row["index"] >= protocol["tail_window_start_release"]]
        gates = {
            "contract": (
                ran["schema"] == "softwall-four-cell-two-endpoint-v1"
                and ran["iterations"] == protocol["iterations_per_arm"]
                and ran["cells"] == protocol["cells"]
                and ran["period_ms"] == protocol["period_ms"]
                and ran["deadline_ms"] == protocol["deadline_ms"]
                and ran["nrx_bound_ms"] == protocol["current_nrx_bound_ms"]
                and ran["conv_bound_ms"] == protocol["current_conv_bound_ms"]
                and ran["ai_budget_ms"] == protocol["ai_budget_ms"]
                and ran["ai_rpc_timeout_ms"] == protocol["ai_rpc_timeout_ms"]
                and ran["ai_deadline_ms"] == protocol["ai_deadline_ms"]
                and ran["gate_threshold"] == protocol["gate_threshold"]
                and ran["inject_correlated_failure_every"]
                    == protocol["inject_correlated_failure_every"]
                and ran["payload_seed"] == arm["payload_seed"]
                and ran["channel_seed_base"] == arm["channel_seed_base"]
                and len(records) == protocol["iterations_per_arm"] * protocol["cells"]
                and all(row["channel_seed"] == arm["channel_seed_base"]
                        + row["index"] * protocol["cells"] + row["cell"]
                        for row in records)
                and ran["runtime_metrics"].get("mandatory_reserved") == len(records)
                and requal["iterations"] == 200 and requal["deadline_misses"] == 0
            ),
            "provenance": (
                len({x["host"] for x in (ran, *workers, background, requal)}) == 1
                and ran["host"] == protocol["node"]
                and all(str(x["slurm_job_id"]) == protocol["job"]
                        for x in (ran, *workers, background, requal))
                and all(40 <= x["visible_sm_count"] <= 44 for x in workers)
                and int(background["mps_active_thread_percentage"]) == 20
            ),
            "current_contract_sample": (
                ran["deadline_misses"] == 0
                and ran["nrx_bound_violations"] == 0
                and ran["conv_bound_violations"] == 0
                and ran["conv_path_bound_violations"] == 0
                and ran["background_budget_violations"] == 0
                and ran["background_release_crossings"] == 0
                and ran["pre_radio_ai_guard_violations"] == 0
                and not ran["background_faults"] and not ran["endpoint_faults"]
                and ran["fallback_calendar_final"]["outstanding"] == 0
                and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
                and all(x["outstanding"] == 0 for x in ran["endpoint_final"].values())
                and ran["background_units"] == background["completed_units"]
                and all(row["commit_kind"] in ("nrx", "conventional")
                        for row in records)
            ),
            "tail_window_observed": len(nrx_tail) >= 100,
        }
        arms.append({
            "name": arm["name"], "prefix": prefix, "gates": gates,
            "all_pass": all(gates.values()),
            "nrx_paths": len(nrx), "nrx_max_ms": max(nrx),
            "nrx_p99_9_ms": percentile(nrx, 0.999),
            "nrx_after_tail_window": len(nrx_tail),
            "nrx_tail_window_max_ms": max(row["nrx_response_ms"] for row in nrx_tail),
            "nrx_above_25_ms": sum(x > 25 for x in nrx),
            "nrx_above_30_ms": sum(x > 30 for x in nrx),
            "worst_nrx_rows": [{"release_index": row["index"],
                                "cell": row["cell"],
                                "response_ms": row["nrx_response_ms"]}
                               for row in sorted(
                                   (row for row in records if row["nrx_response_ms"] is not None),
                                   key=lambda row: row["nrx_response_ms"], reverse=True)[:5]],
            "conv_paths": len(conv), "conv_max_ms": max(conv),
            "conv_p99_9_ms": percentile(conv, 0.999),
            "conv_above_8_ms": sum(x > 8 for x in conv),
            "ai_units": len(ai), "ai_host_max_ms": max(ai),
            "joint_ai_units": ran["ai_before_recovery_units"],
            "deadline_misses": ran["deadline_misses"],
        })
    nrx_values = [x for arm in arms for x in (arm["nrx_max_ms"],)]
    conv_values = [x for arm in arms for x in (arm["conv_max_ms"],)]
    candidate_gates = {
        "warm_nrx_25ms_observed": all(arm["nrx_above_25_ms"] == 0 for arm in arms),
        "warm_nrx_30ms_observed": all(arm["nrx_above_30_ms"] == 0 for arm in arms),
        "warm_conv_8ms_observed": all(arm["conv_above_8_ms"] == 0 for arm in arms),
    }
    report = {
        "schema": "softwall-confirm82-persistent-bound-campaign-v1",
        "job": protocol["job"], "source_hashes": source_ok,
        "arms": arms, "candidate_gates": candidate_gates,
        "all_current_contract_gates_pass": source_ok and all(arm["all_pass"] for arm in arms),
        "combined_nrx_max_ms": max(nrx_values),
        "combined_conv_max_ms": max(conv_values),
        "interpretation": "Candidate thresholds are observational tests of the persistent prewarmed four-cell co-run mode only. Passing cannot establish WCET or permit a smaller bound in cold/lifecycle modes. Confirm40 and Confirm43/44 older-mode tails remain independent counterevidence.",
    }
    output = base / f"confirm82_bound_campaign_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_current_contract_gates_pass": report["all_current_contract_gates_pass"],
                      "candidate_gates": candidate_gates,
                      "arm_nrx_max_ms": [arm["nrx_max_ms"] for arm in arms],
                      "arm_conv_max_ms": [arm["conv_max_ms"] for arm in arms]}, indent=2))
    if not report["all_current_contract_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
