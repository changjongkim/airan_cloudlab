#!/usr/bin/env python3.11
"""Frozen audit of real Qwen b1c16 prefill with four-cell MPS radio."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from collections import defaultdict
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm97_qwen_fourcell_protocol.json")
    prior = read(base / protocol["physical_predecessor"])
    source_ok = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms = []
    for arm in protocol["arms"]:
        prefix = arm["prefix"]
        raw = base / "raw"
        ran = read(raw / f"{prefix}_controller.json")
        workers = [read(raw / f"{prefix}_worker{i}.json") for i in range(2)]
        background = read(raw / f"{prefix}_background.json")
        requal = read(raw / f"{prefix}_requalification.json")
        records = ran["records"]
        groups = [records[4*i:4*i+4] for i in range(protocol["iterations"])]
        ai_by_release = defaultdict(list)
        for item in ran["background_records"]:
            ai_by_release[item["release_index"]].append(item)
        two_admitted = sum(sum(row["admitted"] for row in group) == 2
                           for group in groups)
        all_fail = [i for i, group in enumerate(groups)
                    if sum(row["admitted"] for row in group) == 2
                    and all(row["forced_nrx_failure"] for row in group
                            if row["admitted"])
                    and all(row["commit_kind"] == "conventional"
                            for row in group)]
        ai_d153 = [sum(item["returned_ns"] <= group[0]["release_ns"]
                             + protocol["ai_deadline_ms"] * 1_000_000
                         for item in ai_by_release[i])
                   for i, group in enumerate(groups)]
        nrx = [row["nrx_response_ms"] for row in records
               if row["nrx_response_ms"] is not None]
        conv = [row["conventional_host_path_ms"] for row in records
                if row["conventional_host_path_ms"] is not None]
        ai = ran["background_records"]
        joint = [row for row in ai
                 if row.get("phase") == "after_nrx_before_recovery"]
        ai_host = [row["execution_ms"] for row in ai]
        pressure = [i for i in all_fail if ai_d153[i] < 7]
        gates = {
            "contract": (
                ran["schema"] == "softwall-four-cell-gc-intervention-v1"
                and ran["gc_mode"] == "off"
                and ran["iterations"] == protocol["iterations"]
                and ran["cells"] == 4
                and ran["period_ms"] == 180
                and ran["deadline_ms"] == 155
                and ran["nrx_bound_ms"] == 30
                and ran["conv_bound_ms"] == 12
                and ran["ai_budget_ms"] == 40
                and ran["ai_deadline_ms"] == 153
                and ran["payload_seed"] == arm["payload_seed"]
                and ran["channel_seed_base"] == arm["channel_seed_base"]
                and len(records) == 4 * protocol["iterations"]
                and len(ran["recovery_decisions"]) == protocol["iterations"]
                and requal["iterations"] == 200
                and requal["deadline_misses"] == 0
                and background["schema"] == "softwall-qwen-unit-v1"
                and background["model"] == "Qwen/Qwen2.5-1.5B"
                and background["phase"] == "prefill"
                and background["context_length"] == 16
                and background["batch_size"] == 1
                and str(background["mps_active_thread_percentage"]) == "20"
            ),
            "provenance": (
                len({x["host"] for x in (ran, *workers, background, requal)}) == 1
                and ran["host"] == protocol["node"]
                and all(str(x["slurm_job_id"]) == protocol["job"]
                        for x in (ran, *workers, background, requal))
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
                and not ran["gc_events"]
                and ran["fallback_calendar_final"]["outstanding"] == 0
                and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
                and all(x["outstanding"] == 0
                        for x in ran["endpoint_final"].values())
                and ran["background_units"] == background["completed_units"]
                and max(nrx) <= 30 and max(conv) <= 12
                and max(ai_host) <= 40
                and all(row["commit_kind"] in ("nrx", "conventional")
                        for row in records)
            ),
            "coverage": (
                two_admitted >= protocol["min_two_admitted"]
                and len(all_fail) >= protocol["min_injected_all_fail"]
                and len(ai) >= protocol["min_ai_units"]
                and len(pressure) >= protocol["min_observed_pressure_releases"]
                and len(joint) == ran["joint_lease_retired_count"]
                and all(row["returned_ns"] + protocol["ai_guard_ns"]
                        <= row["earliest_fallback_start_ns"] for row in joint)
            ),
        }
        arms.append({
            "name": arm["name"], "prefix": prefix,
            "gates": gates, "all_pass": all(gates.values()),
            "nrx_paths": len(nrx), "nrx_max_ms": max(nrx),
            "conv_paths": len(conv), "conv_max_ms": max(conv),
            "ai_units": len(ai), "ai_host_max_ms": max(ai_host),
            "two_admitted_releases": two_admitted,
            "injected_all_fail_releases": len(all_fail),
            "all_fail_releases_with_fewer_than_seven_ai_by_d153": len(pressure),
            "min_ai_completed_by_d153_in_all_fail": min(ai_d153[i] for i in all_fail)
                if all_fail else None,
            "median_ai_completed_by_d153_in_all_fail": statistics.median(
                ai_d153[i] for i in all_fail) if all_fail else None,
            "max_ai_completed_by_d153_in_all_fail": max(ai_d153[i] for i in all_fail)
                if all_fail else None,
            "joint_ai_units": len(joint),
            "deadline_misses": ran["deadline_misses"],
        })
    report = {
        "schema": "softwall-confirm97-qwen-fourcell-canary-v1",
        "job": protocol["job"], "node": protocol["node"],
        "source_hashes": source_ok,
        "prior_mode_pass": prior["all_pass"],
        "arms": arms,
        "all_pass": source_ok and prior["all_pass"] and all(a["all_pass"] for a in arms),
        "interpretation": "Canary of real Qwen2.5-1.5B b1c16 prefill under four-cell MPS radio. Observed fewer-than-seven Qwen returns by D153 in all-fail releases measures pressure under this controller and continuous backlog; it is not a proof that no other safe schedule could finish seven. Radio/AI bound pass is finite prewarmed GC-OFF sample, not WCET, online seven-job lease, production DU, or policy superiority. The cap20 Qwen co-run is a new mode and does not inherit prior AI8 qualification.",
    }
    output = base / f"confirm97_qwen_fourcell_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "all_pass": report["all_pass"],
        "gates": [a["gates"] for a in arms],
        "nrx_max_ms": [a["nrx_max_ms"] for a in arms],
        "conv_max_ms": [a["conv_max_ms"] for a in arms],
        "ai_host_max_ms": [a["ai_host_max_ms"] for a in arms],
        "all_fail_min_ai_d153": [a["min_ai_completed_by_d153_in_all_fail"] for a in arms],
        "observed_pressure_releases": [
            a["all_fail_releases_with_fewer_than_seven_ai_by_d153"] for a in arms],
    }, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
