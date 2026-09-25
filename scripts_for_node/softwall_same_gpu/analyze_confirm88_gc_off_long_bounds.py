#!/usr/bin/env python3.11
"""Frozen long-run service-bound screen with cyclic GC disabled during traffic."""

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
    protocol = read(base / "confirm88_gc_off_long_bounds_protocol.json")
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
        nrx = [r["nrx_response_ms"] for r in records
               if r["nrx_response_ms"] is not None]
        conv = [r["conventional_host_path_ms"] for r in records
                if r["conventional_host_path_ms"] is not None]
        late_window = [r for r in records if r["index"] >= 2500
                       and r["nrx_response_ms"] is not None]
        gates = {
            "contract": (
                ran["schema"] == "softwall-four-cell-gc-intervention-v1"
                and ran["gc_mode"] == "off"
                and ran["iterations"] == protocol["iterations"]
                and ran["cells"] == 4
                and ran["period_ms"] == 180
                and ran["deadline_ms"] == 155
                and ran["nrx_bound_ms"] == 50
                and ran["conv_bound_ms"] == 25
                and ran["ai_budget_ms"] == 15
                and ran["payload_seed"] == arm["payload_seed"]
                and ran["channel_seed_base"] == arm["channel_seed_base"]
                and len(records) == 4 * protocol["iterations"]
                and len(ran["release_stage_records"]) == protocol["iterations"]
                and all(r["channel_seed"] == arm["channel_seed_base"]
                        + r["index"] * 4 + r["cell"] for r in records)
                and requal["iterations"] == 200
                and requal["deadline_misses"] == 0
            ),
            "provenance": (
                len({x["host"] for x in (ran, *workers, background, requal)}) == 1
                and ran["host"] == protocol["node"]
                and all(str(x["slurm_job_id"]) == protocol["job"]
                        for x in (ran, *workers, background, requal))
            ),
            "current_contract_sample": (
                ran["deadline_misses"] == 0
                and ran["nrx_bound_violations"] == 0
                and ran["conv_path_bound_violations"] == 0
                and ran["background_budget_violations"] == 0
                and ran["background_release_crossings"] == 0
                and ran["pre_radio_ai_guard_violations"] == 0
                and not ran["background_faults"]
                and not ran["endpoint_faults"]
                and ran["fallback_calendar_final"]["outstanding"] == 0
                and ran["fallback_calendar_final"]["joint_leases_outstanding"] == 0
                and ran["background_units"] == background["completed_units"]
                and all(r["commit_kind"] in ("nrx", "conventional") for r in records)
            ),
            "gc_disabled_and_tail_window": (
                not ran["gc_events"] and len(late_window) >= 100
            ),
        }
        arms.append({
            "name": arm["name"], "prefix": prefix,
            "gates": gates, "all_pass": all(gates.values()),
            "nrx_paths": len(nrx), "nrx_max_ms": max(nrx),
            "nrx_p99_9_ms": percentile(nrx, 0.999),
            "nrx_over_25_ms": sum(x > 25 for x in nrx),
            "nrx_over_30_ms": sum(x > 30 for x in nrx),
            "tail_window_nrx": len(late_window),
            "tail_window_nrx_max_ms": max(r["nrx_response_ms"] for r in late_window),
            "conv_paths": len(conv), "conv_max_ms": max(conv),
            "conv_p99_9_ms": percentile(conv, 0.999),
            "conv_over_8_ms": sum(x > 8 for x in conv),
            "conv_over_12_ms": sum(x > 12 for x in conv),
            "background_units": ran["background_units"],
            "joint_ai_units": ran["ai_before_recovery_units"],
            "deadline_misses": ran["deadline_misses"],
        })
    candidates = {
        "nrx25_observed": all(a["nrx_over_25_ms"] == 0 for a in arms),
        "nrx30_observed": all(a["nrx_over_30_ms"] == 0 for a in arms),
        "conv8_observed": all(a["conv_over_8_ms"] == 0 for a in arms),
        "conv12_observed": all(a["conv_over_12_ms"] == 0 for a in arms),
    }
    report = {
        "schema": "softwall-confirm88-gc-off-long-bound-screen-v1",
        "job": protocol["job"], "source_hashes": source_ok,
        "arms": arms, "candidate_gates": candidates,
        "all_current_contract_gates_pass": source_ok and all(a["all_pass"] for a in arms),
        "interpretation": "Two independent 5,000-release prewarmed four-cell runs with Python cyclic GC disabled only during timed traffic. Candidate thresholds are observational; an actual tighter calendar, cold/restart modes, independent nodes, other interference and hard WCET remain unqualified. Confirm43/44 old-mode GPU tails are not erased by this test.",
    }
    output = base / f"confirm88_gc_off_long_bounds_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_current_contract_gates_pass":
                      report["all_current_contract_gates_pass"],
                      "candidate_gates": candidates,
                      "nrx_max_ms": [a["nrx_max_ms"] for a in arms],
                      "conv_max_ms": [a["conv_max_ms"] for a in arms]}, indent=2))
    if not report["all_current_contract_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
