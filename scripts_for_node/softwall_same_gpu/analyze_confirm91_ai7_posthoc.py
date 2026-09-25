#!/usr/bin/env python3.11
"""Posthoc physical seven-AI capacity check on frozen Confirm91 raw traces."""

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
    base = args.root.resolve() / "results/softwall_same_gpu"
    prior = read(base / "confirm91_conv12_nrx30_gc_off_job58743005.json")
    if not prior["all_pass"]:
        raise RuntimeError("physical predecessor failed")
    arms = []
    for name in ("a", "b"):
        path = base / "raw" / f"confirm91_{name}_nrx30_job58743005_controller.json"
        ran = read(path)
        if ran["iterations"] != 3000 or ran["ai_budget_ms"] != 15:
            raise RuntimeError("unexpected predecessor contract")
        by_release = defaultdict(list)
        for item in ran["background_records"]:
            by_release[item["release_index"]].append(item)
        groups = [ran["records"][4*i:4*i+4] for i in range(3000)]
        all_fail = [i for i, group in enumerate(groups)
                    if sum(row["admitted"] for row in group) == 2
                    and all(row["forced_nrx_failure"] for row in group
                            if row["admitted"])
                    and all(row["commit_kind"] == "conventional"
                            for row in group)]
        ai_d153 = [sum(item["returned_ns"] <= group[0]["release_ns"]
                             + 153_000_000 for item in by_release[i])
                   for i, group in enumerate(groups)]
        host = [item["execution_ms"] for item in ran["background_records"]]
        arms.append({
            "name": name, "raw_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ai_units": len(host), "ai_host_max_ms": max(host),
            "ai_host_above_8ms": sum(value > 8 for value in host),
            "all_fail_releases": len(all_fail),
            "all_fail_with_at_least_seven_ai_by_d153":
                sum(ai_d153[i] >= 7 for i in all_fail),
            "all_fail_min_ai_by_d153": min(ai_d153[i] for i in all_fail),
            "all_fail_median_ai_by_d153": statistics.median(ai_d153[i] for i in all_fail),
            "all_fail_max_ai_by_d153": max(ai_d153[i] for i in all_fail),
        })
    report = {
        "schema": "softwall-confirm91-ai7-physical-posthoc-v1",
        "posthoc": True,
        "predecessor_sha256": hashlib.sha256((base / "confirm91_conv12_nrx30_gc_off_job58743005.json").read_bytes()).hexdigest(),
        "arms": arms,
        "arithmetic_if_ai8_qualifies_ms": 30 + 4 * 12 + 7 * 8 + 2,
        "deadline_ms": 155,
        "interpretation": "Descriptive raw-trace reanalysis selected after Confirm93's 15-ms abstract seven-unit pressure. All AI host durations were measured under an actual AI15 calendar; sub-8-ms samples do not qualify a new AI8 calendar. AI returned by D153 includes after-radio work without seven atomic pre-recovery leases. The observed physical workload is far shorter than the abstract 15-ms unit, so Confirm93 contention must not be treated as physically established.",
    }
    output = base / "confirm91_ai7_physical_posthoc.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"arms": arms, "arithmetic_if_ai8_qualifies_ms": 136}, indent=2))


if __name__ == "__main__":
    main()
