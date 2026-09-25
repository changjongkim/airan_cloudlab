#!/usr/bin/env python3.11
"""Analyze controlled one-versus-two NeuralRx pre-observation Qwen latency."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def quantile(values: list[float], q: float) -> float:
    return sorted(values)[round((len(values) - 1) * q)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm103_nrx_count_abba_protocol.json")
    source_ok = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                    for name, digest in protocol["source_sha256"].items())
    if not source_ok:
        raise RuntimeError("frozen source changed")
    arms = []
    raw_by_name = {}
    for spec in protocol["arms"]:
        prefix = f"confirm103_{spec['name']}_job{protocol['job']}"
        controller = read(base / "raw" / f"{prefix}_controller.json")
        background = read(base / "raw" / f"{prefix}_background.json")
        raw_by_name[spec["name"]] = (controller, background)
        admitted = {}
        seeds = {}
        for row in controller["records"]:
            admitted.setdefault(row["index"], 0)
            admitted[row["index"]] += int(row["admitted"])
            seeds.setdefault(row["index"], []).append(row["channel_seed"])
        pre = {row["release_index"]: row for row in controller["background_records"]
               if row.get("phase") == "before_nrx_observation"}
        values = [row["execution_ms"] for row in pre.values()]
        exact_count = (len(admitted) == protocol["iterations"]
                       and all(value == spec["nrx_count"] for value in admitted.values()))
        safety = not any((controller["deadline_misses"],
                          controller["nrx_bound_violations"],
                          controller["conv_path_bound_violations"],
                          controller["background_budget_violations"],
                          controller["pre_radio_ai_guard_violations"],
                          controller["endpoint_faults"],
                          controller["background_faults"]))
        arms.append({
            "name": spec["name"], "nrx_count": spec["nrx_count"],
            "host": controller["host"], "job": controller["slurm_job_id"],
            "exact_nrx_count": exact_count,
            "pre_ai_count": len(values),
            "pre_ai_execution_median_ms": statistics.median(values),
            "pre_ai_execution_p95_ms": quantile(values, 0.95),
            "pre_ai_execution_max_ms": max(values),
            "safety": safety,
            "channel_seeds_by_release": seeds,
        })
    pairs = []
    for one_name, two_name in (("a_one", "b_two"), ("d_one", "c_two")):
        one_c, _ = raw_by_name[one_name]
        two_c, _ = raw_by_name[two_name]
        one_pre = {row["release_index"]: row["execution_ms"]
                   for row in one_c["background_records"]
                   if row.get("phase") == "before_nrx_observation"}
        two_pre = {row["release_index"]: row["execution_ms"]
                   for row in two_c["background_records"]
                   if row.get("phase") == "before_nrx_observation"}
        indices = sorted(set(one_pre) & set(two_pre))
        diffs = [two_pre[i] - one_pre[i] for i in indices]
        one_seeds = [[row["channel_seed"] for row in one_c["records"]
                      if row["index"] == i] for i in indices]
        two_seeds = [[row["channel_seed"] for row in two_c["records"]
                      if row["index"] == i] for i in indices]
        pairs.append({
            "one": one_name, "two": two_name,
            "matched_phy": one_seeds == two_seeds,
            "paired_releases": len(indices),
            "two_minus_one_mean_ms": statistics.mean(diffs),
            "two_minus_one_median_ms": statistics.median(diffs),
            "two_slower_fraction": sum(value > 0 for value in diffs) / len(diffs),
        })
    result = {
        "schema": "softwall-confirm103-nrx-count-abba-v1",
        "source_hashes": source_ok, "arms": arms, "pairs": pairs,
        "all_safety": all(row["safety"] and row["exact_nrx_count"]
                          and row["pre_ai_count"] == protocol["iterations"]
                          for row in arms),
        "both_directions_positive": all(
            row["matched_phy"] and row["two_minus_one_mean_ms"] > 0
            for row in pairs
        ),
        "interpretation": (
            "Controlled same-PHY ABBA probe of one versus two concurrent "
            "NeuralRx requests and the pre-observation Qwen unit. A positive "
            "latency difference is a physical count-dependent MPS cost, not "
            "a policy throughput result, AI-RAN application, or WCET."
        ),
    }
    output = base / f"confirm103_nrx_count_abba_job{protocol['job']}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
