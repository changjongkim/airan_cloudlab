#!/usr/bin/env python3.11
"""Analyze the long controlled one-versus-two NeuralRx Qwen ABBA."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[round((len(values) - 1) * q)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm104_nrx_count_abba_long_protocol.json")
    source_ok = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                    for name, digest in protocol["source_sha256"].items())
    if not source_ok:
        raise RuntimeError("frozen source changed")
    raw = {}
    arms = []
    for spec in protocol["arms"]:
        prefix = f"confirm104_{spec['name']}_job{protocol['job']}"
        controller = read(base / "raw" / f"{prefix}_controller.json")
        raw[spec["name"]] = controller
        admitted = {}
        for row in controller["records"]:
            admitted.setdefault(row["index"], 0)
            admitted[row["index"]] += int(row["admitted"])
        pre = [row["execution_ms"] for row in controller["background_records"]
               if row.get("phase") == "before_nrx_observation"]
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
            "exact_nrx_count": (len(admitted) == protocol["iterations"]
                                and all(value == spec["nrx_count"]
                                        for value in admitted.values())),
            "pre_ai_count": len(pre),
            "pre_ai_execution_median_ms": statistics.median(pre),
            "pre_ai_execution_p95_ms": percentile(pre, 0.95),
            "pre_ai_execution_max_ms": max(pre),
            "safety": safety,
        })
    pairs = []
    for pair_index, (one_name, two_name) in enumerate(
            (("a_one", "b_two"), ("d_one", "c_two"))):
        one, two = raw[one_name], raw[two_name]
        one_pre = {row["release_index"]: row["execution_ms"]
                   for row in one["background_records"]
                   if row.get("phase") == "before_nrx_observation"}
        two_pre = {row["release_index"]: row["execution_ms"]
                   for row in two["background_records"]
                   if row.get("phase") == "before_nrx_observation"}
        indices = sorted(set(one_pre) & set(two_pre))
        diffs = [two_pre[i] - one_pre[i] for i in indices]
        one_seeds = [[row["channel_seed"] for row in one["records"]
                      if row["index"] == i] for i in indices]
        two_seeds = [[row["channel_seed"] for row in two["records"]
                      if row["index"] == i] for i in indices]
        rng = random.Random(protocol["bootstrap_seed"] + pair_index)
        means = sorted(statistics.mean(rng.choices(diffs, k=len(diffs)))
                       for _ in range(protocol["bootstrap_resamples"]))
        lower = means[round(0.025 * (len(means) - 1))]
        upper = means[round(0.975 * (len(means) - 1))]
        pairs.append({
            "one": one_name, "two": two_name,
            "matched_phy": one_seeds == two_seeds,
            "paired_releases": len(indices),
            "two_minus_one_mean_ms": statistics.mean(diffs),
            "two_minus_one_median_ms": statistics.median(diffs),
            "two_slower_fraction": sum(value > 0 for value in diffs) / len(diffs),
            "paired_bootstrap_mean_95_ms": [lower, upper],
        })
    result = {
        "schema": "softwall-confirm104-nrx-count-long-abba-v1",
        "source_hashes": source_ok, "arms": arms, "pairs": pairs,
        "all_safety": all(row["safety"] and row["exact_nrx_count"]
                          and row["pre_ai_count"] == protocol["iterations"]
                          for row in arms),
        "both_bootstrap_lower_positive": all(
            row["matched_phy"] and row["paired_releases"] == protocol["iterations"]
            and row["paired_bootstrap_mean_95_ms"][0] > 0 for row in pairs
        ),
        "all_pass": False,
        "interpretation": (
            "Independent long same-PHY ABBA confirmation of the physical "
            "one-to-two NeuralRx count cost on pre-observation Qwen. This "
            "does not establish policy throughput, AI-RAN workload value, "
            "or a WCET."
        ),
    }
    result["all_pass"] = result["all_safety"] and result["both_bootstrap_lower_positive"]
    output = base / f"confirm104_nrx_count_long_abba_job{protocol['job']}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
