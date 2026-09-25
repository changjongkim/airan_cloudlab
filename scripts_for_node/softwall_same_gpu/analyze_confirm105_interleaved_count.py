#!/usr/bin/env python3.11
"""Analyze within-process matched 1/2 NeuralRx count pairs."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text())


def bootstrap(values: list[float], seed: int, resamples: int) -> list[float]:
    rng = random.Random(seed)
    means = sorted(statistics.mean(rng.choices(values, k=len(values)))
                   for _ in range(resamples))
    return [means[round(0.025 * (len(means) - 1))],
            means[round(0.975 * (len(means) - 1))]]


def summarize(values: list[float], seed: int, resamples: int) -> dict:
    return {
        "n": len(values), "mean": statistics.mean(values),
        "median": statistics.median(values),
        "positive_fraction": sum(value > 0 for value in values) / len(values),
        "paired_bootstrap_mean_95": bootstrap(values, seed, resamples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm105_interleaved_count_protocol.json")
    source_ok = all(hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
                    for name, digest in protocol["source_sha256"].items())
    if not source_ok:
        raise RuntimeError("frozen source changed")
    runs = []
    all_strata = []
    for run_index, spec in enumerate(protocol["runs"]):
        prefix = f"confirm105_{spec['name']}_job{protocol['job']}"
        controller = read(base / "raw" / f"{prefix}_controller.json")
        by_release = {}
        for row in controller["records"]:
            entry = by_release.setdefault(row["index"], {
                "seeds": [], "admitted": 0, "limit": row["nrx_admission_limit"],
                "release_ns": row["release_ns"], "correct": 0,
            })
            entry["seeds"].append(row["channel_seed"])
            entry["admitted"] += int(row["admitted"])
            entry["correct"] += int(row["correct"])
        pre = {row["release_index"]: row["execution_ms"]
               for row in controller["background_records"]
               if row.get("phase") == "before_nrx_observation"}
        timely = {i: 0 for i in by_release}
        for row in controller["background_records"]:
            i = row["release_index"]
            if row["returned_ns"] <= by_release[i]["release_ns"] + 153_000_000:
                timely[i] += 1
        strata = {"one_two": {"latency": [], "ai": [], "correct": []},
                  "two_one": {"latency": [], "ai": [], "correct": []}}
        matched = True
        exact = True
        for first in range(0, protocol["iterations"], 2):
            second = first + 1
            a, b = by_release[first], by_release[second]
            matched &= a["seeds"] == b["seeds"]
            exact &= a["admitted"] == a["limit"] and b["admitted"] == b["limit"]
            if (a["limit"], b["limit"]) == (1, 2):
                one, two, order = first, second, "one_two"
            elif (a["limit"], b["limit"]) == (2, 1):
                one, two, order = second, first, "two_one"
            else:
                raise AssertionError("invalid matched count pair")
            strata[order]["latency"].append(pre[two] - pre[one])
            strata[order]["ai"].append(timely[two] - timely[one])
            strata[order]["correct"].append(
                by_release[two]["correct"] - by_release[one]["correct"])
        safety = not any((controller["deadline_misses"],
                          controller["nrx_bound_violations"],
                          controller["conv_path_bound_violations"],
                          controller["background_budget_violations"],
                          controller["pre_radio_ai_guard_violations"],
                          controller["endpoint_faults"],
                          controller["background_faults"]))
        summaries = {}
        for order_index, (order, values) in enumerate(strata.items()):
            seed = protocol["bootstrap_seed"] + 10 * run_index + order_index
            summaries[order] = {
                "two_minus_one_pre_ai_ms": summarize(
                    values["latency"], seed, protocol["bootstrap_resamples"]),
                "two_minus_one_timely_ai": summarize(
                    values["ai"], seed + 100, protocol["bootstrap_resamples"]),
                "two_minus_one_correct": summarize(
                    values["correct"], seed + 200, protocol["bootstrap_resamples"]),
            }
            all_strata.append(summaries[order]["two_minus_one_pre_ai_ms"])
        runs.append({
            "name": spec["name"], "host": controller["host"],
            "job": controller["slurm_job_id"], "matched_phy": matched,
            "exact_count": exact, "pre_ai_count": len(pre),
            "safety": safety, "strata": summaries,
        })
    result = {
        "schema": "softwall-confirm105-interleaved-count-v1",
        "source_hashes": source_ok, "runs": runs,
        "all_safety_and_pairing": all(
            row["safety"] and row["matched_phy"] and row["exact_count"]
            and row["pre_ai_count"] == protocol["iterations"] for row in runs
        ),
        "all_four_latency_ci_lower_positive": all(
            row["paired_bootstrap_mean_95"][0] > 0 for row in all_strata
        ),
        "all_pass": False,
        "interpretation": (
            "Within one persistent MPS/Qwen process, each adjacent release "
            "pair replays identical PHY with 1/2 or 2/1 NeuralRx counts. "
            "This controls between-run state drift but remains a Qwen canary, "
            "not a policy or AI-RAN application result."
        ),
    }
    result["all_pass"] = (result["all_safety_and_pairing"]
                          and result["all_four_latency_ci_lower_positive"])
    output = base / f"confirm105_interleaved_count_job{protocol['job']}.json"
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
