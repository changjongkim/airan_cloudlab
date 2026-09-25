#!/usr/bin/env python3
"""Post-hoc decomposition of Confirm43 gap AI throughput and timing tails."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def interval(counts: list[int], block_size: int) -> dict:
    blocks = [sum(counts[i:i + block_size]) for i in range(0, len(counts), block_size)]
    mean = statistics.mean(blocks)
    stderr = statistics.stdev(blocks) / math.sqrt(len(blocks))
    return {
        "block_size_releases": block_size,
        "blocks": len(blocks),
        "mean_extra_ai_per_block": mean,
        "normal_95_interval_extra_ai_per_block": [
            mean - 1.96 * stderr, mean + 1.96 * stderr,
        ],
    }


def count_by_release(data: dict) -> list[int]:
    counts = [0] * data["iterations"]
    for unit in data["background_records"]:
        counts[unit["release_index"]] += 1
    if sum(counts) != data["background_units"]:
        raise ValueError("background unit count does not match controller records")
    return counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    conditions = {}
    names = ("r1_off", "r2_on", "r3_on", "r4_off")
    for name in names:
        path = args.raw / f"confirm43_gap_{name}_cap80_job{args.job}_controller.json"
        data = json.loads(path.read_text(encoding="utf-8"))
        if data["slurm_job_id"] != args.job or data["iterations"] != 10000:
            raise ValueError(f"wrong campaign identity: {path}")
        conditions[name] = data
    counts = {name: count_by_release(data) for name, data in conditions.items()}
    pairs = []
    all_differences = []
    for off, on in (("r1_off", "r2_on"), ("r4_off", "r3_on")):
        differences = [a - b for a, b in zip(counts[on], counts[off])]
        all_differences.extend(differences)
        pairs.append({
            "off": off,
            "on": on,
            "off_ai_units": conditions[off]["background_units"],
            "on_ai_units": conditions[on]["background_units"],
            "net_extra_ai_units": sum(differences),
            "gap_ai_units": conditions[on]["recovery_gap_ai_units"],
            "post_ran_ai_difference": (
                conditions[on]["background_units"]
                - conditions[on]["recovery_gap_ai_units"]
                - conditions[off]["background_units"]
            ),
            "exploratory_100_release_block_interval": interval(differences, 100),
        })
    violations = []
    for name, data in conditions.items():
        for row in data["records"]:
            if row["nrx_bound_violation"]:
                violations.append({
                    "condition": name,
                    "release_index": row["index"],
                    "channel_seed": row["channel_seed"],
                    "post_gpu_ms": row["post_gpu_ms"],
                    "response_ms": row["response_ms"],
                    "deadline_miss": row["deadline_miss"],
                    "fallback": row["fallback"],
                })
    report = {
        "schema": "softwall-confirm43-posthoc-mechanism-diagnostic-v1",
        "job": args.job,
        "status": "exploratory; does not change frozen Confirm43 gates",
        "pairs": pairs,
        "total_net_extra_ai_units": sum(p["net_extra_ai_units"] for p in pairs),
        "total_gap_ai_units": sum(p["gap_ai_units"] for p in pairs),
        "total_post_ran_ai_difference": sum(p["post_ran_ai_difference"] for p in pairs),
        "exploratory_combined_100_release_block_interval": interval(all_differences, 100),
        "nrx_bound_violations": violations,
    }
    lines = [
        "# Confirm43 post-hoc mechanism diagnostic", "",
        "This exploratory decomposition does not revise or replace the failed frozen Confirm43 gate.", "",
        "| matched pair | gap AI | other AI change | net completed AI change | exploratory 100-release block 95% interval (AI/block) |",
        "|---|---:|---:|---:|---|",
    ]
    for pair in pairs:
        ci = pair["exploratory_100_release_block_interval"]["normal_95_interval_extra_ai_per_block"]
        lines.append(
            f"| {pair['on']} − {pair['off']} | {pair['gap_ai_units']} | "
            f"{pair['post_ran_ai_difference']:+d} | {pair['net_extra_ai_units']:+d} | "
            f"[{ci[0]:+.3f}, {ci[1]:+.3f}] |"
        )
    ci = report["exploratory_combined_100_release_block_interval"]["normal_95_interval_extra_ai_per_block"]
    lines.extend([
        "", f"Combined net AI: {report['total_net_extra_ai_units']:+d}; "
        f"gap AI {report['total_gap_ai_units']}, other AI "
        f"{report['total_post_ran_ai_difference']:+d}.",
        f"Exploratory combined block interval: [{ci[0]:+.3f}, {ci[1]:+.3f}] AI per 100 releases.",
        f"NRx 30 ms bound violations: {len(violations)}; all had deadline miss = "
        f"{[v['deadline_miss'] for v in violations]}.",
    ])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
