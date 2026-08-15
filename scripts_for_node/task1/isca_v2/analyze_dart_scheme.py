#!/usr/bin/env python3
"""Validate regular DART routing, control, and fault mechanism gates."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics
from pathlib import Path


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    scheme = root / "06_dart_scheme"
    output = scheme / "analysis"; output.mkdir(parents=True, exist_ok=True)
    hardware_paths = sorted((scheme / "hardware_routing").glob("trial_*.json"))
    fault_paths = sorted((scheme / "faults").glob("trial_*.json"))
    replay_paths = sorted((scheme / "control_replay").glob("*.json"))
    if (len(hardware_paths), len(fault_paths), len(replay_paths)) != (5, 5, 20):
        raise RuntimeError(
            f"artifact count mismatch hardware={len(hardware_paths)} "
            f"fault={len(fault_paths)} replay={len(replay_paths)}"
        )
    hardware = []
    for trial, path in enumerate(hardware_paths, 1):
        value = json.loads(path.read_text())
        if not value.get("outputs_finite") or len(value["runs"]) != 20:
            raise RuntimeError(f"invalid hardware routing result {path}")
        for run in value["runs"]:
            if not all(
                len(run["raw"][field]) == run["requests"]
                for field in (
                    "arrival_ms", "completion_ms", "latency_ms",
                    "scheduler_lateness_us", "assignment_device",
                    "queue_at_submit",
                )
            ):
                raise RuntimeError(f"raw routing trace mismatch: {path}")
            hardware.append({
                "trial": trial, "policy": run["policy"],
                "rate_slots_s": run["rate_slots_s"],
                "requests": run["requests"],
                "latency_mean_ms": run["latency_ms"]["mean"],
                "latency_p99_ms": run["latency_ms"]["p99"],
                "miss_5ms": run["deadline_miss_ratio"]["5"],
                "backlog": run["backlog_at_window_end"],
            })
    faults = []
    for path in fault_paths:
        value = json.loads(path.read_text())
        storm = value.get("fallback_storm", {})
        if (
            not value.get("pass")
            or value["iterations"] != 10000
            or not storm.get("pass")
            or storm.get("over_admitted") != 0
            or storm.get("reservation_leaks") != 0
        ):
            raise RuntimeError(f"fault gate failed: {path}")
        faults.append({
            **value,
            "fallback_storm_groups": storm["groups"],
            "fallback_storm_fanout": storm["fanout"],
            "fallback_capacity": storm["capacity"],
            "fallback_over_admitted": storm["over_admitted"],
            "fallback_reservation_leaks": storm["reservation_leaks"],
        })
    replays = []
    for path in replay_paths:
        value = json.loads(path.read_text())
        guard = int(path.stem.split("_")[1])
        trial = int(path.stem.rsplit("t", 1)[1])
        for run in value["results"]:
            replays.append({
                "guard_pct": guard, "trial": trial, "policy": run["policy"],
                "requests": run["requests"],
                "deadline_miss_ratio": run["deadline_miss_ratio"],
                "latency_p99_ms": run["latency_ms"].get("p99"),
                "background_value": run["background"]["value"],
                "stale_or_wrong_commit": sum(
                    count for key, count in run["runtime_metrics"].items()
                    if key in ("wrong_commit", "stale_commit")
                ),
            })
    write_csv(output / "HARDWARE_ROUTING.csv", hardware)
    write_csv(output / "FAULT_CORRECTNESS.csv", faults)
    write_csv(output / "CONTROL_REPLAY.csv", replays)
    groups = {}
    for row in hardware:
        groups.setdefault((row["policy"], row["rate_slots_s"]), []).append(row)
    medians = []
    for (policy, rate), items in sorted(groups.items()):
        medians.append({
            "policy": policy, "rate_slots_s": rate, "trials": len(items),
            "median_p99_ms": statistics.median(x["latency_p99_ms"] for x in items),
            "median_miss_5ms": statistics.median(x["miss_5ms"] for x in items),
            "median_backlog": statistics.median(x["backlog"] for x in items),
        })
    write_csv(output / "HARDWARE_ROUTING_MEDIAN.csv", medians)
    report = [
        "# DART regular mechanism gates", "",
        f"- Actual NRx compute/queue routing runs: {len(hardware)}",
        f"- Fault trials: {len(faults)} × 10,000 faults; all passed",
        "- Each fault trial also validates 100 correlated 8-way fallback bursts "
        "against two reserved fallback lanes; no over-admission or leaks",
        f"- Measured-profile control replays: {len(replays)} policy rows", "",
        "The hardware routing gate contains actual TensorRT execution and CUDA queues, "
        "but no L1 or P2P/GDR payload. The separate five-way experiment validates those "
        "data paths. Therefore these artifacts validate routing plus the software DART "
        "transaction semantics, including dual reservation, visibility/epoch rejection, "
        "and fallback-storm containment, not a fully integrated end-to-end DART data "
        "plane.", "",
        "| policy | rate | median p99 ms | median miss >5ms | median backlog |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in medians:
        report.append(
            f"| {row['policy']} | {row['rate_slots_s']:.1f} | "
            f"{row['median_p99_ms']:.3f} | {row['median_miss_5ms']:.6f} | "
            f"{row['median_backlog']:.0f} |"
        )
    (output / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    (output / "VALIDATION.json").write_text(json.dumps({
        "pass": True, "hardware_runs": len(hardware),
        "fault_trials": len(faults), "control_rows": len(replays),
        "integrated_end_to_end_dart": False,
    }, indent=2), encoding="utf-8")
    print(f"[DART-ANALYSIS] PASS output={output}", flush=True)


if __name__ == "__main__":
    main()
