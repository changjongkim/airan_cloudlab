#!/usr/bin/env python3
"""Summarize confirmatory SoftWall runs and enforce basic result validity."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import statistics
from pathlib import Path


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] * (upper - position) + ordered[upper] * (position - lower)


def ran_records(document: dict) -> list[dict]:
    return document.get("ran_records", document.get("records", []))


def load_pairs(raw: Path, campaign: str) -> dict[tuple[int, str, str], dict[str, Path]]:
    pattern = re.compile(
        re.escape(campaign)
        + r"_r(?P<round>[0-9]+)_(?P<condition>mps_alone|uncontrolled_(?:gemm|hbm|nrx|qwen)|protected_(?:gemm|hbm|nrx|qwen))"
        + r"_job(?P<job>[0-9]+)(?P<role>_ran|_ai)?[.]json$"
    )
    pairs: dict[tuple[int, str, str], dict[str, Path]] = {}
    for path in raw.glob(f"{campaign}_r*_job*.json"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        key = (int(match["round"]), match["condition"], match["job"])
        role = (match["role"] or "_ran").lstrip("_")
        pairs.setdefault(key, {})[role] = path
    return pairs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--output-prefix", type=Path, required=True)
    parser.add_argument("--deadline-ms", type=float, default=25.0)
    parser.add_argument("--campaign", default="confirm")
    args = parser.parse_args()

    rows = []
    errors = []
    for (round_number, condition, job), paths in sorted(
        load_pairs(args.raw, args.campaign).items()
    ):
        if "ran" not in paths:
            errors.append(f"missing RAN result: round={round_number} condition={condition}")
            continue
        if condition != "mps_alone" and "ai" not in paths:
            errors.append(f"missing AI result: round={round_number} condition={condition}")
            continue
        ran = json.loads(paths["ran"].read_text(encoding="utf-8"))
        records = ran_records(ran)
        if not records:
            errors.append(f"empty RAN records: {paths['ran']}")
            continue
        response = [float(item["response_ms"]) for item in records]
        releases = [int(item["release_ns"]) for item in records]
        window_start = releases[0]
        window_end = releases[-1] + round(float(ran["period_ms"]) * 1e6)
        window_s = (window_end - window_start) / 1e9
        ai_units = 0
        budget_violations = 0
        crossed_release = 0
        ai_rate = 0.0
        if condition.startswith("uncontrolled_"):
            ai = json.loads(paths["ai"].read_text(encoding="utf-8"))
            ai_units = sum(
                window_start <= int(item["completed_ns"]) <= window_end
                for item in ai.get("records", [])
            )
            ai_rate = ai_units / window_s
        elif condition.startswith("protected_"):
            ai_units = int(ran["ai_completed_units"])
            budget_violations = int(ran["ai_budget_violations"])
            crossed_release = int(ran["ai_crossed_next_release"])
            ai_rate = ai_units / window_s
        incorrect_transport_blocks = sum(
            item.get("correct_tb") is False for item in records
        )
        rows.append({
            "round": round_number,
            "job": job,
            "condition": condition,
            "samples": len(response),
            "deadline_ms": args.deadline_ms,
            "ran_mean_ms": statistics.mean(response),
            "ran_p99_ms": percentile(response, 0.99),
            "ran_p999_ms": percentile(response, 0.999),
            "ran_max_ms": max(response),
            "deadline_misses": sum(value > args.deadline_ms for value in response),
            "incorrect_transport_blocks": incorrect_transport_blocks,
            "ai_units_in_window": ai_units,
            "ai_units_per_s": ai_rate,
            "ai_budget_violations": budget_violations,
            "ai_crossed_release": crossed_release,
        })

    if errors:
        raise SystemExit("\n".join(errors))
    if not rows:
        raise SystemExit("no confirmatory results found")

    args.output_prefix.parent.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_prefix.with_suffix(".csv")
    json_path = args.output_prefix.with_suffix(".json")
    md_path = args.output_prefix.with_suffix(".md")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    lines = [
        "# SoftWall confirmatory campaign",
        "",
        "| condition | runs | RAN samples | misses | incorrect TB | run p99 range (ms) | worst max (ms) | AI units/s mean | budget violations | release crossings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, items in sorted(grouped.items()):
        p99 = [float(item["ran_p99_ms"]) for item in items]
        ai_rates = [float(item["ai_units_per_s"]) for item in items if item["ai_units_per_s"]]
        lines.append(
            f"| {condition} | {len(items)} | {sum(int(x['samples']) for x in items)} | "
            f"{sum(int(x['deadline_misses']) for x in items)} | "
            f"{sum(int(x['incorrect_transport_blocks']) for x in items)} | "
            f"{min(p99):.3f}–{max(p99):.3f} | "
            f"{max(float(x['ran_max_ms']) for x in items):.3f} | "
            f"{statistics.mean(ai_rates) if ai_rates else 0.0:.1f} | "
            f"{sum(int(x['ai_budget_violations']) for x in items)} | "
            f"{sum(int(x['ai_crossed_release']) for x in items)} |"
        )
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(md_path.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
