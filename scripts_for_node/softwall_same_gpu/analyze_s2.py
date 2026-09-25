#!/usr/bin/env python3
"""Summarize S2 hardware policy campaigns."""

from __future__ import annotations

import argparse
import json
import re
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    pattern = re.compile(
        re.escape(args.campaign)
        + r"_r(?P<round>[0-9]+)_(?P<policy>conventional_only|nrx_only|eager_dual|s2_reserved)"
        + r"_job(?P<job>[0-9]+)_ran[.]json$"
    )
    rows = []
    for path in sorted(args.raw.glob(f"{args.campaign}_r*_ran.json")):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        value = json.loads(path.read_text(encoding="utf-8"))
        duration_s = value["iterations"] * value["period_ms"] / 1000.0
        rows.append({
            "round": int(match["round"]),
            "job": match["job"],
            "policy": match["policy"],
            "releases": value["iterations"],
            "deadline_misses": value["deadline_misses"],
            "incorrect_releases": value["incorrect_releases"],
            "p99_ms": value["response_ms"]["p99"],
            "max_ms": value["response_ms"]["max"],
            "nrx_commits": value["nrx_commits"],
            "conv_commits": value["conv_commits"],
            "fallback_started": value["fallback_started"],
            "duplicate_commits": value["duplicate_commits"],
            "admission_rejections": value["admission_rejections"],
            "nrx_bound_violations": value["nrx_bound_violations"],
            "conv_bound_violations": value["conv_bound_violations"],
            "background_units": value["background_units"],
            "background_units_per_s": value["background_units"] / duration_s,
            "background_budget_violations": value["background_budget_violations"],
            "background_release_crossings": value["background_release_crossings"],
        })
    if not rows:
        raise SystemExit("no S2 campaign results found")
    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["policy"], []).append(row)
    lines = [
        "# SoftWall S2 hardware policy campaign",
        "",
        "| policy | runs | releases | misses | incorrect | p99 range (ms) | worst max (ms) | NRx commits | conv commits | fallback | duplicate commits | admission rejects | bound violations | background/s | AI violations/crossings |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for policy, items in sorted(grouped.items()):
        p99 = [float(item["p99_ms"]) for item in items]
        lines.append(
            f"| {policy} | {len(items)} | {sum(x['releases'] for x in items)} | "
            f"{sum(x['deadline_misses'] for x in items)} | "
            f"{sum(x['incorrect_releases'] for x in items)} | "
            f"{min(p99):.3f}–{max(p99):.3f} | "
            f"{max(float(x['max_ms']) for x in items):.3f} | "
            f"{sum(x['nrx_commits'] for x in items)} | "
            f"{sum(x['conv_commits'] for x in items)} | "
            f"{sum(x['fallback_started'] for x in items)} | "
            f"{sum(x['duplicate_commits'] for x in items)} | "
            f"{sum(x['admission_rejections'] for x in items)} | "
            f"{sum(x['nrx_bound_violations'] + x['conv_bound_violations'] for x in items)} | "
            f"{statistics.mean(x['background_units_per_s'] for x in items):.1f} | "
            f"{sum(x['background_budget_violations'] for x in items)}/"
            f"{sum(x['background_release_crossings'] for x in items)} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
