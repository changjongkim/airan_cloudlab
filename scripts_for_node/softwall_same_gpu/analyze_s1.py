#!/usr/bin/env python3
"""Summarize the NeuralRx MPS cap/priority overlap sweep."""

from __future__ import annotations

import argparse
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--campaign", default="s1_nrx")
    parser.add_argument("--deadline-ms", type=float, default=35.0)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ai-label", default="NeuralRx")
    args = parser.parse_args()
    pattern = re.compile(
        re.escape(args.campaign)
        + r"_r(?P<round>[0-9]+)_(?P<condition>alone|cap[0-9]+_p[01])"
        + r"_job(?P<job>[0-9]+)(?P<role>_ran|_ai)?[.]json$"
    )
    pairs: dict[tuple[int, str, str], dict[str, Path]] = {}
    for path in args.raw.glob(f"{args.campaign}_r*_job*.json"):
        match = pattern.fullmatch(path.name)
        if not match:
            continue
        key = (int(match["round"]), match["condition"], match["job"])
        role = (match["role"] or "_ran").lstrip("_")
        pairs.setdefault(key, {})[role] = path

    rows = []
    for (round_number, condition, job), paths in sorted(pairs.items()):
        if "ran" not in paths or (condition != "alone" and "ai" not in paths):
            raise SystemExit(f"incomplete S1 result: {round_number}/{condition}/{job}")
        ran = json.loads(paths["ran"].read_text(encoding="utf-8"))
        records = ran["records"]
        response = [float(item["response_ms"]) for item in records]
        ai_units = 0
        ai_rate = 0.0
        ai_sms = 0
        if condition != "alone":
            ai = json.loads(paths["ai"].read_text(encoding="utf-8"))
            window_start = int(records[0]["release_ns"])
            window_end = int(records[-1]["release_ns"]) + round(
                float(ran["period_ms"]) * 1e6
            )
            ai_units = sum(
                window_start <= int(item["completed_ns"]) <= window_end
                for item in ai["records"]
            )
            ai_rate = ai_units / ((window_end - window_start) / 1e9)
            ai_sms = int(ai["visible_sm_count"])
        rows.append({
            "round": round_number,
            "condition": condition,
            "job": job,
            "samples": len(response),
            "ran_p99_ms": percentile(response, 0.99),
            "ran_max_ms": max(response),
            "deadline_misses": sum(x > args.deadline_ms for x in response),
            "incorrect_transport_blocks": sum(
                item.get("correct_tb") is False for item in records
            ),
            "ran_visible_sms": int(ran["visible_sm_count"]),
            "ai_visible_sms": ai_sms,
            "ai_units": ai_units,
            "ai_units_per_s": ai_rate,
        })
    if not rows:
        raise SystemExit("no S1 results found")

    grouped: dict[str, list[dict]] = {}
    for row in rows:
        grouped.setdefault(row["condition"], []).append(row)
    lines = [
        f"# SoftWall S1 {args.ai_label} overlap sweep",
        "",
        f"| condition | AI visible SMs | runs | RAN samples | misses | incorrect TB | p99 range (ms) | worst max (ms) | {args.ai_label}/s mean |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition, items in sorted(grouped.items()):
        p99 = [float(item["ran_p99_ms"]) for item in items]
        rates = [float(item["ai_units_per_s"]) for item in items]
        lines.append(
            f"| {condition} | {items[0]['ai_visible_sms']} | {len(items)} | "
            f"{sum(int(x['samples']) for x in items)} | "
            f"{sum(int(x['deadline_misses']) for x in items)} | "
            f"{sum(int(x['incorrect_transport_blocks']) for x in items)} | "
            f"{min(p99):.3f}–{max(p99):.3f} | "
            f"{max(float(x['ran_max_ms']) for x in items):.3f} | "
            f"{statistics.mean(rates):.1f} |"
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(rows, indent=2), encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
