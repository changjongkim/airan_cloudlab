#!/usr/bin/env python3.11
"""Posthoc unique GPU interval overlap for two profiled MPS clients."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_nsys_client_pair import TABLES, activity


def clipped_union(rows: list[tuple[int, int]], lower: int, upper: int) -> list[tuple[int, int]]:
    clipped = sorted((max(start, lower), min(end, upper)) for start, end in rows
                     if start < upper and end > lower)
    merged: list[list[int]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def intersect(left: list[tuple[int, int]], right: list[tuple[int, int]]) -> tuple[int, int]:
    i = j = count = total_ns = 0
    while i < len(left) and j < len(right):
        start = max(left[i][0], right[j][0])
        end = min(left[i][1], right[j][1])
        if start < end:
            count += 1
            total_ns += end - start
        if left[i][1] <= right[j][1]:
            i += 1
        else:
            j += 1
    return count, total_ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--nrx-sqlite", type=Path, required=True)
    parser.add_argument("--ai-sqlite", type=Path, required=True)
    parser.add_argument("--controller", type=Path, required=True)
    parser.add_argument("--granularity", choices=tuple(TABLES), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    controller = json.loads(args.controller.read_text(encoding="utf-8"))
    nrx, _ = activity(args.nrx_sqlite, TABLES[args.granularity])
    ai, _ = activity(args.ai_sqlite, TABLES[args.granularity])
    first_release = min(row["release_ns"] for row in controller["records"])
    last_release = max(row["release_ns"] for row in controller["records"])
    upper = last_release + round(controller["deadline_ms"] * 1_000_000)
    count, total_ns = intersect(
        clipped_union(nrx, first_release, upper),
        clipped_union(ai, first_release, upper),
    )
    per_early_ai = []
    for record in controller["background_records"]:
        if record.get("phase") != "before_nrx_observation":
            continue
        lower, end = record["admitted_ns"], record["returned_ns"]
        segments, duration_ns = intersect(
            clipped_union(nrx, lower, end), clipped_union(ai, lower, end)
        )
        per_early_ai.append({
            "release_index": record["release_index"],
            "unique_overlap_segments": segments,
            "unique_overlap_ms": duration_ns / 1_000_000,
        })
    report = {
        "schema": "softwall-nsys-unique-overlap-posthoc-v1",
        "granularity": args.granularity,
        "runtime_unique_overlap_segments": count,
        "runtime_unique_overlap_ms": total_ns / 1_000_000,
        "early_ai": per_early_ai,
        "scope": "Posthoc unique interval intersection, aligned by the two Nsight systemClockNs values on one node. One profiled NeuralRx client and one profiled AI client only. This is observed concurrency, not a co-run worst-case bound.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"segments": count, "unique_overlap_ms": total_ns / 1_000_000}))


if __name__ == "__main__":
    main()
