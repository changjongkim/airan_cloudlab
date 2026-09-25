#!/usr/bin/env python3.11
"""Compare Nsight GPU activity intervals from two clients on one node clock.

Nsight's exported activity timestamps are relative to each capture's session
clock. TARGET_INFO_SESSION_START_TIME.systemClockNs aligns those captures on
the same host. The controller's monotonic timestamps provide a separate check
that the translated intervals land within the measured request windows.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path


TABLES = {
    "graph": "CUPTI_ACTIVITY_KIND_GRAPH_TRACE",
    "kernel": "CUPTI_ACTIVITY_KIND_KERNEL",
}


def activity(path: Path, table: str) -> tuple[list[tuple[int, int]], int]:
    with sqlite3.connect(path) as database:
        base = database.execute(
            "SELECT systemClockNs FROM TARGET_INFO_SESSION_START_TIME"
        ).fetchone()[0]
        present = database.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name=?", (table,)
        ).fetchone()
        if not present:
            return [], base
        rows = database.execute(f"SELECT start, end FROM {table} ORDER BY start").fetchall()
    return [(base + start, base + end) for start, end in rows], base


def overlapping_pairs(
    left: list[tuple[int, int]], right: list[tuple[int, int]],
    lower: int, upper: int,
) -> tuple[int, int]:
    count = 0
    total_ns = 0
    for l_start, l_end in left:
        if l_end <= lower or l_start >= upper:
            continue
        for r_start, r_end in right:
            overlap = min(l_end, r_end, upper) - max(l_start, r_start, lower)
            if overlap > 0:
                count += 1
                total_ns += overlap
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
    table = TABLES[args.granularity]
    nrx, nrx_base = activity(args.nrx_sqlite, table)
    ai, ai_base = activity(args.ai_sqlite, table)
    first_release = min(row["release_ns"] for row in controller["records"])
    last_release = max(row["release_ns"] for row in controller["records"])
    upper = last_release + round(controller["deadline_ms"] * 1_000_000)
    nrx_runtime = [row for row in nrx if row[0] >= first_release and row[0] <= upper]
    ai_runtime = [row for row in ai if row[0] >= first_release and row[0] <= upper]
    pairs, overlap_ns = overlapping_pairs(nrx_runtime, ai_runtime, first_release, upper)
    early_rows = []
    for record in controller["background_records"]:
        if record.get("phase") != "before_nrx_observation":
            continue
        lower, end = record["admitted_ns"], record["returned_ns"]
        nrx_in = [row for row in nrx_runtime if row[0] < end and row[1] > lower]
        ai_in = [row for row in ai_runtime if row[0] < end and row[1] > lower]
        count, duration = overlapping_pairs(nrx_in, ai_in, lower, end)
        early_rows.append({
            "release_index": record["release_index"],
            "nrx_intervals": len(nrx_in),
            "ai_intervals": len(ai_in),
            "overlap_pairs": count,
            "overlap_ms": duration / 1_000_000,
        })
    report = {
        "schema": "softwall-nsys-client-pair-overlap-v1",
        "granularity": args.granularity,
        "clock_alignment": "systemClockNs plus Nsight activity timestamp, cross-checked against controller monotonic request windows",
        "nrx_session_start_ns": nrx_base,
        "ai_session_start_ns": ai_base,
        "first_release_ns": first_release,
        "runtime_nrx_intervals": len(nrx_runtime),
        "runtime_ai_intervals": len(ai_runtime),
        "runtime_overlap_pairs": pairs,
        "runtime_overlap_ms_sum": overlap_ns / 1_000_000,
        "early_ai_records": early_rows,
        "controller_deadline_misses": controller["deadline_misses"],
        "interpretation": (
            "Graph granularity reports overlapping GPU graph execution windows, "
            "not necessarily individual kernel concurrency. Kernel granularity "
            "can show simultaneous kernel execution intervals but this profiled "
            "small sample does not establish a worst-case service bound or deadline guarantee."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: report[key] for key in (
        "runtime_nrx_intervals", "runtime_ai_intervals",
        "runtime_overlap_pairs", "runtime_overlap_ms_sum",
    )}))


if __name__ == "__main__":
    main()
