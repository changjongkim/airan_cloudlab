#!/usr/bin/env python3
"""Extract stable MIGRx summaries from an Nsight Systems SQLite export.

The script intentionally treats profiling data as causal evidence, not as the
source of primary latency numbers. It only uses Python's standard library so it
can run on the Mac after reports are downloaded from CloudLab.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sqlite3
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def percentile(values: list[float], q: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    fraction = position - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def duration_summary(values_ns: list[int]) -> dict[str, float | int]:
    values_us = [value / 1_000.0 for value in values_ns]
    total_ns = sum(values_ns)
    return {
        "count": len(values_ns),
        "total_ms": total_ns / 1_000_000.0,
        "mean_us": (total_ns / len(values_ns) / 1_000.0)
        if values_ns
        else math.nan,
        "p50_us": percentile(values_us, 0.50),
        "p95_us": percentile(values_us, 0.95),
        "p99_us": percentile(values_us, 0.99),
        "max_us": max(values_us) if values_us else math.nan,
    }


def table_names(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table'"
        )
    }


def group_durations(
    rows: Iterable[tuple[Any, int, int]],
) -> list[dict[str, Any]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for raw_name, start, end in rows:
        name = str(raw_name) if raw_name not in (None, "") else "<unnamed>"
        if end is not None and end >= start:
            grouped[name].append(end - start)
    result = []
    for name, durations in grouped.items():
        result.append({"name": name, **duration_summary(durations)})
    result.sort(key=lambda row: (-float(row["total_ms"]), str(row["name"])))
    total = sum(float(row["total_ms"]) for row in result)
    for row in result:
        row["time_fraction"] = (
            float(row["total_ms"]) / total if total > 0.0 else math.nan
        )
    return result


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def extract_kernel(
    connection: sqlite3.Connection, tables: set[str]
) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_KERNEL" not in tables:
        return []
    rows = connection.execute(
        """
        SELECT strings.value, kernel.start, kernel.end
        FROM CUPTI_ACTIVITY_KIND_KERNEL AS kernel
        JOIN StringIds AS strings ON strings.id = kernel.demangledName
        """
    )
    return group_durations(rows)


def extract_runtime(
    connection: sqlite3.Connection, tables: set[str]
) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_RUNTIME" not in tables:
        return []
    rows = connection.execute(
        """
        SELECT strings.value, runtime.start, runtime.end
        FROM CUPTI_ACTIVITY_KIND_RUNTIME AS runtime
        JOIN StringIds AS strings ON strings.id = runtime.nameId
        """
    )
    return group_durations(rows)


def extract_nvtx(
    connection: sqlite3.Connection, tables: set[str]
) -> list[dict[str, Any]]:
    if "NVTX_EVENTS" not in tables:
        return []
    rows = connection.execute(
        """
        SELECT COALESCE(events.text, strings.value, '<unnamed>'),
               events.start, events.end
        FROM NVTX_EVENTS AS events
        LEFT JOIN StringIds AS strings ON strings.id = events.textId
        WHERE events.end IS NOT NULL
        """
    )
    return group_durations(rows)


def extract_memcpy(
    connection: sqlite3.Connection, tables: set[str]
) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_MEMCPY" not in tables:
        return []
    grouped: dict[tuple[str, str, str], dict[str, Any]] = defaultdict(
        lambda: {"durations": [], "bytes": 0}
    )
    rows = connection.execute(
        """
        SELECT COALESCE(operation.label, operation.name, CAST(copy.copyKind AS TEXT)),
               COALESCE(source.label, source.name, CAST(copy.srcKind AS TEXT)),
               COALESCE(destination.label, destination.name,
                        CAST(copy.dstKind AS TEXT)),
               copy.start, copy.end, copy.bytes
        FROM CUPTI_ACTIVITY_KIND_MEMCPY AS copy
        LEFT JOIN ENUM_CUDA_MEMCPY_OPER AS operation
               ON operation.id = copy.copyKind
        LEFT JOIN ENUM_CUDA_MEM_KIND AS source ON source.id = copy.srcKind
        LEFT JOIN ENUM_CUDA_MEM_KIND AS destination ON destination.id = copy.dstKind
        """
    )
    for operation, source, destination, start, end, byte_count in rows:
        if end < start:
            continue
        key = (str(operation), str(source), str(destination))
        grouped[key]["durations"].append(end - start)
        grouped[key]["bytes"] += int(byte_count)
    result = []
    for (operation, source, destination), data in grouped.items():
        result.append(
            {
                "operation": operation,
                "source": source,
                "destination": destination,
                "bytes": data["bytes"],
                **duration_summary(data["durations"]),
            }
        )
    result.sort(key=lambda row: (-int(row["bytes"]), str(row["operation"])))
    return result


def extract_sync(
    connection: sqlite3.Connection, tables: set[str]
) -> list[dict[str, Any]]:
    if "CUPTI_ACTIVITY_KIND_SYNCHRONIZATION" not in tables:
        return []
    rows = connection.execute(
        """
        SELECT COALESCE(sync_type.label, sync_type.name,
                        CAST(sync.syncType AS TEXT)),
               sync.start, sync.end
        FROM CUPTI_ACTIVITY_KIND_SYNCHRONIZATION AS sync
        LEFT JOIN ENUM_CUPTI_SYNC_TYPE AS sync_type
               ON sync_type.id = sync.syncType
        """
    )
    return group_durations(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("sqlite", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    if not args.sqlite.is_file():
        parser.error(f"SQLite report does not exist: {args.sqlite}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(f"file:{args.sqlite}?mode=ro", uri=True)
    try:
        tables = table_names(connection)
        summaries = {
            "kernel": extract_kernel(connection, tables),
            "runtime_api": extract_runtime(connection, tables),
            "memcpy": extract_memcpy(connection, tables),
            "synchronization": extract_sync(connection, tables),
            "nvtx": extract_nvtx(connection, tables),
        }
    finally:
        connection.close()

    for name, rows in summaries.items():
        write_csv(args.output_dir / f"{name}_summary.csv", rows)

    payload = {
        "source_sqlite": str(args.sqlite.resolve()),
        "source_size": args.sqlite.stat().st_size,
        "tables": sorted(tables),
        "row_counts": {name: len(rows) for name, rows in summaries.items()},
        "top": {
            name: rows[:10]
            for name, rows in summaries.items()
            if rows
        },
    }
    temporary = args.output_dir / "summary.json.tmp"
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    os.replace(temporary, args.output_dir / "summary.json")
    print(json.dumps(payload["row_counts"], sort_keys=True))


if __name__ == "__main__":
    main()
