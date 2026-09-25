#!/usr/bin/env python3.11
"""Build deterministic non-overlapping BurstGPT windows for C159."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path


BUCKETS = (16, 32, 64, 128, 256, 512)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def bucket_tokens(tokens: int) -> tuple[int, bool]:
    for bucket in BUCKETS:
        if tokens <= bucket:
            return bucket, False
    return BUCKETS[-1], True


def valid_rows(source: Path, start_s: int, end_s: int) -> list[dict]:
    rows = []
    with source.open(newline="", encoding="utf-8") as stream:
        for source_row, row in enumerate(csv.DictReader(stream)):
            timestamp = int(row["Timestamp"])
            if timestamp < start_s or timestamp >= end_s:
                continue
            request_tokens = int(row["Request tokens"])
            response_tokens = int(row["Response tokens"])
            if request_tokens <= 0 or response_tokens <= 0:
                continue
            rows.append({
                "source_row": source_row,
                "timestamp_s": timestamp,
                "model": row["Model"],
                "log_type": row["Log Type"],
                "request_tokens": request_tokens,
                "response_tokens": response_tokens,
            })
    rows.sort(key=lambda item: (item["timestamp_s"], item["source_row"]))
    return rows


def select_nonoverlapping_windows(
    rows: list[dict], window_s: int, count: int, min_gap_s: int = 0
) -> list[dict]:
    """Rank request-anchored windows by density and greedily keep disjoint ones."""
    timestamps = [row["timestamp_s"] for row in rows]
    unique_starts = sorted(set(timestamps))
    candidates = []
    right = 0
    left = 0
    for start in unique_starts:
        while left < len(timestamps) and timestamps[left] < start:
            left += 1
        right = max(right, left)
        while right < len(timestamps) and timestamps[right] < start + window_s:
            right += 1
        candidates.append({"start_s": start, "end_s": start + window_s, "requests": right - left})
    candidates.sort(key=lambda item: (-item["requests"], item["start_s"]))

    selected = []
    for candidate in candidates:
        if all(
            candidate["end_s"] + min_gap_s <= prior["start_s"]
            or prior["end_s"] + min_gap_s <= candidate["start_s"]
            for prior in selected
        ):
            selected.append(candidate)
            if len(selected) == count:
                break
    if len(selected) != count:
        raise RuntimeError(f"only found {len(selected)} non-overlapping windows; requested {count}")
    return selected


def within_second_offsets_ms(rows: list[dict], mode: str) -> list[float]:
    if mode == "burst":
        return [0.0] * len(rows)
    counts = Counter(row["timestamp_s"] for row in rows)
    ranks = defaultdict(int)
    offsets = []
    for row in rows:
        timestamp = row["timestamp_s"]
        rank = ranks[timestamp]
        ranks[timestamp] += 1
        offsets.append((rank + 0.5) * 1000.0 / counts[timestamp])
    return offsets


def materialize_window(
    rows: list[dict], selection: dict, index: int, slo_ms: int, within_second: str
) -> dict:
    chosen = [
        row for row in rows
        if selection["start_s"] <= row["timestamp_s"] < selection["end_s"]
    ]
    offsets = within_second_offsets_ms(chosen, within_second)
    requests = []
    capped = 0
    for order, (row, offset_ms) in enumerate(zip(chosen, offsets)):
        context, was_capped = bucket_tokens(row["request_tokens"])
        capped += int(was_capped)
        arrival_ms = round(
            (row["timestamp_s"] - selection["start_s"]) * 1000.0 + offset_ms,
            6,
        )
        requests.append({
            "request_id": f"c159-w{index:02d}-burstgpt-{row['source_row']}",
            "source_row": row["source_row"],
            "source_timestamp_s": row["timestamp_s"],
            "arrival_ms": arrival_ms,
            "deadline_ms": arrival_ms + slo_ms,
            "raw_request_tokens": row["request_tokens"],
            "raw_response_tokens": row["response_tokens"],
            "context_length": context,
            "value_tokens": min(row["request_tokens"], BUCKETS[-1]),
            "model": row["model"],
            "log_type": row["log_type"],
            "source_order": order,
        })
    if len(requests) != selection["requests"]:
        raise RuntimeError("window request count changed during materialization")
    return {
        "window_id": f"w{index:02d}-{selection['start_s']}",
        "source_start_s": selection["start_s"],
        "source_end_exclusive_s": selection["end_s"],
        "summary": {
            "requests": len(requests),
            "capped_requests": capped,
            "offered_value_tokens": sum(item["value_tokens"] for item in requests),
            "bucket_counts": {
                str(bucket): sum(item["context_length"] == bucket for item in requests)
                for bucket in BUCKETS
            },
        },
        "requests": requests,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--partition-name", required=True)
    parser.add_argument("--partition-start-s", type=int, required=True)
    parser.add_argument("--partition-end-s", type=int, required=True)
    parser.add_argument("--window-s", type=int, default=60)
    parser.add_argument("--windows", type=int, default=4)
    parser.add_argument("--min-gap-s", type=int, default=0)
    parser.add_argument("--slo-ms", type=int, default=1000)
    parser.add_argument("--within-second", choices=("burst", "uniform"), default="burst")
    args = parser.parse_args()
    if not (
        args.partition_start_s < args.partition_end_s
        and args.window_s > 0
        and args.windows > 0
        and args.min_gap_s >= 0
        and args.slo_ms > 0
    ):
        parser.error("invalid partition, window, gap, count, or SLO")

    rows = valid_rows(args.source, args.partition_start_s, args.partition_end_s)
    selected = select_nonoverlapping_windows(
        rows, args.window_s, args.windows, args.min_gap_s
    )
    windows = [
        materialize_window(rows, item, index, args.slo_ms, args.within_second)
        for index, item in enumerate(selected)
    ]
    result = {
        "schema": "softwall-c159-partitioned-burstgpt-trace-v1",
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "license": "CC-BY-4.0",
        "partition": {
            "name": args.partition_name,
            "start_s": args.partition_start_s,
            "end_exclusive_s": args.partition_end_s,
        },
        "selection": {
            "rule": "rank request-anchored fixed windows by valid-request count descending then start ascending; greedily retain non-overlapping windows",
            "valid_request": "positive request and response tokens",
            "window_s": args.window_s,
            "windows": args.windows,
            "min_gap_s": args.min_gap_s,
            "within_second": args.within_second,
            "fixed_slo_ms": args.slo_ms,
        },
        "mapping": {
            "phase": "Qwen2.5-1.5B prefill",
            "allowed_context_lengths": BUCKETS,
            "rule": "round positive request tokens up to an allowed bucket and cap above 512",
            "value": "min(raw request tokens, 512) completed before fixed deadline",
        },
        "summary": {
            "valid_partition_requests": len(rows),
            "selected_windows": len(windows),
            "selected_requests": sum(item["summary"]["requests"] for item in windows),
            "offered_value_tokens": sum(item["summary"]["offered_value_tokens"] for item in windows),
        },
        "windows": windows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": result["summary"], "windows": [
        {"window_id": item["window_id"], **item["summary"]} for item in windows
    ]}, indent=2))


if __name__ == "__main__":
    main()
