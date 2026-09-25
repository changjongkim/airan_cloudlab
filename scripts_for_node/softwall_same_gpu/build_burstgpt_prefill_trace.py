#!/usr/bin/env python3
"""Build the frozen bounded-Qwen trace used by the SoftWall request gate."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict, deque
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


def valid_rows(source: Path):
    with source.open(newline="", encoding="utf-8") as stream:
        for source_row, row in enumerate(csv.DictReader(stream)):
            request_tokens = int(row["Request tokens"])
            response_tokens = int(row["Response tokens"])
            if request_tokens <= 0 or response_tokens <= 0:
                continue
            yield {
                "source_row": source_row,
                "timestamp_s": int(row["Timestamp"]),
                "model": row["Model"],
                "log_type": row["Log Type"],
                "request_tokens": request_tokens,
                "response_tokens": response_tokens,
            }


def densest_window(rows: list[dict], window_s: int) -> tuple[int, int]:
    queue: deque[dict] = deque()
    best_count = -1
    best_start = -1
    for row in rows:
        queue.append(row)
        while queue and row["timestamp_s"] - queue[0]["timestamp_s"] >= window_s:
            queue.popleft()
        candidate = (len(queue), -queue[0]["timestamp_s"])
        current = (best_count, -best_start) if best_start >= 0 else (-1, 0)
        if candidate > current:
            best_count = len(queue)
            best_start = queue[0]["timestamp_s"]
    return best_start, best_count


def within_second_offsets_ms(rows: list[dict], mode: str) -> list[float]:
    """Resolve integer-second timestamps without changing per-second counts."""
    if mode == "burst":
        return [0.0] * len(rows)
    counts = Counter(row["timestamp_s"] for row in rows)
    ranks = defaultdict(int)
    offsets = []
    for row in rows:
        timestamp = row["timestamp_s"]
        rank = ranks[timestamp]
        ranks[timestamp] += 1
        # Midpoints avoid placing a synthetic request exactly on a second
        # boundary while preserving the source order within that second.
        offsets.append((rank + 0.5) * 1000.0 / counts[timestamp])
    return offsets


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--window-s", type=int, default=60)
    parser.add_argument("--slo-ms", type=int, default=1000)
    parser.add_argument("--time-scale", type=float, default=1.0)
    parser.add_argument(
        "--within-second", choices=("burst", "uniform"), default="burst"
    )
    args = parser.parse_args()
    if args.window_s <= 0 or args.slo_ms <= 0 or args.time_scale <= 0:
        parser.error("window, SLO, and time scale must be positive")

    rows = list(valid_rows(args.source))
    start_s, expected_count = densest_window(rows, args.window_s)
    selected = [
        row for row in rows
        if start_s <= row["timestamp_s"] < start_s + args.window_s
    ]
    if len(selected) != expected_count:
        raise RuntimeError("densest-window replay is internally inconsistent")

    requests = []
    capped = 0
    offsets_ms = within_second_offsets_ms(selected, args.within_second)
    for index, (row, within_second_ms) in enumerate(zip(selected, offsets_ms)):
        context_length, was_capped = bucket_tokens(row["request_tokens"])
        capped += int(was_capped)
        arrival_ms = round(
            ((row["timestamp_s"] - start_s) * 1000.0 + within_second_ms)
            / args.time_scale,
            6,
        )
        requests.append({
            "request_id": f"burstgpt-{row['source_row']}",
            "source_row": row["source_row"],
            "source_timestamp_s": row["timestamp_s"],
            "arrival_ms": arrival_ms,
            "deadline_ms": arrival_ms + args.slo_ms,
            "raw_request_tokens": row["request_tokens"],
            "raw_response_tokens": row["response_tokens"],
            "context_length": context_length,
            "value_tokens": min(row["request_tokens"], BUCKETS[-1]),
            "model": row["model"],
            "log_type": row["log_type"],
            "source_order": index,
        })

    result = {
        "schema": "softwall-burstgpt-prefill-trace-v1",
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "license": "CC-BY-4.0",
        "selection": {
            "rule": "earliest densest valid fixed-width window; valid means positive request and response tokens",
            "window_s": args.window_s,
            "source_start_s": start_s,
            "source_end_exclusive_s": start_s + args.window_s,
            "time_scale": args.time_scale,
            "within_second": args.within_second,
            "within_second_rule": (
                "preserve integer-second bursts"
                if args.within_second == "burst" else
                "deterministic source-order midpoints uniformly spaced within each integer second"
            ),
            "fixed_slo_ms": args.slo_ms,
        },
        "mapping": {
            "phase": "Qwen2.5-1.5B prefill",
            "allowed_context_lengths": BUCKETS,
            "rule": "round positive request tokens up to an allowed bucket and cap above 512",
            "value": "min(raw request tokens, 512) completed before fixed deadline",
        },
        "summary": {
            "requests": len(requests),
            "capped_requests": capped,
            "capped_fraction": capped / len(requests),
            "arrival_duration_ms": requests[-1]["arrival_ms"] if requests else 0,
            "offered_value_tokens": sum(row["value_tokens"] for row in requests),
            "bucket_counts": {
                str(bucket): sum(row["context_length"] == bucket for row in requests)
                for bucket in BUCKETS
            },
        },
        "requests": requests,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
