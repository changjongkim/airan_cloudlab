#!/usr/bin/env python3.11
"""C176 bursty AI-arrival traces for the P180/D155 two-node mode.

Every pattern carries the same request sequence and differs only in arrival
times. Prompt lengths are drawn i.i.d. with a fixed seed from all valid
BurstGPT requests and rounded up to the Qwen prefill classes (capped at 512).

- steady: evenly spaced arrivals (inter-arrival CV 0);
- gamma_cv1, gamma_cv2, gamma_cv4, gamma_cv8: Gamma renewal arrivals with the
  given inter-arrival CV (CV 1 is Poisson), the common way to vary burstiness;
- burstgpt: the per-second arrival counts of a real BurstGPT segment.

The BurstGPT segment is chosen by a rule fixed before any run: among the
first qualifying windows whose request count is within 5% of the target
rate times the duration, the window whose inter-arrival CV is closest to the
median CV of those windows. BurstGPT timestamps have one-second resolution,
so the requests of one second are spread evenly inside that second.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import statistics
from pathlib import Path


BUCKETS = (16, 32, 64, 128, 256, 512)


def bucket(tokens: int) -> int:
    for value in BUCKETS:
        if tokens <= value:
            return value
    return BUCKETS[-1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def spread_within_seconds(stamps: list[float]) -> list[float]:
    """Spread the requests of each one-second timestamp evenly inside it."""
    out = []
    index = 0
    while index < len(stamps):
        end = index
        while end < len(stamps) and stamps[end] == stamps[index]:
            end += 1
        count = end - index
        out.extend(stamps[index] + (k + 0.5) / count for k in range(count))
        index = end
    return out


def cv(arrivals_s: list[float]) -> float:
    gaps = [b - a for a, b in zip(arrivals_s, arrivals_s[1:])]
    mean = statistics.mean(gaps)
    return statistics.pstdev(gaps) / mean if mean > 0 else 0.0


def gamma_arrivals(count: int, duration_s: float, target_cv: float, rng: random.Random) -> list[float]:
    """Gamma renewal arrivals with the given inter-arrival CV, scaled to the duration."""
    shape = 1.0 / (target_cv * target_cv)
    gaps = [rng.gammavariate(shape, 1.0) for _ in range(count)]
    total = sum(gaps)
    times, clock = [], 0.0
    for gap in gaps:
        times.append(clock)
        clock += gap * duration_s / total
    return times


def load_burstgpt(path: Path) -> list[tuple[float, int]]:
    rows = []
    with path.open(newline="") as stream:
        for row in csv.DictReader(stream):
            request = int(float(row["Request tokens"]))
            response = int(float(row["Response tokens"]))
            if request > 0 and response > 0:
                rows.append((float(row["Timestamp"]), request))
    rows.sort(key=lambda item: item[0])
    return rows


def choose_window(rows, duration_s: float, target: int, candidates: int) -> dict:
    stamps = [stamp for stamp, _ in rows]
    low, high = math.floor(target * 0.95), math.ceil(target * 1.05)
    windows = []
    end = 0
    for start in range(len(stamps)):
        # Windows start at the first request of a new second.
        if start > 0 and stamps[start] == stamps[start - 1]:
            continue
        while end < len(stamps) and stamps[end] < stamps[start] + duration_s:
            end += 1
        count = end - start
        if low <= count <= high:
            arrivals = spread_within_seconds(stamps[start:end])
            windows.append({"start": start, "end": end, "count": count, "cv": cv(arrivals)})
            if len(windows) >= candidates:
                break
    if not windows:
        raise SystemExit("no BurstGPT window matches the target rate")
    median_cv = statistics.median(window["cv"] for window in windows)
    chosen = min(windows, key=lambda window: (abs(window["cv"] - median_cv), window["start"]))
    chosen["median_cv"] = median_cv
    chosen["candidates"] = len(windows)
    return chosen


def bursts(count: int, duration_s: float, rng: random.Random, burst_size: int, burst_span_s: float) -> list[float]:
    """Groups of burst_size requests inside burst_span_s, separated by exponential gaps."""
    groups = math.ceil(count / burst_size)
    mean_gap = (duration_s - groups * burst_span_s) / groups
    if mean_gap <= 0:
        raise ValueError("burst span too long for the duration")
    times, clock = [], 0.0
    for group in range(groups):
        clock += rng.expovariate(1.0 / mean_gap)
        size = min(burst_size, count - len(times))
        times.extend(clock + burst_span_s * (k + rng.random()) / size for k in range(size))
        clock += burst_span_s
    times.sort()
    scale = duration_s / max(times[-1], 1e-9) * 0.999
    return [t * scale for t in times]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--burstgpt-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--rate", type=float, default=4.0, help="mean AI requests per second")
    parser.add_argument("--duration-s", type=float, default=108.0)
    parser.add_argument("--seed", type=int, default=17600001)
    parser.add_argument("--candidates", type=int, default=2000)
    args = parser.parse_args()

    rows = load_burstgpt(args.burstgpt_csv)
    target = round(args.rate * args.duration_s)
    window = choose_window(rows, args.duration_s, target, args.candidates)
    segment = rows[window["start"]:window["end"]]
    t0 = segment[0][0]
    natural = [t - t0 for t in spread_within_seconds([stamp for stamp, _ in segment])]
    count = len(segment)
    duration = args.duration_s
    # One prompt sequence for every pattern: i.i.d. draws from all valid requests.
    prompt_rng = random.Random(args.seed)
    prompts = [rows[prompt_rng.randrange(len(rows))][1] for _ in range(count)]
    segment = [(0.0, tokens) for tokens in prompts]
    patterns = {"steady": [duration * k / count for k in range(count)]}
    for offset, target_cv in enumerate((1, 2, 4, 8), start=1):
        patterns[f"gamma_cv{target_cv}"] = gamma_arrivals(count, duration, float(target_cv),
                                                          random.Random(args.seed + offset))
    patterns["burstgpt"] = natural
    args.output_dir.mkdir(parents=True, exist_ok=True)
    source = {"path": str(args.burstgpt_csv), "sha256": sha256(args.burstgpt_csv),
              "window_first_row": window["start"], "window_requests": count,
              "window_start_timestamp_s": t0, "window_cv": window["cv"],
              "median_cv_of_candidates": window["median_cv"], "candidates": window["candidates"],
              "rule": ("first qualifying windows with a request count within 5% of rate x duration; "
                       "choose the window whose CV is closest to their median CV")}
    summary = {}
    for name, arrivals in patterns.items():
        requests = []
        for index, ((_, tokens), arrival) in enumerate(zip(segment, arrivals)):
            context = bucket(tokens)
            requests.append({"request_id": f"c176-{name}-{index:05d}", "arrival_ms": round(arrival * 1000.0, 3),
                             "raw_request_tokens": tokens, "context_length": context,
                             "value_tokens": context})
        requests.sort(key=lambda item: item["arrival_ms"])
        counts = {str(b): sum(1 for r in requests if r["context_length"] == b) for b in BUCKETS}
        info = {"pattern": name, "requests": len(requests), "duration_s": duration,
                "rate_per_s": len(requests) / duration,
                "interarrival_cv": cv([r["arrival_ms"] / 1000.0 for r in requests]),
                "bucket_counts": counts}
        summary[name] = info
        (args.output_dir / f"c176_trace_{name}.json").write_text(json.dumps({
            "schema": "softwall-c176-burst-trace-v1", "source": source, "summary": info,
            "requests": requests}, indent=1) + "\n")
    print(json.dumps({"source": source, "patterns": summary}, indent=1))


if __name__ == "__main__":
    main()
