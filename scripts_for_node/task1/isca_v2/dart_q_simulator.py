#!/usr/bin/env python3
"""Trace-driven control-path scaffold for Host/Device/DART-Q comparison.

This simulator does not predict GPU kernel or transport time.  Those values
come from a ``dart-transaction-input-v1`` trace.  It models only when each
architecture observes releases/completions, fires the reserved fallback, and
commits a winner.  Built-in profiles are explicitly illustrative placeholders;
paper results must replace them with measured/calibrated profiles.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable, Optional

from dart_transaction_trace import TransactionSample, read_trace


OUTPUT_SCHEMA = "dart-q-control-simulation-v1"


@dataclasses.dataclass(frozen=True)
class ArchitectureProfile:
    name: str
    submit_control_ns: int
    completion_signal_ns: int
    completion_poll_ns: int
    fallback_timer_ns: int
    graph_launch_ns: int
    commit_ns: int
    reserved_sms: int
    provenance: str = "illustrative-placeholder"

    def __post_init__(self) -> None:
        numeric = dataclasses.astuple(self)[1:8]
        if any(value < 0 for value in numeric):
            raise ValueError("architecture latencies must be non-negative")


DEFAULT_PROFILES = {
    "host": ArchitectureProfile(
        "host", 20_000, 20_000, 50_000, 25_000, 15_000, 5_000, 0
    ),
    "device": ArchitectureProfile(
        "device", 5_000, 5_000, 0, 5_000, 5_000, 2_000, 1
    ),
    "dartq": ArchitectureProfile(
        "dartq", 200, 200, 0, 200, 200, 100, 0
    ),
}


def percentile(values: list[float], pct: float) -> Optional[float]:
    if not values:
        return None
    ordered = sorted(values)
    rank = (len(ordered) - 1) * pct / 100.0
    lower = math.floor(rank)
    upper = math.ceil(rank)
    if lower == upper:
        return ordered[lower]
    weight = rank - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def observe_completion(ready_ns: int, profile: ArchitectureProfile) -> int:
    observed = ready_ns
    if profile.completion_poll_ns:
        period = profile.completion_poll_ns
        observed = ((observed + period - 1) // period) * period
    return observed + profile.completion_signal_ns


def simulate_one(
    sample: TransactionSample,
    profile: ArchitectureProfile,
) -> dict:
    request_observed_ns = sample.release_ns + profile.submit_control_ns
    fallback_fire_ns = max(
        request_observed_ns,
        sample.fallback_latest_start_ns + profile.fallback_timer_ns,
    )
    fallback_started = False

    remote_commit_ns = None
    remote_valid = (
        sample.remote_admitted
        and sample.remote_ready_ns is not None
        and sample.payload_visible
        and sample.endpoint_epoch_valid
    )
    if remote_valid:
        remote_commit_ns = (
            observe_completion(sample.remote_ready_ns, profile)
            + profile.commit_ns
        )

    if remote_commit_ns is not None and remote_commit_ns < fallback_fire_ns:
        winner = "nrx"
        commit_ns = remote_commit_ns
    else:
        fallback_started = True
        fallback_queue_delay_ns = (
            0
            if sample.fallback_reserved
            else sample.unreserved_fallback_delay_ns
        )
        conventional_ready_ns = (
            fallback_fire_ns
            + fallback_queue_delay_ns
            + profile.graph_launch_ns
            + sample.conventional_service_ns
        )
        conventional_commit_ns = conventional_ready_ns + profile.commit_ns
        if (
            remote_commit_ns is not None
            and remote_commit_ns < conventional_commit_ns
        ):
            winner = "nrx"
            commit_ns = remote_commit_ns
        else:
            winner = "conventional"
            commit_ns = conventional_commit_ns

    timely = commit_ns <= sample.deadline_ns
    if not timely:
        winner = "deadline_miss"
    return {
        "slot_id": sample.slot_id,
        "epoch": sample.epoch,
        "winner": winner,
        "timely": timely,
        "fallback_started": fallback_started,
        "release_ns": sample.release_ns,
        "deadline_ns": sample.deadline_ns,
        "commit_ns": commit_ns,
        "latency_ns": commit_ns - sample.release_ns,
        "remote_valid": remote_valid,
        "fallback_reserved": sample.fallback_reserved,
        "delivered_utility": sample.utility if winner == "nrx" else 0.0,
    }


def summarize(rows: list[dict], profile: ArchitectureProfile) -> dict:
    states = Counter(row["winner"] for row in rows)
    latencies_us = [row["latency_ns"] / 1000.0 for row in rows]
    timely = sum(row["timely"] for row in rows)
    return {
        "profile": dataclasses.asdict(profile),
        "transactions": len(rows),
        "winners": dict(states),
        "deadline_miss_ratio": 1.0 - timely / len(rows) if rows else 0.0,
        "fallback_start_ratio": (
            sum(row["fallback_started"] for row in rows) / len(rows)
            if rows
            else 0.0
        ),
        "delivered_utility": sum(row["delivered_utility"] for row in rows),
        "latency_us": {
            "p50": percentile(latencies_us, 50),
            "p95": percentile(latencies_us, 95),
            "p99": percentile(latencies_us, 99),
            "p99_9": percentile(latencies_us, 99.9),
            "max": max(latencies_us) if latencies_us else None,
        },
        "rows": rows,
    }


def simulate(
    transactions: Iterable[TransactionSample],
    profile: ArchitectureProfile,
) -> dict:
    return summarize([simulate_one(item, profile) for item in transactions], profile)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--modes", default="host,device,dartq")
    args = parser.parse_args()

    trace_path = Path(args.input).resolve()
    trace = read_trace(trace_path)
    modes = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown = sorted(set(modes) - set(DEFAULT_PROFILES))
    if unknown:
        parser.error(f"unknown modes: {','.join(unknown)}")
    results = {
        mode: simulate(trace.transactions, DEFAULT_PROFILES[mode])
        for mode in modes
    }
    output = {
        "schema": OUTPUT_SCHEMA,
        "scope": (
            "control-path what-if simulation; component readiness comes from input; "
            "built-in control profiles are illustrative and are not measured results"
        ),
        "input": str(trace_path),
        "input_metadata": trace.metadata,
        "results": results,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(output, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(output_path)
    for mode, result in results.items():
        print(
            f"[DART-Q-SIM] mode={mode} miss={result['deadline_miss_ratio']:.6f} "
            f"p99={result['latency_us']['p99']:.3f}us "
            f"fallback={result['fallback_start_ratio']:.6f}",
            flush=True,
        )
    print(f"[DART-Q-SIM] OK output={output_path}", flush=True)


if __name__ == "__main__":
    main()

