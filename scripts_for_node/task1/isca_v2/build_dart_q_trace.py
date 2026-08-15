#!/usr/bin/env python3
"""Convert actual NRx open-loop hardware results into a G8 control trace."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dart_transaction_trace import (
    TransactionSample,
    TransactionTrace,
    write_trace,
)


def select_run(value: dict, policy: str, rate: float) -> dict:
    matches = [
        item
        for item in value["runs"]
        if item["policy"] == policy
        and abs(float(item["rate_slots_s"]) - rate) < 1e-9
    ]
    if len(matches) != 1:
        raise ValueError(
            f"expected one run for policy={policy} rate={rate}, got {len(matches)}"
        )
    return matches[0]


def convert(
    value: dict,
    policy: str,
    rate: float,
    deadline_ns: int,
    conventional_ns: int,
    commit_guard_ns: int,
) -> TransactionTrace:
    if value.get("schema") != "dart-nrx-policy-hardware-v0":
        raise ValueError("input is not an actual NRx policy hardware artifact")
    if conventional_ns + commit_guard_ns >= deadline_ns:
        raise ValueError("conventional recovery and guard do not fit the deadline")
    run = select_run(value, policy, rate)
    raw = run["raw"]
    arrivals = raw["arrival_ms"]
    completions = raw["completion_ms"]
    assignments = raw["assignment_device"]
    if not (len(arrivals) == len(completions) == len(assignments) == run["requests"]):
        raise ValueError("raw hardware arrays have inconsistent lengths")
    samples = []
    for index, (arrival_ms, completion_ms, endpoint) in enumerate(
        zip(arrivals, completions, assignments)
    ):
        release_ns = round(float(arrival_ms) * 1_000_000)
        remote_ready_ns = round(float(completion_ms) * 1_000_000)
        samples.append(TransactionSample(
            slot_id=index,
            epoch=index + 1,
            release_ns=release_ns,
            deadline_ns=release_ns + deadline_ns,
            fallback_latest_start_ns=(
                release_ns + deadline_ns - conventional_ns - commit_guard_ns
            ),
            conventional_service_ns=conventional_ns,
            remote_ready_ns=max(release_ns, remote_ready_ns),
            remote_admitted=True,
            payload_visible=True,
            endpoint_epoch_valid=True,
            fallback_reserved=True,
            utility=1.0,
        ))
    return TransactionTrace.build(samples, {
        "source_schema": value["schema"],
        "source_scope": value.get("scope"),
        "selected_policy": policy,
        "selected_rate_slots_s": rate,
        "assignment_devices": sorted(set(assignments)),
        "deadline_ns": deadline_ns,
        "conventional_service_ns": conventional_ns,
        "commit_guard_ns": commit_guard_ns,
        "radio_utility_provenance": "unavailable; utility fixed to 1.0",
        "transport_included": False,
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--policy", required=True)
    parser.add_argument("--rate", type=float, required=True)
    parser.add_argument("--deadline-ms", type=float, default=5.0)
    parser.add_argument("--conventional-ms", type=float, default=1.0)
    parser.add_argument("--commit-guard-us", type=float, default=50.0)
    args = parser.parse_args()
    if min(
        args.rate,
        args.deadline_ms,
        args.conventional_ms,
        args.commit_guard_us,
    ) <= 0:
        parser.error("rates and timing arguments must be positive")

    input_path = Path(args.input).resolve()
    value = json.loads(input_path.read_text(encoding="utf-8"))
    trace = convert(
        value,
        args.policy,
        args.rate,
        round(args.deadline_ms * 1_000_000),
        round(args.conventional_ms * 1_000_000),
        round(args.commit_guard_us * 1_000),
    )
    output_path = Path(args.output).resolve()
    write_trace(output_path, trace)
    print(
        f"[DART-Q-TRACE] OK transactions={len(trace.transactions)} "
        f"output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()

