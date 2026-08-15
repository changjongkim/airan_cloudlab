#!/usr/bin/env python3
"""Control-only DART policy replay using measured component profiles.

This is a scheme/debug artifact, not a hardware performance result. It uses
measured profile bounds but synthetic, seeded service-time variation.
"""

from __future__ import annotations

import argparse
import heapq
import itertools
import json
import math
import random
from pathlib import Path

import numpy as np

from dart_runtime import (
    BackgroundUnit,
    CommitState,
    DartRequest,
    DartRuntime,
    EndpointState,
    ProfileTable,
    ServiceProfile,
)


def parse_trace(value: str):
    phases = []
    for raw in value.split(","):
        fields = raw.split(":")
        if len(fields) != 3:
            raise argparse.ArgumentTypeError("trace must be label:rate:duration,...")
        label, rate, duration = fields
        rate = float(rate)
        duration = float(duration)
        if rate <= 0 or duration <= 0:
            raise argparse.ArgumentTypeError("rate and duration must be positive")
        phases.append((label, rate, duration))
    return phases


def arrivals(phases):
    result = []
    now_ns = 0
    for label, rate, duration in phases:
        count = round(rate * duration)
        period_ns = 1_000_000_000 / rate
        for index in range(count):
            result.append((round(now_ns + index * period_ns), label))
        now_ns += round(duration * 1_000_000_000)
    return result


def percentile(values):
    if not values:
        return {"n": 0}
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "max": float(array.max()),
    }


def load_profiles(path: Path):
    raw = json.loads(path.read_text(encoding="utf-8"))
    profiles = {}
    endpoint_ids = []
    for item in raw["profiles"]:
        key = (
            item["endpoint_id"],
            item["tensor_class"],
            item["graph_id"],
            item["background_mode"],
        )
        profiles[key] = ServiceProfile(
            item["forward_ns"],
            item["service_ns"],
            item["backward_ns"],
            item["control_ns"],
            item["positive_error_ns"],
        )
        endpoint_ids.append(item["endpoint_id"])
    return raw, tuple(dict.fromkeys(endpoint_ids)), ProfileTable(profiles)


def simulate(
    profile_raw,
    endpoint_ids,
    profile_table,
    phases,
    policy,
    deadline_ns,
    conventional_ns,
    qmax_ns,
    seed,
):
    rng = random.Random(seed)
    endpoints = [EndpointState(endpoint_id, ring_depth=4096) for endpoint_id in endpoint_ids]
    runtime = DartRuntime(
        endpoints,
        profile_table,
        commit_guard_ns=50_000,
        background_mode="isolated",
    )
    actual_tails = {endpoint_id: 0 for endpoint_id in endpoint_ids}
    event_heap = []
    serial = itertools.count()
    transactions = []
    prediction_errors_us = []
    background_value = 0.0
    lease_count = 0

    def schedule(when_ns, kind, transaction):
        heapq.heappush(event_heap, (when_ns, next(serial), kind, transaction))

    def process(until_ns):
        while event_heap and event_heap[0][0] <= until_ns:
            when_ns, _, kind, transaction = heapq.heappop(event_heap)
            if kind == "fallback":
                if runtime.start_fallback(transaction, when_ns):
                    schedule(when_ns + conventional_ns, "conventional", transaction)
            elif kind == "conventional":
                runtime.complete_conventional(transaction, when_ns)
            elif kind == "nrx":
                runtime.complete_nrx(
                    transaction,
                    transaction.request.slot_id,
                    transaction.request.epoch,
                    when_ns,
                )
            elif kind == "expire":
                runtime.expire(transaction, when_ns)
            else:
                raise AssertionError(kind)

    trace_arrivals = arrivals(phases)
    for slot_id, (arrival_ns, phase) in enumerate(trace_arrivals):
        process(arrival_ns)
        request = DartRequest(
            slot_id=slot_id,
            epoch=slot_id + 1,
            graph_id=1,
            tensor_class=1,
            release_ns=arrival_ns,
            deadline_ns=arrival_ns + deadline_ns,
            fallback_latest_start_ns=arrival_ns + deadline_ns - conventional_ns - 50_000,
            payload_checksum=slot_id & 0xFFFFFFFF,
        )
        route_policy = {
            "S0": "static",
            "S1": "round_robin",
            "S2": "shortest_queue",
        }.get(policy, "predicted_finish")
        transaction = runtime.submit(request, arrival_ns, policy=route_policy)
        transactions.append((phase, transaction))
        schedule(request.deadline_ns + 1, "expire", transaction)

        if transaction.reservation is None:
            schedule(arrival_ns + conventional_ns, "conventional", transaction)
        else:
            reservation = transaction.reservation
            profile = reservation.profile
            base_ns = (
                profile.forward_ns
                + profile.service_ns
                + profile.backward_ns
                + profile.control_ns
            )
            sample_ns = round(base_ns * rng.lognormvariate(math.log(0.94), 0.08))
            if rng.random() < 0.01:
                sample_ns = round(sample_ns * 1.5)
            start_ns = max(arrival_ns, actual_tails[reservation.endpoint_id])
            finish_ns = start_ns + sample_ns
            actual_tails[reservation.endpoint_id] = finish_ns
            prediction_errors_us.append(
                (finish_ns - reservation.predicted_finish_ns) / 1000.0
            )
            schedule(finish_ns, "nrx", transaction)
            if policy in {"S6", "S7", "S8", "S9"}:
                schedule(request.fallback_latest_start_ns, "fallback", transaction)

        if policy in {"S7", "S8", "S9"}:
            next_arrival = (
                trace_arrivals[slot_id + 1][0]
                if slot_id + 1 < len(trace_arrivals)
                else arrival_ns + deadline_ns
            )
            endpoint_id = endpoint_ids[-1]
            if policy == "S7":
                units = [BackgroundUnit("fixed", max(1, qmax_ns - 25_000), 1.0)]
                guard = arrival_ns + qmax_ns
            else:
                units = [
                    BackgroundUnit("decode_step", 50_000, 1.0),
                    BackgroundUnit("microbatch", 200_000, 2.5),
                    BackgroundUnit("long_unit", 500_000, 4.0),
                ]
                guard = next_arrival + deadline_ns - conventional_ns
            lease = runtime.try_lease(
                endpoint_id,
                arrival_ns,
                earliest_latest_start_ns=guard,
                qmax_ns=qmax_ns,
                drain_guard_ns=25_000,
                units=units,
            )
            if lease:
                lease_count += 1
                background_value += lease.value
                actual_tails[endpoint_id] = max(
                    actual_tails[endpoint_id], lease.predicted_finish_ns
                )

    process(10**30)
    latency_ms = []
    by_phase = {}
    states = {state.value: 0 for state in CommitState}
    for phase, transaction in transactions:
        states[transaction.commit_state.value] += 1
        if transaction.commit_ns is not None:
            latency_ms.append(
                (transaction.commit_ns - transaction.request.release_ns) / 1_000_000
            )
        phase_data = by_phase.setdefault(phase, {"requests": 0, "misses": 0})
        phase_data["requests"] += 1
        if transaction.commit_state is CommitState.DEADLINE_MISS:
            phase_data["misses"] += 1
    for phase_data in by_phase.values():
        phase_data["miss_ratio"] = phase_data["misses"] / phase_data["requests"]

    misses = states[CommitState.DEADLINE_MISS.value]
    return {
        "policy": policy,
        "requests": len(transactions),
        "commit_states": states,
        "deadline_miss_ratio": misses / len(transactions),
        "latency_ms": percentile(latency_ms),
        "positive_prediction_error_us": percentile([
            max(0.0, value) for value in prediction_errors_us
        ]),
        "background": {"lease_count": lease_count, "value": background_value},
        "by_phase": by_phase,
        "runtime_metrics": dict(runtime.metrics),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--trace",
        type=parse_trace,
        default=parse_trace("low:500:1,burst:1400:1,low:500:1"),
    )
    parser.add_argument(
        "--policies", default="S0,S1,S2,S3,S6,S7,S8"
    )
    parser.add_argument("--deadline-ms", type=float, default=5.0)
    parser.add_argument("--conventional-ms", type=float, default=1.0)
    parser.add_argument("--qmax-us", type=float, default=500.0)
    parser.add_argument("--seed", type=int, default=20260813)
    args = parser.parse_args()
    if min(args.deadline_ms, args.conventional_ms, args.qmax_us) <= 0:
        parser.error("deadline, conventional time, and qmax must be positive")
    profile_path = Path(args.profile).resolve()
    raw, endpoint_ids, table = load_profiles(profile_path)
    policies = [item.strip() for item in args.policies.split(",") if item.strip()]
    results = [
        simulate(
            raw,
            endpoint_ids,
            table,
            args.trace,
            policy,
            round(args.deadline_ms * 1_000_000),
            round(args.conventional_ms * 1_000_000),
            round(args.qmax_us * 1000),
            args.seed,
        )
        for policy in policies
    ]
    output = {
        "schema": "dart-control-replay-v0",
        "scope": "control-only model using measured profiles; not hardware performance",
        "profile": str(profile_path),
        "profile_schema": raw["schema"],
        "trace": args.trace,
        "deadline_ms": args.deadline_ms,
        "conventional_ms": args.conventional_ms,
        "qmax_us": args.qmax_us,
        "seed": args.seed,
        "results": results,
    }
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(json.dumps(output, indent=2), encoding="utf-8")
    temporary.replace(output_path)
    for result in results:
        print(
            f"[DART-REPLAY] {result['policy']} miss={result['deadline_miss_ratio']:.6f} "
            f"p99={result['latency_ms'].get('p99', float('nan')):.3f}ms "
            f"bg_value={result['background']['value']:.1f}",
            flush=True,
        )
    print(f"[DART-REPLAY] OK output={output_path}", flush=True)


if __name__ == "__main__":
    main()
