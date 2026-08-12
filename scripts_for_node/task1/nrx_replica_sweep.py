#!/usr/bin/env python3
"""Closed- and open-loop NeuralRx CUDA Graph replica sweep."""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time

import cupy as cp
import numpy as np

from nrx_trt_direct import DirectNrx, stats


def parse_ints(value):
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return result


def parse_floats(value):
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return result


def wait_until(target_ns):
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


def make_runtimes(engine, replicas, warmup_rounds):
    runtimes = [DirectNrx(engine) for _ in range(replicas)]
    for runtime in runtimes:
        runtime.capture_graph()
    for _ in range(warmup_rounds):
        for runtime in runtimes:
            runtime.launch(use_graph=True)
    for runtime in runtimes:
        runtime.stream.synchronize()
    return runtimes


def closed_loop(runtimes, total_requests):
    replicas = len(runtimes)
    rounds = int(math.ceil(total_requests / replicas))
    actual_total = rounds * replicas
    events = [(cp.cuda.Event(), cp.cuda.Event()) for _ in range(actual_total)]
    index = 0
    wall_start = time.perf_counter_ns()
    for _ in range(rounds):
        for runtime in runtimes:
            start, end = events[index]
            with runtime.stream:
                start.record()
                runtime.launch(use_graph=True)
                end.record()
            index += 1
    for runtime in runtimes:
        runtime.stream.synchronize()
    wall_s = (time.perf_counter_ns() - wall_start) / 1e9
    service_ms = [float(cp.cuda.get_elapsed_time(start, end)) for start, end in events]
    return {
        "requests": actual_total,
        "rounds": rounds,
        "wall_s": wall_s,
        "throughput_slots_per_s": actual_total / wall_s,
        "service_ms": stats(service_ms),
    }


def queue_depth_metrics(arrivals_ms, completions_ms, duration_ms):
    completion_order = sorted(completions_ms)
    completed = 0
    max_outstanding = 0
    for index, arrival in enumerate(arrivals_ms):
        while completed < len(completion_order) and completion_order[completed] <= arrival:
            completed += 1
        outstanding = index + 1 - completed
        max_outstanding = max(max_outstanding, outstanding)
    return {
        "max_outstanding": max_outstanding,
        "backlog_at_window_end": sum(value > duration_ms for value in completions_ms),
    }


def open_loop(runtimes, rate, duration_s):
    replicas = len(runtimes)
    requests = int(round(rate * duration_s))
    completion_events = [cp.cuda.Event() for _ in range(requests)]
    origin = cp.cuda.Event()
    origin.record()
    origin.synchronize()
    wall_zero_ns = time.perf_counter_ns()
    intended_arrival_ms = []
    actual_lateness_us = []
    for index in range(requests):
        intended_ns = wall_zero_ns + round(index * 1e9 / rate)
        wait_until(intended_ns)
        now_ns = time.perf_counter_ns()
        intended_arrival_ms.append((intended_ns - wall_zero_ns) / 1e6)
        actual_lateness_us.append((now_ns - intended_ns) / 1e3)
        runtime = runtimes[index % replicas]
        with runtime.stream:
            runtime.launch(use_graph=True)
            completion_events[index].record()
    for runtime in runtimes:
        runtime.stream.synchronize()
    completion_ms = [
        float(cp.cuda.get_elapsed_time(origin, event)) for event in completion_events
    ]
    latency_ms = [
        completion - arrival
        for completion, arrival in zip(completion_ms, intended_arrival_ms)
    ]
    window_ms = duration_s * 1000.0
    queue = queue_depth_metrics(intended_arrival_ms, completion_ms, window_ms)
    last_arrival = intended_arrival_ms[-1] if intended_arrival_ms else 0.0
    drain_ms = max(completion_ms, default=0.0) - last_arrival
    return {
        "arrival_rate_slots_per_s": rate,
        "duration_s": duration_s,
        "requests": requests,
        "latency_ms": stats(latency_ms),
        "scheduler_lateness_us": stats(actual_lateness_us),
        "deadline_miss_ratio": {
            "1ms": float(np.mean(np.asarray(latency_ms) > 1.0)),
            "2ms": float(np.mean(np.asarray(latency_ms) > 2.0)),
            "5ms": float(np.mean(np.asarray(latency_ms) > 5.0)),
            "10ms": float(np.mean(np.asarray(latency_ms) > 10.0)),
        },
        "completion_span_ms": max(completion_ms, default=0.0),
        "drain_after_last_arrival_ms": drain_ms,
        **queue,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--replicas", type=parse_ints, default=parse_ints("1,2,4,8,16"))
    parser.add_argument("--rates", type=parse_floats, default=parse_floats("500,700,750,1000"))
    parser.add_argument("--closed-loop-requests", type=int, default=4000)
    parser.add_argument("--open-loop-duration", type=float, default=3.0)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.closed_loop_requests <= 0 or args.open_loop_duration <= 0:
        parser.error("request count and duration must be positive")

    result = {
        "engine": args.engine,
        "replica_counts": args.replicas,
        "arrival_rates_slots_per_s": args.rates,
        "closed_loop_requests_target": args.closed_loop_requests,
        "open_loop_duration_s": args.open_loop_duration,
        "configurations": [],
    }
    for replicas in args.replicas:
        print(f"[NRX-SWEEP] replicas={replicas} initializing", flush=True)
        runtimes = make_runtimes(args.engine, replicas, args.warmup_rounds)
        configuration = {
            "replicas": replicas,
            "closed_loop": closed_loop(runtimes, args.closed_loop_requests),
            "open_loop": [],
        }
        print(
            f"[NRX-SWEEP] replicas={replicas} "
            f"throughput={configuration['closed_loop']['throughput_slots_per_s']:.3f} slots/s",
            flush=True,
        )
        for rate in args.rates:
            item = open_loop(runtimes, rate, args.open_loop_duration)
            configuration["open_loop"].append(item)
            print(
                f"[NRX-SWEEP] replicas={replicas} rate={rate:.1f} "
                f"p99={item['latency_ms']['p99']:.3f}ms "
                f"backlog={item['backlog_at_window_end']}",
                flush=True,
            )
        finite = all(
            bool(cp.all(cp.isfinite(value)).item())
            for runtime in runtimes
            for value in runtime.outputs.values()
        )
        if not finite:
            raise RuntimeError(f"non-finite output with {replicas} replicas")
        configuration["outputs_finite"] = finite
        result["configurations"].append(configuration)
        del runtimes
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"[NRX-SWEEP] OK output={args.output}", flush=True)


if __name__ == "__main__":
    main()
