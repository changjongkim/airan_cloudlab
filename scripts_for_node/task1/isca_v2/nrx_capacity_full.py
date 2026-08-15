#!/usr/bin/env python3
"""Regular single-device NRx replica capacity and queue-stability sweep."""

from __future__ import annotations

import argparse
import gc
import json
import os
import time

import cupy as cp
import numpy as np

from nrx_trt_direct import DirectNrx


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size),
        "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "min": float(array.min()),
        "max": float(array.max()),
    }


def parse_ints(value: str) -> list[int]:
    values = [int(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive integers")
    return values


def parse_floats(value: str) -> list[float]:
    values = [float(item) for item in value.split(",") if item]
    if not values or any(item <= 0 for item in values):
        raise argparse.ArgumentTypeError("expected comma-separated positive numbers")
    return values


def wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


def make_runtimes(engine: str, replicas: int, warmup: int) -> list[DirectNrx]:
    runtimes = [DirectNrx(engine) for _ in range(replicas)]
    for runtime in runtimes:
        runtime.capture_graph()
    for _ in range(warmup):
        for runtime in runtimes:
            runtime.launch(use_graph=True)
    for runtime in runtimes:
        runtime.stream.synchronize()
    return runtimes


def synchronize(runtimes: list[DirectNrx]) -> None:
    for runtime in runtimes:
        runtime.stream.synchronize()


def saturated(runtimes: list[DirectNrx], duration_s: float) -> dict:
    events = [(cp.cuda.Event(), cp.cuda.Event()) for _ in runtimes]
    service_ms: list[float] = []
    rounds = 0
    start_ns = time.perf_counter_ns()
    deadline_ns = start_ns + round(duration_s * 1e9)
    while time.perf_counter_ns() < deadline_ns:
        for runtime, (begin, end) in zip(runtimes, events):
            with runtime.stream:
                begin.record()
                runtime.launch(use_graph=True)
                end.record()
        synchronize(runtimes)
        for begin, end in events:
            service_ms.append(float(cp.cuda.get_elapsed_time(begin, end)))
        rounds += 1
    end_ns = time.perf_counter_ns()
    requests = rounds * len(runtimes)
    wall_s = (end_ns - start_ns) / 1e9
    return {
        "duration_target_s": duration_s,
        "duration_actual_s": wall_s,
        "requests": requests,
        "rounds": rounds,
        "throughput_slots_per_s": requests / wall_s,
        "service_ms": stats(service_ms),
        "raw_service_ms": service_ms,
    }


def queue_metrics(arrivals_ms: list[float], completions_ms: list[float], window_ms: float) -> dict:
    completion_order = sorted(completions_ms)
    completed = 0
    max_outstanding = 0
    for index, arrival in enumerate(arrivals_ms):
        while completed < len(completion_order) and completion_order[completed] <= arrival:
            completed += 1
        max_outstanding = max(max_outstanding, index + 1 - completed)
    return {
        "max_outstanding": max_outstanding,
        "backlog_at_window_end": sum(item > window_ms for item in completions_ms),
    }


def open_loop(runtimes: list[DirectNrx], rate: float, duration_s: float) -> dict:
    replicas = len(runtimes)
    requests = int(round(rate * duration_s))
    completions = [cp.cuda.Event() for _ in range(requests)]
    origin = cp.cuda.Event()
    synchronize(runtimes)
    origin.record()
    origin.synchronize()
    wall_zero_ns = time.perf_counter_ns() + 50_000_000
    arrivals_ms: list[float] = []
    lateness_us: list[float] = []
    for index in range(requests):
        target_ns = wall_zero_ns + round(index * 1e9 / rate)
        wait_until(target_ns)
        now_ns = time.perf_counter_ns()
        arrivals_ms.append((target_ns - wall_zero_ns) / 1e6)
        lateness_us.append((now_ns - target_ns) / 1e3)
        runtime = runtimes[index % replicas]
        with runtime.stream:
            runtime.launch(use_graph=True)
            completions[index].record()
    synchronize(runtimes)
    completion_ms = [float(cp.cuda.get_elapsed_time(origin, event)) for event in completions]
    # The origin event precedes the 50 ms host scheduling guard.
    origin_to_zero_ms = 50.0
    completion_ms = [item - origin_to_zero_ms for item in completion_ms]
    latency_ms = [done - arrival for done, arrival in zip(completion_ms, arrivals_ms)]
    latency = np.asarray(latency_ms, dtype=np.float64)
    window_ms = duration_s * 1000.0
    last_arrival = arrivals_ms[-1] if arrivals_ms else 0.0
    result = {
        "arrival_rate_slots_per_s": rate,
        "duration_s": duration_s,
        "requests": requests,
        "routing": "round_robin_independent_contexts",
        "latency_ms": stats(latency_ms),
        "scheduler_lateness_us": stats(lateness_us),
        "deadline_miss_ratio": {
            "1ms": float(np.mean(latency > 1.0)),
            "2ms": float(np.mean(latency > 2.0)),
            "5ms": float(np.mean(latency > 5.0)),
            "10ms": float(np.mean(latency > 10.0)),
        },
        "completion_span_ms": max(completion_ms, default=0.0),
        "drain_after_last_arrival_ms": max(completion_ms, default=0.0) - last_arrival,
        "raw_latency_ms": latency_ms,
        "raw_scheduler_lateness_us": lateness_us,
        **queue_metrics(arrivals_ms, completion_ms, window_ms),
    }
    del completions
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--replicas", type=parse_ints, default=parse_ints("1,2,4,8"))
    parser.add_argument(
        "--load-fractions", type=parse_floats, default=parse_floats("0.50,0.85,0.95,1.05")
    )
    parser.add_argument("--duration", type=float, default=60.0)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--topology", choices=("4g", "3g", "full"), required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.trial <= 0:
        parser.error("duration/trial must be positive and warmup non-negative")

    properties = cp.cuda.runtime.getDeviceProperties(0)
    name = properties["name"]
    if isinstance(name, bytes):
        name = name.decode("utf-8", errors="replace")
    result = {
        "schema": "nrx-capacity-full-v1",
        "topology": args.topology,
        "trial": args.trial,
        "engine": args.engine,
        "device": {
            "name": name,
            "sm_count": int(properties["multiProcessorCount"]),
            "total_memory_bytes": int(properties["totalGlobalMem"]),
        },
        "batch": {
            "measured": [1],
            "batch_2_4_status": "requires separately built dynamic-batch TensorRT engines",
        },
        "configurations": [],
    }
    for replicas in args.replicas:
        print(f"[NRX-CAPACITY] topology={args.topology} replicas={replicas} init", flush=True)
        free_before, total = cp.cuda.runtime.memGetInfo()
        runtimes = make_runtimes(args.engine, replicas, args.warmup)
        free_resident, _ = cp.cuda.runtime.memGetInfo()
        saturation = saturated(runtimes, args.duration)
        configuration = {
            "replicas": replicas,
            "resident_bytes": int(free_before - free_resident),
            "saturation": saturation,
            "open_loop": [],
        }
        capacity = saturation["throughput_slots_per_s"]
        for fraction in args.load_fractions:
            rate = capacity * fraction
            item = open_loop(runtimes, rate, args.duration)
            item["load_fraction_of_measured_capacity"] = fraction
            configuration["open_loop"].append(item)
            print(
                f"[NRX-CAPACITY] topology={args.topology} replicas={replicas} "
                f"load={fraction:.2f} p99={item['latency_ms']['p99']:.3f}ms "
                f"miss5={item['deadline_miss_ratio']['5ms']:.6f}",
                flush=True,
            )
        if not all(
            bool(cp.all(cp.isfinite(value)).item())
            for runtime in runtimes for value in runtime.outputs.values()
        ):
            raise RuntimeError(f"non-finite output replicas={replicas}")
        configuration["outputs_finite"] = True
        result["configurations"].append(configuration)
        del runtimes
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
    result["pass"] = True
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    temporary = args.output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream)
    os.replace(temporary, args.output)
    print(f"[NRX-CAPACITY] PASS output={args.output}", flush=True)


if __name__ == "__main__":
    main()
