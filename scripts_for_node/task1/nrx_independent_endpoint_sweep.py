#!/usr/bin/env python3
"""Open-loop NeuralRx sweep across independent physical GPU endpoints.

This is a compute-capacity and queue-stability gate.  Each endpoint owns a
TensorRT engine/context, CUDA graph, stream, buffers, and physical GPU.  It does
not include L1 or inter-process transport; those are deliberately separate
follow-up gates.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import time
from dataclasses import dataclass

import cupy as cp
import numpy as np

from nrx_trt_direct import DirectNrx, stats


def parse_ints(value):
    result = [int(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive integers")
    return result


def parse_floats(value):
    result = [float(item) for item in value.split(",") if item.strip()]
    if not result or any(item <= 0 for item in result):
        raise argparse.ArgumentTypeError(
            "expected comma-separated positive numbers")
    return result


def wait_until(target_ns):
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


@dataclass
class Endpoint:
    device: int
    runtime: DirectNrx


def device_metadata(device):
    with cp.cuda.Device(device):
        properties = cp.cuda.runtime.getDeviceProperties(device)
        name = properties["name"]
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="replace")
        pci_bus_id = cp.cuda.runtime.deviceGetPCIBusId(device)
        if isinstance(pci_bus_id, bytes):
            pci_bus_id = pci_bus_id.decode("ascii", errors="replace")
        return {
            "logical_device": device,
            "name": name,
            "pci_bus_id": pci_bus_id,
            "total_memory_bytes": int(properties["totalGlobalMem"]),
            "multi_processor_count": int(properties["multiProcessorCount"]),
        }


def make_endpoints(engine, count, warmup_rounds):
    endpoints = []
    for device in range(count):
        with cp.cuda.Device(device):
            runtime = DirectNrx(engine)
            runtime.capture_graph()
            for _ in range(warmup_rounds):
                runtime.launch(use_graph=True)
            runtime.stream.synchronize()
        endpoints.append(Endpoint(device=device, runtime=runtime))
    return endpoints


def make_event(device):
    with cp.cuda.Device(device):
        return cp.cuda.Event()


def synchronize(endpoints):
    for endpoint in endpoints:
        with cp.cuda.Device(endpoint.device):
            endpoint.runtime.stream.synchronize()


def closed_loop(endpoints, total_requests):
    count = len(endpoints)
    rounds = int(math.ceil(total_requests / count))
    actual_total = rounds * count
    assignments = [endpoints[index % count] for index in range(actual_total)]
    events = [
        (endpoint.device, make_event(endpoint.device), make_event(endpoint.device))
        for endpoint in assignments
    ]

    wall_start = time.perf_counter_ns()
    for endpoint, (_, start, end) in zip(assignments, events):
        with cp.cuda.Device(endpoint.device), endpoint.runtime.stream:
            start.record()
            endpoint.runtime.launch(use_graph=True)
            end.record()
    synchronize(endpoints)
    wall_s = (time.perf_counter_ns() - wall_start) / 1e9

    service_ms = []
    for device, start, end in events:
        with cp.cuda.Device(device):
            service_ms.append(float(cp.cuda.get_elapsed_time(start, end)))
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
        while (
            completed < len(completion_order)
            and completion_order[completed] <= arrival
        ):
            completed += 1
        outstanding = index + 1 - completed
        max_outstanding = max(max_outstanding, outstanding)
    return {
        "max_outstanding": max_outstanding,
        "backlog_at_window_end": sum(
            completion > duration_ms for completion in completions_ms),
    }


def make_origins(endpoints):
    origins = {}
    for endpoint in endpoints:
        with cp.cuda.Device(endpoint.device):
            origin = cp.cuda.Event()
            endpoint.runtime.stream.synchronize()
            origin.record(endpoint.runtime.stream)
            origin.synchronize()
            origin_host_ns = time.perf_counter_ns()
        origins[endpoint.device] = (origin, origin_host_ns)
    return origins


def open_loop(endpoints, rate, duration_s):
    count = len(endpoints)
    requests = int(round(rate * duration_s))
    assignments = [endpoints[index % count] for index in range(requests)]
    completion_events = [
        make_event(endpoint.device) for endpoint in assignments
    ]
    origins = make_origins(endpoints)
    global_zero_ns = time.perf_counter_ns() + 50_000_000
    intended_arrival_ms = []
    actual_lateness_us = []

    for index, (endpoint, completion) in enumerate(
        zip(assignments, completion_events)
    ):
        intended_ns = global_zero_ns + round(index * 1e9 / rate)
        wait_until(intended_ns)
        now_ns = time.perf_counter_ns()
        intended_arrival_ms.append((intended_ns - global_zero_ns) / 1e6)
        actual_lateness_us.append((now_ns - intended_ns) / 1e3)
        with cp.cuda.Device(endpoint.device), endpoint.runtime.stream:
            endpoint.runtime.launch(use_graph=True)
            completion.record()

    synchronize(endpoints)
    completion_ms = []
    for endpoint, completion in zip(assignments, completion_events):
        origin, origin_host_ns = origins[endpoint.device]
        with cp.cuda.Device(endpoint.device):
            elapsed_ms = float(cp.cuda.get_elapsed_time(origin, completion))
        completion_host_ns = origin_host_ns + round(elapsed_ms * 1e6)
        completion_ms.append((completion_host_ns - global_zero_ns) / 1e6)

    latency_ms = [
        completion - arrival
        for completion, arrival in zip(completion_ms, intended_arrival_ms)
    ]
    window_ms = duration_s * 1000.0
    last_arrival = intended_arrival_ms[-1] if intended_arrival_ms else 0.0
    per_endpoint_requests = {
        str(endpoint.device): sum(
            assigned.device == endpoint.device for assigned in assignments)
        for endpoint in endpoints
    }
    return {
        "arrival_rate_slots_per_s": rate,
        "duration_s": duration_s,
        "requests": requests,
        "routing": "round_robin_independent_physical_gpu",
        "per_endpoint_requests": per_endpoint_requests,
        "latency_ms": stats(latency_ms),
        "scheduler_lateness_us": stats(actual_lateness_us),
        "deadline_miss_ratio": {
            "1ms": float(np.mean(np.asarray(latency_ms) > 1.0)),
            "2ms": float(np.mean(np.asarray(latency_ms) > 2.0)),
            "5ms": float(np.mean(np.asarray(latency_ms) > 5.0)),
            "10ms": float(np.mean(np.asarray(latency_ms) > 10.0)),
        },
        "completion_span_ms": max(completion_ms, default=0.0),
        "drain_after_last_arrival_ms": (
            max(completion_ms, default=0.0) - last_arrival),
        **queue_depth_metrics(
            intended_arrival_ms, completion_ms, window_ms),
    }


def outputs_finite(endpoints):
    for endpoint in endpoints:
        with cp.cuda.Device(endpoint.device):
            for value in endpoint.runtime.outputs.values():
                if not bool(cp.all(cp.isfinite(value)).item()):
                    return False
    return True


def release_endpoints(endpoints):
    devices = [endpoint.device for endpoint in endpoints]
    endpoints.clear()
    gc.collect()
    for device in devices:
        with cp.cuda.Device(device):
            cp.get_default_memory_pool().free_all_blocks()
            cp.get_default_pinned_memory_pool().free_all_blocks()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument(
        "--endpoint-counts", type=parse_ints, default=parse_ints("1,2,3"))
    parser.add_argument(
        "--rates", type=parse_floats,
        default=parse_floats("750,1000,1250,1750,2250,2750,3250,3500"))
    parser.add_argument("--closed-loop-requests", type=int, default=6000)
    parser.add_argument("--open-loop-duration", type=float, default=3.0)
    parser.add_argument("--warmup-rounds", type=int, default=20)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.closed_loop_requests <= 0 or args.open_loop_duration <= 0:
        parser.error("request count and duration must be positive")

    visible_devices = cp.cuda.runtime.getDeviceCount()
    if max(args.endpoint_counts) > visible_devices:
        parser.error(
            f"requested {max(args.endpoint_counts)} endpoints but only "
            f"{visible_devices} CUDA devices are visible")

    result = {
        "experiment": "independent_physical_gpu_nrx_endpoint_scaling",
        "scope": "nrx_compute_and_queue_only_no_l1_or_transport",
        "engine": args.engine,
        "visible_devices": [
            device_metadata(device) for device in range(visible_devices)
        ],
        "endpoint_counts": args.endpoint_counts,
        "arrival_rates_slots_per_s": args.rates,
        "closed_loop_requests_target": args.closed_loop_requests,
        "open_loop_duration_s": args.open_loop_duration,
        "configurations": [],
    }

    for count in args.endpoint_counts:
        print(
            f"[INDEPENDENT-NRX] endpoints={count} initializing", flush=True)
        endpoints = make_endpoints(args.engine, count, args.warmup_rounds)
        configuration = {
            "endpoints": count,
            "devices": [device_metadata(item.device) for item in endpoints],
            "closed_loop": closed_loop(
                endpoints, args.closed_loop_requests),
            "open_loop": [],
        }
        print(
            f"[INDEPENDENT-NRX] endpoints={count} "
            f"throughput={configuration['closed_loop']['throughput_slots_per_s']:.3f} "
            "slots/s",
            flush=True,
        )
        for rate in args.rates:
            item = open_loop(endpoints, rate, args.open_loop_duration)
            configuration["open_loop"].append(item)
            print(
                f"[INDEPENDENT-NRX] endpoints={count} rate={rate:.1f} "
                f"p99={item['latency_ms']['p99']:.3f}ms "
                f"backlog={item['backlog_at_window_end']} "
                f"max_outstanding={item['max_outstanding']}",
                flush=True,
            )
        configuration["outputs_finite"] = outputs_finite(endpoints)
        if not configuration["outputs_finite"]:
            raise RuntimeError(
                f"non-finite output with {count} independent endpoints")
        result["configurations"].append(configuration)
        release_endpoints(endpoints)

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    print(f"[INDEPENDENT-NRX] OK output={args.output}", flush=True)


if __name__ == "__main__":
    main()
