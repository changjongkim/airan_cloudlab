#!/usr/bin/env python3
"""Actual-GPU DART-F routing sweep across heterogeneous resident NRx endpoints.

Each visible CUDA device owns a TensorRT context, stream, buffers, and graph.
This gate measures NRx compute/queue only; L1 and transport are separate gates.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import time
from dataclasses import dataclass, field

import cupy as cp
import numpy as np

from nrx_trt_direct import DirectNrx, measure, stats


def parse_csv(value, conversion):
    result = [conversion(item) for item in value.split(",") if item.strip()]
    if not result:
        raise argparse.ArgumentTypeError("empty comma-separated value")
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
    service_p99_ns: int
    service_mean_ns: int
    predicted_tail_ns: int = 0
    outstanding: collections.deque = field(default_factory=collections.deque)

    def refresh(self):
        while self.outstanding:
            try:
                # CuPy 13 exposes the non-blocking CUDA event query as the
                # ``done`` property (older wrappers sometimes used query()).
                complete = bool(self.outstanding[0].done)
            except cp.cuda.runtime.CUDARuntimeError:
                complete = False
            if not complete:
                break
            self.outstanding.popleft()


def device_metadata(device):
    with cp.cuda.Device(device):
        props = cp.cuda.runtime.getDeviceProperties(device)
        name = props["name"]
        if isinstance(name, bytes):
            name = name.decode(errors="replace")
        bus = cp.cuda.runtime.deviceGetPCIBusId(device)
        if isinstance(bus, bytes):
            bus = bus.decode(errors="replace")
        return {
            "logical_device": device,
            "name": name,
            "pci_bus_id": bus,
            "sm_count": int(props["multiProcessorCount"]),
            "memory_bytes": int(props["totalGlobalMem"]),
        }


def make_endpoints(engine, calibration_requests, warmup):
    endpoints = []
    for device in range(cp.cuda.runtime.getDeviceCount()):
        with cp.cuda.Device(device):
            runtime = DirectNrx(engine)
            runtime.capture_graph()
            for _ in range(warmup):
                runtime.launch(use_graph=True)
            runtime.stream.synchronize()
            calibration = measure(runtime, calibration_requests, use_graph=True)
        endpoints.append(Endpoint(
            device=device,
            runtime=runtime,
            service_p99_ns=round(calibration["gpu_ms"]["p99"] * 1_000_000),
            service_mean_ns=round(calibration["gpu_ms"]["mean"] * 1_000_000),
        ))
    return endpoints


def synchronize(endpoints):
    for endpoint in endpoints:
        with cp.cuda.Device(endpoint.device):
            endpoint.runtime.stream.synchronize()


def origins(endpoints):
    result = {}
    for endpoint in endpoints:
        with cp.cuda.Device(endpoint.device):
            endpoint.runtime.stream.synchronize()
            event = cp.cuda.Event()
            event.record(endpoint.runtime.stream)
            event.synchronize()
            result[endpoint.device] = (event, time.perf_counter_ns())
    return result


def choose_endpoint(endpoints, policy, index, intended_ns):
    if policy == "round_robin":
        return endpoints[index % len(endpoints)]
    for endpoint in endpoints:
        endpoint.refresh()
        # A conservative predicted tail is useful while requests remain in
        # flight, but once the CUDA event queue drains it must be reconciled
        # with observed reality.  Otherwise early completions leave phantom
        # future work and permanently bias routing away from that endpoint.
        if not endpoint.outstanding:
            endpoint.predicted_tail_ns = intended_ns
    if policy == "shortest_queue":
        return min(
            endpoints,
            key=lambda item: (len(item.outstanding), item.device),
        )
    if policy == "predicted_finish":
        return min(
            endpoints,
            key=lambda item: (
                max(intended_ns, item.predicted_tail_ns) + item.service_p99_ns,
                item.device,
            ),
        )
    if policy == "hybrid_finish":
        # v1 exploratory split estimator: live/mean service ranks endpoints;
        # the full DART admission path still uses the separate p99 feasibility
        # profile.  This sweep isolates only the routing half of that design.
        return min(
            endpoints,
            key=lambda item: (
                max(intended_ns, item.predicted_tail_ns) + item.service_mean_ns,
                item.device,
            ),
        )
    raise ValueError(policy)


def queue_metrics(arrivals_ms, completions_ms, duration_ms):
    ordered = sorted(completions_ms)
    complete = 0
    maximum = 0
    for index, arrival in enumerate(arrivals_ms):
        while complete < len(ordered) and ordered[complete] <= arrival:
            complete += 1
        maximum = max(maximum, index + 1 - complete)
    return {
        "max_outstanding": maximum,
        "backlog_at_window_end": sum(item > duration_ms for item in completions_ms),
    }


def run_open_loop(endpoints, policy, rate, duration_s):
    synchronize(endpoints)
    for endpoint in endpoints:
        endpoint.predicted_tail_ns = 0
        endpoint.outstanding.clear()
    reference = origins(endpoints)
    count = round(rate * duration_s)
    zero_ns = time.perf_counter_ns() + 50_000_000
    arrivals_ms = []
    lateness_us = []
    assignments = []
    completions = []
    queue_at_submit = []

    for index in range(count):
        intended_ns = zero_ns + round(index * 1e9 / rate)
        wait_until(intended_ns)
        now_ns = time.perf_counter_ns()
        endpoint = choose_endpoint(endpoints, policy, index, intended_ns)
        queue_at_submit.append(len(endpoint.outstanding))
        with cp.cuda.Device(endpoint.device), endpoint.runtime.stream:
            # CUDA events are device-owned resources.  Constructing this
            # outside the selected endpoint's device context makes recording
            # fail when the scheduler routes to another GPU.
            completion = cp.cuda.Event()
            endpoint.runtime.launch(use_graph=True)
            completion.record()
        endpoint.outstanding.append(completion)
        predicted_service_ns = (
            endpoint.service_mean_ns
            if policy == "hybrid_finish"
            else endpoint.service_p99_ns
        )
        endpoint.predicted_tail_ns = (
            max(intended_ns, endpoint.predicted_tail_ns) + predicted_service_ns
        )
        assignments.append(endpoint.device)
        completions.append((endpoint.device, completion))
        arrivals_ms.append((intended_ns - zero_ns) / 1e6)
        lateness_us.append((now_ns - intended_ns) / 1e3)

    synchronize(endpoints)
    completion_ms = []
    for device, completion in completions:
        origin, host_ns = reference[device]
        with cp.cuda.Device(device):
            elapsed_ms = float(cp.cuda.get_elapsed_time(origin, completion))
        completion_ms.append((host_ns + round(elapsed_ms * 1e6) - zero_ns) / 1e6)
    latency_ms = [
        completion - arrival
        for completion, arrival in zip(completion_ms, arrivals_ms)
    ]
    result = {
        "policy": policy,
        "rate_slots_s": rate,
        "duration_s": duration_s,
        "requests": count,
        "per_endpoint_requests": {
            str(endpoint.device): assignments.count(endpoint.device)
            for endpoint in endpoints
        },
        "latency_ms": stats(latency_ms),
        "scheduler_lateness_us": stats(lateness_us),
        "queue_at_submit": stats(queue_at_submit),
        "deadline_miss_ratio": {
            deadline: float(np.mean(np.asarray(latency_ms) > deadline))
            for deadline in (1, 2, 5, 10)
        },
        "raw": {
            "arrival_ms": arrivals_ms,
            "completion_ms": completion_ms,
            "latency_ms": latency_ms,
            "scheduler_lateness_us": lateness_us,
            "assignment_device": assignments,
            "queue_at_submit": queue_at_submit,
        },
        **queue_metrics(arrivals_ms, completion_ms, duration_s * 1000),
    }
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--policies",
        type=lambda value: parse_csv(value, str),
        default=parse_csv("round_robin,shortest_queue,predicted_finish", str),
    )
    parser.add_argument(
        "--rates",
        type=lambda value: parse_csv(value, float),
        default=parse_csv("1500,2000,2500,3000", float),
    )
    parser.add_argument("--duration", type=float, default=2.0)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--calibration-requests", type=int, default=300)
    args = parser.parse_args()
    if args.duration <= 0 or args.warmup < 0 or args.calibration_requests <= 0:
        parser.error("invalid duration/warmup/calibration count")
    allowed = {
        "round_robin", "shortest_queue", "predicted_finish", "hybrid_finish"
    }
    if not set(args.policies) <= allowed:
        parser.error(f"policies must be within {sorted(allowed)}")

    endpoints = make_endpoints(args.engine, args.calibration_requests, args.warmup)
    if len(endpoints) < 2:
        parser.error("DART routing sweep requires at least two visible endpoints")
    result = {
        "schema": "dart-nrx-policy-hardware-v0",
        "scope": "actual NRx compute/queue; no L1, transport, fallback, or background",
        "engine": args.engine,
        "endpoints": [{
            **device_metadata(endpoint.device),
            "service_mean_ms": endpoint.service_mean_ns / 1e6,
            "service_p99_ms": endpoint.service_p99_ns / 1e6,
        } for endpoint in endpoints],
        "policies": args.policies,
        "rates": args.rates,
        "runs": [],
    }
    for policy in args.policies:
        for rate in args.rates:
            item = run_open_loop(endpoints, policy, rate, args.duration)
            result["runs"].append(item)
            print(
                f"[DART-NRX] policy={policy} rate={rate:.0f} "
                f"p99={item['latency_ms']['p99']:.3f}ms "
                f"miss5={item['deadline_miss_ratio'][5]:.6f} "
                f"assign={item['per_endpoint_requests']}",
                flush=True,
            )
    integrity = True
    for endpoint in endpoints:
        with cp.cuda.Device(endpoint.device):
            integrity = integrity and all(
                bool(cp.all(cp.isfinite(value)).item())
                for value in endpoint.runtime.outputs.values()
            )
    result["outputs_finite"] = integrity
    if not integrity:
        raise RuntimeError("non-finite NRx output")
    output = os.path.abspath(args.output)
    os.makedirs(os.path.dirname(output), exist_ok=True)
    temporary = output + ".tmp"
    with open(temporary, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    os.replace(temporary, output)
    print(f"[DART-NRX] OK output={output}", flush=True)


if __name__ == "__main__":
    main()
