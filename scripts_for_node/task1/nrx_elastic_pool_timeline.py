#!/usr/bin/env python3
"""Burst-trace experiment for reclaiming a resident NRx endpoint.

Endpoint 0 is the always-on primary. Endpoint 1 shares a physical GPU with a
resident, cooperatively gated background model. Policies determine whether the
second endpoint is used and whether the background model yields during bursts.
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import threading
import time
from pathlib import Path

import cupy as cp
import numpy as np

from nrx_trt_direct import DirectNrx, stats


def wait_until(target_ns):
    while True:
        remaining = target_ns - time.monotonic_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


def parse_trace(value):
    phases = []
    for index, item in enumerate(value.split(",")):
        name, rate, duration = item.split(":", 2)
        rate = float(rate)
        duration = float(duration)
        if rate <= 0 or duration <= 0:
            raise argparse.ArgumentTypeError("rate and duration must be positive")
        phases.append({
            "index": index,
            "name": f"{index}_{name}",
            "kind": name,
            "rate": rate,
            "duration_s": duration,
        })
    if not phases:
        raise argparse.ArgumentTypeError("trace must not be empty")
    return phases


def set_gate(path, enabled):
    path.write_text("1" if enabled else "0", encoding="utf-8")


def read_state(path):
    try:
        return path.read_text(encoding="utf-8").split()[0]
    except (FileNotFoundError, IndexError):
        return "missing"


def wait_state(path, expected, timeout_s):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if read_state(path) == expected:
            return time.monotonic_ns()
        time.sleep(0.00025)
    raise TimeoutError(f"Qwen did not reach {expected}: last={read_state(path)}")


def make_runtime(engine, device, warmup):
    with cp.cuda.Device(device):
        runtime = DirectNrx(engine)
        runtime.capture_graph()
        for _ in range(warmup):
            runtime.launch(use_graph=True)
        runtime.stream.synchronize()
    return runtime


def make_event(device):
    with cp.cuda.Device(device):
        return cp.cuda.Event()


def queue_metrics(arrivals_ms, completions_ms, duration_ms):
    ordered = sorted(completions_ms)
    completed = 0
    maximum = 0
    for index, arrival in enumerate(arrivals_ms):
        while completed < len(ordered) and ordered[completed] <= arrival:
            completed += 1
        maximum = max(maximum, index + 1 - completed)
    return {
        "max_outstanding": maximum,
        "backlog_at_window_end": sum(x > duration_ms for x in completions_ms),
    }


def deadline_ratios(values):
    array = np.asarray(values)
    return {
        f"{threshold}ms": float(np.mean(array > threshold))
        for threshold in (1, 2, 5, 10, 100)
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--gate-file", required=True)
    parser.add_argument("--qwen-state-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--policy",
        choices=(
            "static_one", "dedicated_two", "naive_share",
            "elastic_reclaim", "adaptive_reclaim",
        ),
        required=True,
    )
    parser.add_argument(
        "--trace", type=parse_trace,
        default=parse_trace("low:800:3,burst:1600:3,low:800:4"),
    )
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--qwen-idle-timeout", type=float, default=2.0)
    parser.add_argument("--adaptive-high-rate", type=float, default=1200.0)
    parser.add_argument("--adaptive-low-rate", type=float, default=1000.0)
    parser.add_argument("--adaptive-resume-ms", type=float, default=200.0)
    parser.add_argument("--primary-device", type=int, default=0)
    parser.add_argument("--spare-device", type=int, default=1)
    args = parser.parse_args()

    gate_path = Path(args.gate_file)
    state_path = Path(args.qwen_state_file)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    endpoint_count = 1 if args.policy == "static_one" else 2
    device_ids = [args.primary_device]
    if endpoint_count == 2:
        if args.spare_device == args.primary_device:
            parser.error("primary and spare CUDA devices must differ")
        device_ids.append(args.spare_device)
    visible_count = cp.cuda.runtime.getDeviceCount()
    if any(device < 0 or device >= visible_count for device in device_ids):
        parser.error(
            f"device mapping {device_ids} is invalid for {visible_count} devices")
    endpoints = [make_runtime(args.engine, device, args.warmup)
                 for device in device_ids]
    origins = {}
    for endpoint_index, (device, endpoint) in enumerate(
            zip(device_ids, endpoints)):
        with cp.cuda.Device(device):
            origin = cp.cuda.Event()
            origin.record(endpoint.stream)
            origin.synchronize()
            origins[endpoint_index] = (origin, time.monotonic_ns())

    start_ns = time.monotonic_ns() + 100_000_000
    phase_starts = []
    cursor_ns = start_ns
    request_plan = []
    for phase in args.trace:
        phase_starts.append((cursor_ns, phase))
        count = int(round(phase["rate"] * phase["duration_s"]))
        for index in range(count):
            request_plan.append((
                cursor_ns + round(index * 1e9 / phase["rate"]), phase))
        cursor_ns += round(phase["duration_s"] * 1e9)
    end_ns = cursor_ns

    active_endpoints = 1
    active_lock = threading.Lock()
    transitions = []
    controller_error = []

    def transition_controller():
        nonlocal active_endpoints
        try:
            if args.policy == "adaptive_reclaim":
                wait_until(start_ns)
                set_gate(gate_path, True)
                transitions.append({
                    "event": "initial_background_enable",
                    "target_ns": start_ns,
                    "actual_ns": time.monotonic_ns(),
                    "active_endpoints": 1,
                })
                return
            for target_ns, phase in phase_starts:
                wait_until(target_ns)
                is_burst = phase["kind"] == "burst"
                before_ns = time.monotonic_ns()
                qwen_enabled = args.policy in ("static_one", "naive_share")
                desired_endpoints = endpoint_count
                if args.policy == "elastic_reclaim":
                    if is_burst:
                        set_gate(gate_path, False)
                        idle_ns = wait_state(
                            state_path, "idle", args.qwen_idle_timeout)
                        with active_lock:
                            active_endpoints = 2
                        qwen_enabled = False
                    else:
                        with active_lock:
                            active_endpoints = 1
                        if len(endpoints) > 1:
                            with cp.cuda.Device(device_ids[1]):
                                endpoints[1].stream.synchronize()
                        set_gate(gate_path, True)
                        idle_ns = None
                        qwen_enabled = True
                    desired_endpoints = 2 if is_burst else 1
                elif args.policy == "static_one":
                    set_gate(gate_path, True)
                    with active_lock:
                        active_endpoints = 1
                    desired_endpoints = 1
                    idle_ns = None
                elif args.policy == "dedicated_two":
                    set_gate(gate_path, False)
                    wait_state(state_path, "idle", args.qwen_idle_timeout)
                    with active_lock:
                        active_endpoints = 2
                    qwen_enabled = False
                    desired_endpoints = 2
                    idle_ns = time.monotonic_ns()
                else:
                    set_gate(gate_path, True)
                    with active_lock:
                        active_endpoints = 2 if is_burst else 1
                    qwen_enabled = True
                    desired_endpoints = 2 if is_burst else 1
                    idle_ns = None
                transitions.append({
                    "phase": phase["name"],
                    "kind": phase["kind"],
                    "target_ns": target_ns,
                    "control_start_ns": before_ns,
                    "ready_ns": time.monotonic_ns(),
                    "qwen_idle_ns": idle_ns,
                    "qwen_enabled": qwen_enabled,
                    "active_endpoints": desired_endpoints,
                })
        except BaseException as error:
            controller_error.append(repr(error))

    controller = threading.Thread(target=transition_controller, daemon=True)
    controller.start()
    events = []
    arrivals_ms = []
    actual_lateness_us = []
    phase_names = []
    assignments = []
    route_index = 0
    recent_arrivals = deque(maxlen=16)
    reclaim_pending = False
    reclaim_request = None
    low_since_ns = None
    for intended_ns, phase in request_plan:
        wait_until(intended_ns)
        actual_ns = time.monotonic_ns()
        if args.policy == "adaptive_reclaim":
            recent_arrivals.append(actual_ns)
            observed_rate = 0.0
            if len(recent_arrivals) >= 4:
                observed_rate = ((len(recent_arrivals) - 1) * 1e9 /
                                 (recent_arrivals[-1] - recent_arrivals[0]))
            with active_lock:
                adaptive_count = active_endpoints
            if (
                adaptive_count == 1
                and not reclaim_pending
                and observed_rate >= args.adaptive_high_rate
            ):
                set_gate(gate_path, False)
                reclaim_pending = True
                reclaim_request = {
                    "event": "reclaim_request",
                    "phase": phase["name"],
                    "kind": phase["kind"],
                    "detect_ns": actual_ns,
                    "observed_rate": observed_rate,
                    "active_endpoints": 1,
                }
                transitions.append(reclaim_request)
            if reclaim_pending and read_state(state_path) == "idle":
                ready_ns = time.monotonic_ns()
                with active_lock:
                    active_endpoints = 2
                reclaim_request["ready_ns"] = ready_ns
                reclaim_request["activation_ms"] = (
                    ready_ns - reclaim_request["detect_ns"]) / 1e6
                reclaim_request["active_endpoints"] = 2
                reclaim_pending = False
                low_since_ns = None
                adaptive_count = 2
            if adaptive_count == 2:
                if observed_rate and observed_rate <= args.adaptive_low_rate:
                    if low_since_ns is None:
                        low_since_ns = actual_ns
                    elif (actual_ns - low_since_ns) / 1e6 >= args.adaptive_resume_ms:
                        with active_lock:
                            active_endpoints = 1
                        with cp.cuda.Device(device_ids[1]):
                            endpoints[1].stream.synchronize()
                        set_gate(gate_path, True)
                        transitions.append({
                            "event": "background_resume",
                            "phase": phase["name"],
                            "kind": phase["kind"],
                            "actual_ns": time.monotonic_ns(),
                            "observed_rate": observed_rate,
                            "active_endpoints": 1,
                        })
                        low_since_ns = None
                else:
                    low_since_ns = None
        with active_lock:
            count = active_endpoints
        endpoint_index = 0 if count == 1 else route_index % count
        route_index += 1
        device = device_ids[endpoint_index]
        completion = make_event(device)
        endpoint = endpoints[endpoint_index]
        with cp.cuda.Device(device), endpoint.stream:
            endpoint.launch(use_graph=True)
            completion.record()
        events.append((endpoint_index, device, completion))
        arrivals_ms.append((intended_ns - start_ns) / 1e6)
        actual_lateness_us.append((actual_ns - intended_ns) / 1e3)
        phase_names.append(phase["name"])
        assignments.append(endpoint_index)

    controller.join(timeout=args.qwen_idle_timeout + 2.0)
    if controller.is_alive() or controller_error:
        raise RuntimeError(f"controller failed: {controller_error}")
    for device, endpoint in zip(device_ids, endpoints):
        with cp.cuda.Device(device):
            endpoint.stream.synchronize()

    completions_ms = []
    for endpoint_index, device, event in events:
        origin, origin_host_ns = origins[endpoint_index]
        with cp.cuda.Device(device):
            elapsed_ms = float(cp.cuda.get_elapsed_time(origin, event))
        completions_ms.append(
            (origin_host_ns + round(elapsed_ms * 1e6) - start_ns) / 1e6)
    latencies_ms = [c - a for a, c in zip(arrivals_ms, completions_ms)]
    records = [{
        "sequence": index + 1,
        "phase": phase,
        "endpoint": endpoint,
        "arrival_ms": arrival,
        "completion_ms": completion,
        "latency_ms": latency,
        "launch_lateness_us": lateness,
    } for index, (phase, endpoint, arrival, completion, latency, lateness) in enumerate(
        zip(phase_names, assignments, arrivals_ms, completions_ms,
            latencies_ms, actual_lateness_us))]
    phase_results = []
    for phase in args.trace:
        selected = [x for x in records if x["phase"] == phase["name"]]
        values = [x["latency_ms"] for x in selected]
        phase_results.append({
            **phase,
            "requests": len(selected),
            "per_endpoint_requests": {
                str(device): sum(x["endpoint"] == device for x in selected)
                for device in range(endpoint_count)
            },
            "latency_ms": stats(values),
            "deadline_miss_ratio": deadline_ratios(values),
        })
    duration_ms = (end_ns - start_ns) / 1e6
    outputs_finite = True
    for device, endpoint in zip(device_ids, endpoints):
        with cp.cuda.Device(device):
            outputs_finite = outputs_finite and all(
                bool(cp.all(cp.isfinite(value)).item())
                for value in endpoint.outputs.values())
    result = {
        "experiment": "drain_free_resident_nrx_endpoint_reclaim",
        "scope": "nrx_compute_queue_and_qwen_utility_no_l1_or_transport",
        "policy": args.policy,
        "trace": args.trace,
        "duration_s": duration_ms / 1000.0,
        "requests": len(records),
        "endpoint_count": endpoint_count,
        "cuda_device_mapping": {
            "primary": args.primary_device,
            "spare": args.spare_device if endpoint_count == 2 else None,
        },
        "transitions": transitions,
        "phase_results": phase_results,
        "latency_ms": stats(latencies_ms),
        "scheduler_lateness_us": stats(actual_lateness_us),
        **queue_metrics(arrivals_ms, completions_ms, duration_ms),
        "outputs_finite": outputs_finite,
        "records": records,
    }
    if not result["outputs_finite"]:
        raise RuntimeError("non-finite NRx output")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[ELASTIC-NRX] policy={args.policy} p99={result['latency_ms']['p99']:.3f}ms "
        f"max_outstanding={result['max_outstanding']} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
