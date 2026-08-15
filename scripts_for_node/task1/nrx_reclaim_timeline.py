#!/usr/bin/env python3
"""Open-loop NRx timeline that drives a resident background-AI gate."""

from __future__ import annotations

import argparse
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


def parse_schedule(value):
    phases = []
    for index, item in enumerate(value.split(",")):
        state, duration = item.split(":", 1)
        if state not in {"off", "on"}:
            raise argparse.ArgumentTypeError(f"invalid phase state: {state}")
        seconds = float(duration)
        if seconds <= 0:
            raise argparse.ArgumentTypeError("phase durations must be positive")
        phases.append({
            "index": index,
            "name": f"{index}_{state}",
            "enabled": state == "on",
            "duration_s": seconds,
        })
    if not phases:
        raise argparse.ArgumentTypeError("schedule must not be empty")
    return phases


def phase_at(phases, elapsed_s):
    cursor = 0.0
    for phase in phases:
        if elapsed_s < cursor + phase["duration_s"]:
            return phase
        cursor += phase["duration_s"]
    return phases[-1]


def queue_metrics(arrivals_ms, completions_ms, duration_ms):
    completion_order = sorted(completions_ms)
    completed = 0
    max_outstanding = 0
    for index, arrival in enumerate(arrivals_ms):
        while (
            completed < len(completion_order)
            and completion_order[completed] <= arrival
        ):
            completed += 1
        max_outstanding = max(max_outstanding, index + 1 - completed)
    return {
        "max_outstanding": max_outstanding,
        "backlog_at_window_end": sum(
            completion > duration_ms for completion in completions_ms),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--gate-file", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--rate", type=float, default=900.0)
    parser.add_argument(
        "--schedule", type=parse_schedule,
        default=parse_schedule("off:2,on:2,off:3,on:2,off:3"))
    parser.add_argument("--warmup", type=int, default=100)
    args = parser.parse_args()
    if args.rate <= 0 or args.warmup < 0:
        parser.error("rate must be positive and warmup non-negative")

    gate_path = Path(args.gate_file)
    output_path = Path(args.output)
    gate_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    gate_path.write_text("0", encoding="utf-8")

    runtime = DirectNrx(args.engine)
    runtime.capture_graph()
    for _ in range(args.warmup):
        runtime.launch(use_graph=True)
    runtime.stream.synchronize()

    duration_s = sum(item["duration_s"] for item in args.schedule)
    requests = int(round(args.rate * duration_s))
    completion_events = [cp.cuda.Event() for _ in range(requests)]
    origin = cp.cuda.Event()
    origin.record(runtime.stream)
    origin.synchronize()
    origin_host_ns = time.monotonic_ns()
    global_zero_ns = time.monotonic_ns() + 100_000_000
    transitions = []

    def drive_gate():
        cursor_s = 0.0
        for phase in args.schedule:
            target_ns = global_zero_ns + round(cursor_s * 1e9)
            wait_until(target_ns)
            gate_path.write_text("1" if phase["enabled"] else "0", encoding="utf-8")
            actual_ns = time.monotonic_ns()
            transitions.append({
                "phase": phase["name"],
                "enabled": phase["enabled"],
                "target_ns": target_ns,
                "actual_ns": actual_ns,
                "lateness_us": (actual_ns - target_ns) / 1e3,
            })
            print(
                f"[NRX-RECLAIM] phase={phase['name']} "
                f"gate={'on' if phase['enabled'] else 'off'} "
                f"lateness={(actual_ns-target_ns)/1e3:.3f}us",
                flush=True,
            )
            cursor_s += phase["duration_s"]

    controller = threading.Thread(target=drive_gate, name="gate-controller")
    controller.start()
    arrivals_ms = []
    launch_lateness_us = []
    phase_names = []
    try:
        for index, completion in enumerate(completion_events):
            intended_ns = global_zero_ns + round(index * 1e9 / args.rate)
            wait_until(intended_ns)
            actual_ns = time.monotonic_ns()
            elapsed_s = (intended_ns - global_zero_ns) / 1e9
            arrivals_ms.append(elapsed_s * 1000.0)
            launch_lateness_us.append((actual_ns - intended_ns) / 1e3)
            phase_names.append(phase_at(args.schedule, elapsed_s)["name"])
            with runtime.stream:
                runtime.launch(use_graph=True)
                completion.record()
        runtime.stream.synchronize()
    finally:
        gate_path.write_text("0", encoding="utf-8")
        controller.join(timeout=5.0)
    if controller.is_alive():
        raise RuntimeError("gate controller did not terminate")

    completion_ms = [
        (origin_host_ns + round(float(cp.cuda.get_elapsed_time(origin, event)) * 1e6)
         - global_zero_ns) / 1e6
        for event in completion_events
    ]
    latency_ms = [
        completion - arrival
        for arrival, completion in zip(arrivals_ms, completion_ms)
    ]
    records = [
        {
            "sequence": index + 1,
            "phase": phase,
            "arrival_ms": arrival,
            "completion_ms": completion,
            "latency_ms": latency,
            "launch_lateness_us": lateness,
        }
        for index, (phase, arrival, completion, latency, lateness) in enumerate(
            zip(
                phase_names, arrivals_ms, completion_ms, latency_ms,
                launch_lateness_us))
    ]
    phase_results = []
    for phase in args.schedule:
        selected = [
            item for item in records if item["phase"] == phase["name"]
        ]
        values = [item["latency_ms"] for item in selected]
        phase_results.append({
            **phase,
            "requests": len(selected),
            "latency_ms": stats(values),
            "deadline_miss_ratio": {
                "1ms": float(np.mean(np.asarray(values) > 1.0)),
                "2ms": float(np.mean(np.asarray(values) > 2.0)),
                "5ms": float(np.mean(np.asarray(values) > 5.0)),
                "10ms": float(np.mean(np.asarray(values) > 10.0)),
            },
        })

    result = {
        "experiment": "resident_background_ai_reclaim_timeline",
        "scope": "nrx_compute_queue_and_qwen_mps_no_l1_or_transport",
        "engine": args.engine,
        "rate_slots_per_s": args.rate,
        "duration_s": duration_s,
        "requests": requests,
        "schedule": args.schedule,
        "transitions": transitions,
        "phase_results": phase_results,
        "latency_ms": stats(latency_ms),
        "launch_lateness_us": stats(launch_lateness_us),
        **queue_metrics(arrivals_ms, completion_ms, duration_s * 1000.0),
        "outputs_finite": all(
            bool(cp.all(cp.isfinite(value)).item())
            for value in runtime.outputs.values()),
        "records": records,
    }
    if not result["outputs_finite"]:
        raise RuntimeError("non-finite NRx output")
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        f"[NRX-RECLAIM] done p99={result['latency_ms']['p99']:.3f}ms "
        f"max_outstanding={result['max_outstanding']} output={output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
