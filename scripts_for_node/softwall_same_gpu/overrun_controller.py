#!/usr/bin/env python3
"""Run mandatory conventional PHY while a timed-out MPS client keeps executing."""

from __future__ import annotations

import argparse
import json
import os
import platform
import select
import socket
import time
from pathlib import Path

from dual_receiver_phy import PairedDualReceiver
from softwall_phy import summary, wait_until


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


class LineSocket:
    def __init__(self, path: str) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)
        self.buffer = b""

    def send(self, value: dict) -> None:
        self.socket.sendall(json.dumps(value).encode() + b"\n")

    def receive(self, timeout_s: float) -> dict | None:
        deadline = time.monotonic() + max(0.0, timeout_s)
        while b"\n" not in self.buffer:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return None
            readable, _, _ = select.select([self.socket], [], [], remaining)
            if not readable:
                return None
            payload = self.socket.recv(65536)
            if not payload:
                raise ConnectionError("overrun worker closed the RPC channel")
            self.buffer += payload
        line, self.buffer = self.buffer.split(b"\n", 1)
        return json.loads(line)

    def close(self) -> None:
        self.socket.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--period-ms", type=float, default=60.0)
    parser.add_argument("--deadline-ms", type=float, default=25.0)
    parser.add_argument("--timeout-ms", type=float, default=5.0)
    parser.add_argument("--drain-guard-ms", type=float, default=2.0)
    parser.add_argument("--fault-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20320001)
    parser.add_argument("--retire-after-timeout", action="store_true")
    parser.add_argument("--retire-before-first-release", action="store_true")
    parser.add_argument("--quiesce-before-first-release", action="store_true")
    parser.add_argument("--lifecycle-ready-file", type=Path)
    parser.add_argument("--lifecycle-start-file", type=Path)
    args = parser.parse_args()
    if min(
        args.iterations, args.period_ms, args.deadline_ms,
        args.timeout_ms, args.fault_every,
    ) <= 0 or args.drain_guard_ms < 0:
        parser.error("invalid timing/count parameter")
    if sum((
        args.retire_after_timeout,
        args.retire_before_first_release,
        args.quiesce_before_first_release,
    )) > 1:
        parser.error("retirement/quiescence modes are mutually exclusive")
    if bool(args.lifecycle_ready_file) != bool(args.lifecycle_start_file):
        parser.error("lifecycle ready/start files must be supplied together")

    receiver = PairedDualReceiver(args.engine, seed=args.seed)
    for _ in range(args.warmup):
        result = receiver.run_conventional()
        if not result[1]:
            raise RuntimeError("conventional warmup failed")
    channel = LineSocket(args.socket)
    period_ns = round(args.period_ms * 1e6)
    deadline_ns = round(args.deadline_ms * 1e6)
    drain_guard_ns = round(args.drain_guard_ms * 1e6)
    outstanding = False
    outstanding_release = None
    endpoint_enabled = True
    endpoint_retired_at_release = None
    stop_response = None
    pre_epoch_stop_ack_ns = None
    quiesce_response = None
    endpoint_quiesced = False
    lifecycle_handshake = None
    records = []
    worker_responses = []

    if args.retire_before_first_release:
        channel.send({"op": "stop"})
        stop_response = channel.receive(2.0)
        pre_epoch_stop_ack_ns = time.perf_counter_ns()
        channel.close()
        endpoint_enabled = False
        endpoint_retired_at_release = -1
    elif args.quiesce_before_first_release:
        channel.send({"op": "quiesce"})
        quiesce_response = channel.receive(2.0)
        if not quiesce_response or not quiesce_response.get("quiesced"):
            raise RuntimeError("GPU worker failed to quiesce before RAN epoch")
        endpoint_enabled = False
        endpoint_quiesced = True

    if args.lifecycle_ready_file is not None:
        args.lifecycle_ready_file.parent.mkdir(parents=True, exist_ok=True)
        args.lifecycle_ready_file.touch()
        handshake_deadline = time.monotonic() + 120.0
        while not args.lifecycle_start_file.is_file():
            if time.monotonic() > handshake_deadline:
                raise TimeoutError("launcher did not confirm worker lifecycle")
            time.sleep(0.001)
        lifecycle_handshake = json.loads(
            args.lifecycle_start_file.read_text(encoding="utf-8")
        )
        expected = "alive" if endpoint_quiesced else "exited"
        if lifecycle_handshake.get("worker_state") != expected:
            raise RuntimeError("launcher worker-state confirmation mismatch")
        if lifecycle_handshake.get("confirmed_ns", 0) > time.perf_counter_ns():
            raise RuntimeError("launcher confirmation time is in the future")

    first_release_ns = time.perf_counter_ns() + 200_000_000

    for index in range(args.iterations):
        release_ns = first_release_ns + index * period_ns
        wait_until(release_ns)
        start_ns = time.perf_counter_ns()

        if outstanding:
            response = channel.receive(0.0)
            if response is not None:
                worker_responses.append({
                    "request_release": outstanding_release,
                    "drained_at_release": index,
                    "response": response,
                })
                outstanding = False
                outstanding_release = None
                if args.retire_after_timeout:
                    channel.send({"op": "stop"})
                    stop_response = channel.receive(2.0)
                    channel.close()
                    endpoint_enabled = False
                    endpoint_retired_at_release = index

        worker_active_before_injection = outstanding

        injected = (
            endpoint_enabled
            and index % args.fault_every == 0
            and not outstanding
        )
        response_before_timeout = None
        timeout_detected = False
        if injected:
            channel.send({"op": "run", "release_index": index})
            outstanding = True
            outstanding_release = index
            response_before_timeout = channel.receive(args.timeout_ms / 1000.0)
            if response_before_timeout is not None:
                worker_responses.append({
                    "request_release": index,
                    "drained_at_release": index,
                    "response": response_before_timeout,
                })
                outstanding = False
                outstanding_release = None
            else:
                timeout_detected = True

        worker_active_during_conventional = outstanding

        conventional = receiver.run_conventional()
        completed_ns = time.perf_counter_ns()
        response_ms = (completed_ns - release_ns) / 1e6

        if outstanding and index + 1 < args.iterations:
            latest_drain_ns = release_ns + period_ns - drain_guard_ns
            remaining_s = max(
                0.0, (latest_drain_ns - time.perf_counter_ns()) / 1e9
            )
            response = channel.receive(remaining_s)
            if response is not None:
                worker_responses.append({
                    "request_release": outstanding_release,
                    "drained_at_release": index,
                    "response": response,
                })
                outstanding = False
                outstanding_release = None
                if args.retire_after_timeout:
                    channel.send({"op": "stop"})
                    stop_response = channel.receive(2.0)
                    channel.close()
                    endpoint_enabled = False
                    endpoint_retired_at_release = index

        records.append({
            "index": index,
            "release_ns": release_ns,
            "start_lateness_ms": (start_ns - release_ns) / 1e6,
            "injected_overrun": injected,
            "worker_active_before_injection": worker_active_before_injection,
            "worker_active_during_conventional": worker_active_during_conventional,
            "timeout_detected": timeout_detected,
            "response_before_timeout": response_before_timeout,
            "conventional_gpu_ms": conventional[0],
            "correct": conventional[1],
            "crc_failures": conventional[2],
            "payload_mismatches": conventional[3],
            "response_ms": response_ms,
            "deadline_miss": response_ms > args.deadline_ms,
        })

    if outstanding:
        response = channel.receive(max(1.0, args.period_ms / 1000.0 * 4))
        if response is not None:
            worker_responses.append({
                "request_release": outstanding_release,
                "drained_at_release": args.iterations,
                "response": response,
            })
            outstanding = False
            if args.retire_after_timeout:
                channel.send({"op": "stop"})
                stop_response = channel.receive(2.0)
                channel.close()
                endpoint_enabled = False
                endpoint_retired_at_release = args.iterations
    if endpoint_enabled or endpoint_quiesced:
        channel.send({"op": "stop"})
        stop_response = channel.receive(2.0)
        channel.close()
        endpoint_retired_at_release = args.iterations

    injected_records = [item for item in records if item["injected_overrun"]]
    active_records = [
        item for item in records if item["worker_active_during_conventional"]
    ]
    result = {
        "schema": "softwall-physical-overrun-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "iterations": args.iterations,
        "period_ms": args.period_ms,
        "deadline_ms": args.deadline_ms,
        "timeout_ms": args.timeout_ms,
        "fault_every": args.fault_every,
        "seed": args.seed,
        "retire_after_timeout": args.retire_after_timeout,
        "retire_before_first_release": args.retire_before_first_release,
        "quiesce_before_first_release": args.quiesce_before_first_release,
        "quiesce_response": quiesce_response,
        "pre_epoch_stop_ack_ns": pre_epoch_stop_ack_ns,
        "gpu_client_live_during_epoch": endpoint_quiesced,
        "lifecycle_handshake": lifecycle_handshake,
        "first_release_ns": first_release_ns,
        "endpoint_retired": (
            endpoint_retired_at_release is not None
            and endpoint_retired_at_release < args.iterations
        ),
        "endpoint_retired_at_release": endpoint_retired_at_release,
        "injected_overruns": len(injected_records),
        "timeouts_detected": sum(x["timeout_detected"] for x in records),
        "responses_before_timeout": sum(
            x["response_before_timeout"] is not None for x in records
        ),
        "responses_drained": len(worker_responses),
        "outstanding_at_end": outstanding,
        "correct_releases": sum(x["correct"] for x in records),
        "deadline_misses": sum(x["deadline_miss"] for x in records),
        "injected_deadline_misses": sum(
            x["deadline_miss"] for x in injected_records
        ),
        "worker_active_releases": len(active_records),
        "worker_active_deadline_misses": sum(
            x["deadline_miss"] for x in active_records
        ),
        "response_ms": summary([x["response_ms"] for x in records]),
        "injected_response_ms": summary(
            [x["response_ms"] for x in injected_records]
        ),
        "worker_active_response_ms": summary(
            [x["response_ms"] for x in active_records]
        ),
        "conventional_gpu_ms": summary(
            [x["conventional_gpu_ms"] for x in records]
        ),
        "injected_conventional_gpu_ms": summary(
            [x["conventional_gpu_ms"] for x in injected_records]
        ),
        "stop_response": stop_response,
        "worker_responses": worker_responses,
        "records": records,
    }
    atomic_json(args.output, result)
    injected_max = result["injected_response_ms"]["max"]
    injected_max_text = "n/a" if injected_max is None else f"{injected_max:.3f}ms"
    print(
        f"[OVERRUN] n={args.iterations} injected={len(injected_records)} "
        f"timeout={result['timeouts_detected']} miss={result['deadline_misses']} "
        f"injected_miss={result['injected_deadline_misses']} "
        f"p99={result['response_ms']['p99']:.3f}ms "
        f"injected_max={injected_max_text}",
        flush=True,
    )


if __name__ == "__main__":
    main()
