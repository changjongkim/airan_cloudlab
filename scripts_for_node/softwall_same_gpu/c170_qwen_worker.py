#!/usr/bin/env python3
"""Actual Qwen worker with a worker-acceptance gate for C170."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import torch

from c159_q2_classes import CLASS_BOUNDS_MS
from c159_q2_qwen_worker import TraceQwenPrefill, atomic_json, parse_lengths, summarize


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--allowed-context-lengths", type=parse_lengths, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-per-length", type=int, default=3)
    parser.add_argument("--socket", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup_per_length < 0:
        parser.error("batch size must be positive and warmup count nonnegative")
    if any(length not in CLASS_BOUNDS_MS for length in args.allowed_context_lengths):
        parser.error("every allowed length must have a frozen C159 class bound")

    torch.cuda.set_device(0)
    unit = TraceQwenPrefill(args.model, args.allowed_context_lengths, args.batch_size)
    memory_after_load = unit.memory_mib()
    warmup = []
    for length in args.allowed_context_lengths:
        for _ in range(args.warmup_per_length):
            warmup.append({"context_length": length, "gpu_ms": unit.run(length)})
    memory_after_warmup = unit.memory_mib()
    torch.cuda.empty_cache()
    memory_after_empty_cache = unit.memory_mib()

    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    socket_path.unlink(missing_ok=True)
    records: list[dict] = []
    rejected: list[dict] = []
    launch_guard_rejections: list[dict] = []
    started = time.perf_counter()
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(socket_path))
    server.listen(1)
    print(f"[C170-PADDED-QWEN] rpc ready {socket_path}", flush=True)
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb", buffering=0) as channel:
            while True:
                line = channel.readline()
                if not line:
                    break
                request = None
                try:
                    request = json.loads(line)
                    if request.get("op") == "stop":
                        channel.write(b'{"ok":true,"stopped":true}\n')
                        break
                    if request.get("op") != "run":
                        raise ValueError("unknown operation")
                    context_length = int(request["context_length"])
                    request_id = str(request["request_id"])
                    latest_start_ns = int(request["latest_start_ns"])
                    not_before_ns = int(request["not_before_ns"])
                    while True:
                        remaining_ns = not_before_ns - time.perf_counter_ns()
                        if remaining_ns <= 0:
                            break
                        if remaining_ns > 200_000:
                            time.sleep((remaining_ns - 100_000) / 1e9)
                    accepted_ns = time.perf_counter_ns()
                    if accepted_ns > latest_start_ns:
                        row = {
                            "request_id": request_id,
                            "context_length": context_length,
                            "accepted_ns": accepted_ns,
                            "latest_start_ns": latest_start_ns,
                            "lateness_ms": (accepted_ns - latest_start_ns) / 1e6,
                            "reason": "latest_start_expired_before_gpu_launch",
                        }
                        launch_guard_rejections.append(row)
                        channel.write(json.dumps({
                            "ok": True, "launched": False, **row,
                        }).encode() + b"\n")
                        continue
                    model_gpu_ms = unit.run(context_length)
                    bound_ms = float(CLASS_BOUNDS_MS[context_length])
                    completed_ns = time.perf_counter_ns()
                    record = {
                        "request_id": request_id,
                        "context_length": context_length,
                        "latest_start_ns": latest_start_ns,
                        "not_before_ns": not_before_ns,
                        "accepted_ns": accepted_ns,
                        "completed_ns": completed_ns,
                        "gpu_ms": model_gpu_ms,
                        "declared_bound_ms": bound_ms,
                        "worker_transaction_ms": (
                            completed_ns - accepted_ns
                        ) / 1e6,
                    }
                    records.append(record)
                    response = {
                        "ok": True, "launched": True,
                        "unit": len(records), **record,
                    }
                except (KeyError, TypeError, ValueError) as error:
                    response = {"ok": False, "error": str(error)}
                    rejected.append({
                        "received_ns": time.perf_counter_ns(),
                        "request": request if isinstance(request, dict) else None,
                        "error": str(error),
                    })
                channel.write(json.dumps(response).encode() + b"\n")
    finally:
        result = {
            "schema": "softwall-c170-qwen-v1",
            "analysis_role": (
                "Actual Qwen execution only; the C170 coordinator accounts for "
                "and pads the complete acceptance-to-observation transaction."
            ),
            "model": args.model,
            "allowed_context_lengths": args.allowed_context_lengths,
            "class_bounds_ms": CLASS_BOUNDS_MS,
            "batch_size": args.batch_size,
            "warmup_per_length": args.warmup_per_length,
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "mps_active_thread_percentage": os.environ.get(
                "CUDA_MPS_ACTIVE_THREAD_PERCENTAGE"
            ),
            "mps_client_priority": os.environ.get("CUDA_MPS_CLIENT_PRIORITY"),
            "memory_mib": {
                "after_load": memory_after_load,
                "after_warmup": memory_after_warmup,
                "after_empty_cache": memory_after_empty_cache,
                "at_exit": unit.memory_mib(),
            },
            "wall_s": time.perf_counter() - started,
            "warmup": warmup,
            "model_gpu_ms": summarize([row["gpu_ms"] for row in records]),
            "transaction_ms": summarize([
                row["worker_transaction_ms"] for row in records
            ]),
            "completed_units": len(records),
            "launch_guard_rejections": launch_guard_rejections,
            "rejected": rejected,
            "records": records,
        }
        atomic_json(args.output, result)
        server.close()
        socket_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
