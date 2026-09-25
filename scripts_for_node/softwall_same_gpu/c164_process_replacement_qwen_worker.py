#!/usr/bin/env python3.11
"""One-epoch Qwen worker used as the victim of safe MPS replacement."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import socket
import time
from pathlib import Path

import torch

from c161_phase2_qwen_worker import TraceQwenPrefill, parse_lengths


def durable_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n"); handle.flush(); os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try: os.fsync(descriptor)
    finally: os.close(descriptor)


def identity(request: dict) -> dict:
    value = {key: request[key] for key in (
        "token", "lifecycle_epoch", "payload_sha256", "worker_epoch"
    )}
    value["token"] = str(value["token"])
    value["lifecycle_epoch"] = int(value["lifecycle_epoch"])
    value["payload_sha256"] = str(value["payload_sha256"])
    value["worker_epoch"] = str(value["worker_epoch"])
    if len(value["payload_sha256"]) != 64:
        raise ValueError("invalid payload digest")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--allowed-context-lengths", type=parse_lengths, required=True)
    parser.add_argument("--warmup-per-length", type=int, default=3)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--predecessor-journal", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    predecessor = None
    if args.predecessor_journal is not None:
        predecessor = json.loads(args.predecessor_journal.read_text())
        if (predecessor.get("recovered_by_worker_epoch") != args.worker_epoch
                or predecessor.get("stage") not in {
                    "nonlaunch_fenced", "quiescence_fenced"
                }):
            raise RuntimeError("predecessor journal is not terminal for this epoch")
    torch.cuda.set_device(0)
    unit = TraceQwenPrefill(args.model, args.allowed_context_lengths, 1)
    for length in args.allowed_context_lengths:
        for _ in range(args.warmup_per_length):
            unit.run(length)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket)); server.listen(1)
    ready = {
        "schema": "softwall-c164-process-worker-ready-v1",
        "host": platform.node(), "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "pid": os.getpid(), "worker_epoch": args.worker_epoch,
        "ready_ns": time.perf_counter_ns(), "socket": str(args.socket),
        "predecessor_journal": (None if args.predecessor_journal is None
                                else str(args.predecessor_journal)),
        "predecessor_journal_valid": (args.predecessor_journal is None
                                      or predecessor is not None),
    }
    durable_json(args.ready_file, ready)
    clean_stop = False
    try:
        connection, _ = server.accept()
        with connection, connection.makefile("rwb", buffering=0) as channel:
            request = json.loads(channel.readline())
            if request.get("op") == "stop":
                channel.write(b'{"ok":true,"stopped":true}\n')
                clean_stop = True
                return
            if request.get("op") != "run_fault":
                raise ValueError("unknown operation")
            expected = identity(request)
            if expected["worker_epoch"] != args.worker_epoch:
                raise ValueError("worker epoch mismatch")
            context = int(request["context_length"])
            fault = str(request["fault_mode"])
            if context not in args.allowed_context_lengths:
                raise ValueError("context outside allowed set")
            if fault not in {"drop_after_prepare", "drop_after_launch"}:
                raise ValueError("unsupported replacement fault")
            record = {
                "schema": "softwall-c164-process-replacement-journal-v1",
                "identity": expected, "stage": "prepared", "journal_seq": 1,
                "context_length": context, "prepared_ns": time.perf_counter_ns(),
                "physical_launch_count": 0, "quiescence_fence_count": 0,
                "worker_pid": os.getpid(),
            }
            durable_json(args.journal, record)
            if fault == "drop_after_prepare":
                connection.shutdown(socket.SHUT_RDWR)
                while True: time.sleep(1)
            record.update({"stage": "launched", "journal_seq": 2,
                           "physical_launch_count": 1,
                           "launch_called_ns": time.perf_counter_ns()})
            durable_json(args.journal, record)
            first = unit.submit(context)
            record.update({"journal_seq": 3,
                           "first_enqueue_return_ns": time.perf_counter_ns(),
                           "physical_submission_observed": True})
            durable_json(args.journal, record)
            connection.shutdown(socket.SHUT_RDWR)
            # Maintain queued/outstanding work until the supervisor invokes the
            # MPS terminate_client primitive. No completion fence is published.
            events = [first]
            while True:
                events.append(unit.submit(context))
                if len(events) > 16:
                    events.pop(0)
    finally:
        if clean_stop:
            durable_json(args.output, {
                "schema": "softwall-c164-process-replacement-worker-output-v1",
                **ready, "clean_stop": True,
                "stopped_ns": time.perf_counter_ns(),
            })
        server.close(); args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
