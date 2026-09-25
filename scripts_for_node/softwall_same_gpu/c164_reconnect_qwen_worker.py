#!/usr/bin/env python3.11
"""Persistent Qwen worker with fsync-backed reconnect reconciliation."""

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

from c161_phase2_qwen_worker import TraceQwenPrefill, parse_lengths, summarize


def durable_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    encoded = (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()
    with temporary.open("wb") as handle:
        handle.write(encoded)
        handle.flush()
        os.fsync(handle.fileno())
    temporary.replace(path)
    descriptor = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def journal_path(directory: Path, token: str) -> Path:
    name = hashlib.sha256(token.encode()).hexdigest()[:32]
    return directory / f"{name}.json"


def identity(request: dict) -> dict:
    value = {
        "token": str(request["token"]),
        "lifecycle_epoch": int(request["lifecycle_epoch"]),
        "payload_sha256": str(request["payload_sha256"]),
        "worker_epoch": str(request["worker_epoch"]),
    }
    if len(value["payload_sha256"]) != 64:
        raise ValueError("invalid payload digest")
    return value


def same_identity(record: dict, expected: dict) -> bool:
    return record.get("identity") == expected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen2.5-1.5B")
    parser.add_argument("--allowed-context-lengths", type=parse_lengths, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--warmup-per-length", type=int, default=3)
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--journal-dir", type=Path, required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.batch_size <= 0 or args.warmup_per_length < 0:
        parser.error("invalid batch size or warmup count")

    torch.cuda.set_device(0)
    unit = TraceQwenPrefill(args.model, args.allowed_context_lengths, args.batch_size)
    warmup = []
    for length in args.allowed_context_lengths:
        for _ in range(args.warmup_per_length):
            warmup.append({"context_length": length, "gpu_ms": unit.run(length)})
    args.journal_dir.mkdir(parents=True, exist_ok=True)
    args.socket.parent.mkdir(parents=True, exist_ok=True)
    args.socket.unlink(missing_ok=True)
    records = []
    events = []
    rejected = []
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(args.socket))
    server.listen(8)
    stopped = False
    started_ns = time.perf_counter_ns()
    try:
        while not stopped:
            connection, _ = server.accept()
            with connection, connection.makefile("rwb", buffering=0) as channel:
                line = channel.readline()
                if not line:
                    continue
                request = None
                try:
                    request = json.loads(line)
                    op = request.get("op")
                    if op == "stop":
                        response = {"ok": True, "stopped": True}
                        stopped = True
                    elif op == "run":
                        expected = identity(request)
                        if expected["worker_epoch"] != args.worker_epoch:
                            raise ValueError("worker epoch mismatch")
                        context = int(request["context_length"])
                        if context not in args.allowed_context_lengths:
                            raise ValueError("context outside qualified set")
                        fault = str(request["fault_mode"])
                        if fault not in {"drop_after_prepare", "drop_after_fence"}:
                            raise ValueError("unsupported reconnect fault")
                        path = journal_path(args.journal_dir, expected["token"])
                        if path.exists():
                            record = json.loads(path.read_text())
                            if not same_identity(record, expected):
                                raise ValueError("token identity conflict")
                            response = {"ok": True, "duplicate": True,
                                        "record": record}
                        else:
                            accepted_ns = time.perf_counter_ns()
                            record = {
                                "schema": "softwall-c164-reconnect-journal-v1",
                                "identity": expected, "stage": "prepared",
                                "journal_seq": 1, "context_length": context,
                                "accepted_ns": accepted_ns,
                                "physical_launch_count": 0,
                                "physical_fence_count": 0,
                            }
                            durable_json(path, record)
                            events.append({"event": "prepare", "token": expected["token"],
                                           "time_ns": time.perf_counter_ns()})
                            if fault == "drop_after_prepare":
                                events.append({"event": fault, "token": expected["token"],
                                               "time_ns": time.perf_counter_ns()})
                                connection.shutdown(socket.SHUT_RDWR)
                                continue
                            latest_start_ns = int(request["latest_start_ns"])
                            launch_called_ns = time.perf_counter_ns()
                            if launch_called_ns > latest_start_ns:
                                record.update({
                                    "stage": "nonlaunch_fenced", "journal_seq": 2,
                                    "physical_fence_count": 1,
                                    "reason": "latest_start_expired",
                                    "terminal_ns": launch_called_ns,
                                })
                                durable_json(path, record)
                                response = {"ok": True, "record": record}
                            else:
                                record.update({
                                    "stage": "launched", "journal_seq": 2,
                                    "physical_launch_count": 1,
                                    "launch_called_ns": launch_called_ns,
                                })
                                durable_json(path, record)
                                gpu_events = unit.submit(context)
                                submitted_ns = time.perf_counter_ns()
                                gpu_ms = unit.finish(gpu_events)
                                completed_ns = time.perf_counter_ns()
                                record.update({
                                    "stage": "fenced", "journal_seq": 3,
                                    "physical_fence_count": 1,
                                    "submitted_ns": submitted_ns,
                                    "completed_ns": completed_ns,
                                    "gpu_ms": gpu_ms,
                                })
                                durable_json(path, record)
                                records.append(dict(record))
                                events.extend((
                                    {"event": "launch_called", "token": expected["token"],
                                     "time_ns": launch_called_ns},
                                    {"event": "enqueue_return", "token": expected["token"],
                                     "time_ns": submitted_ns},
                                    {"event": "fence", "token": expected["token"],
                                     "time_ns": completed_ns},
                                    {"event": fault, "token": expected["token"],
                                     "time_ns": time.perf_counter_ns()},
                                ))
                                connection.shutdown(socket.SHUT_RDWR)
                                continue
                    elif op == "reconcile":
                        expected = identity(request)
                        if expected["worker_epoch"] != args.worker_epoch:
                            raise ValueError("worker epoch mismatch")
                        path = journal_path(args.journal_dir, expected["token"])
                        if not path.exists():
                            response = {"ok": True, "resolution": "retain_absent",
                                        "record": None}
                        else:
                            record = json.loads(path.read_text())
                            if not same_identity(record, expected):
                                raise ValueError("reconnect identity mismatch")
                            if record["stage"] == "prepared":
                                record.update({
                                    "stage": "nonlaunch_fenced",
                                    "journal_seq": record["journal_seq"] + 1,
                                    "physical_fence_count": 1,
                                    "terminal_ns": time.perf_counter_ns(),
                                    "reason": "reconnect_abort_before_launch",
                                })
                                durable_json(path, record)
                                events.append({"event": "nonlaunch_fence",
                                               "token": expected["token"],
                                               "time_ns": record["terminal_ns"]})
                            resolution = (
                                "retire_terminal" if record["stage"]
                                in {"fenced", "nonlaunch_fenced"}
                                else "retain_launched"
                            )
                            response = {"ok": True, "resolution": resolution,
                                        "record": record}
                    else:
                        raise ValueError("unknown operation")
                except Exception as error:
                    rejected.append({"request": request, "error": repr(error),
                                     "time_ns": time.perf_counter_ns()})
                    response = {"ok": False, "error": repr(error)}
                channel.write(json.dumps(response).encode() + b"\n")
    finally:
        journal_records = [json.loads(path.read_text())
                           for path in sorted(args.journal_dir.glob("*.json"))]
        result = {
            "schema": "softwall-c164-reconnect-qwen-worker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "worker_epoch": args.worker_epoch,
            "started_ns": started_ns,
            "stopped_ns": time.perf_counter_ns(),
            "warmup": warmup,
            "gpu_ms": summarize([row["gpu_ms"] for row in records]),
            "physical_records": records,
            "journal_records": journal_records,
            "events": events, "rejected": rejected,
        }
        durable_json(args.output, result)
        server.close()
        args.socket.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
