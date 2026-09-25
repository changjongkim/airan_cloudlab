#!/usr/bin/env python3.11
"""Inject per-token channel loss and reconcile through a new socket."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import socket
import time
from pathlib import Path

from c164_reconnect_state_model_v1 import digest_payload


def rpc(path: Path, request: dict) -> dict:
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(30.0)
    client.connect(str(path))
    with client, client.makefile("rwb", buffering=0) as channel:
        channel.write(json.dumps(request).encode() + b"\n")
        raw = channel.readline()
    if not raw:
        raise ConnectionError("worker channel closed without reply")
    response = json.loads(raw)
    if not response.get("ok"):
        raise RuntimeError(response)
    return response


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--lifecycle-epoch", type=int, required=True)
    parser.add_argument("--tokens", type=int, default=60)
    parser.add_argument("--pacing-ms", type=float, default=90.0)
    parser.add_argument("--mandatory-ready", type=Path, required=True)
    parser.add_argument("--launch-ahead-ms", type=float, default=10.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.tokens <= 0 or args.pacing_ms <= 0 or args.launch_ahead_ms <= 0:
        parser.error("token count and pacing must be positive")
    ready = json.loads(args.mandatory_ready.read_text())
    first_release_ns = int(ready["first_release_ns"])
    period_ns = int(ready["period_ns"])
    next_release_index = 0
    records = []
    started_ns = time.perf_counter_ns()
    for index in range(args.tokens):
        fault = "drop_after_prepare" if index % 2 == 0 else "drop_after_fence"
        context = 128 if (index // 2) % 2 == 0 else 512
        token = f"e{args.lifecycle_epoch}:w{args.worker_epoch}:t{index:04d}"
        payload = {"context_length": context, "source_order": index}
        immutable = {
            "token": token, "lifecycle_epoch": args.lifecycle_epoch,
            "payload_sha256": digest_payload(payload),
            "worker_epoch": args.worker_epoch,
        }
        request = {
            "op": "run", **immutable, "context_length": context,
            "latest_start_ns": time.perf_counter_ns() + 2_000_000_000,
            "fault_mode": fault,
        }
        aligned_release_ns = None
        if fault == "drop_after_fence":
            now_ns = time.perf_counter_ns()
            earliest_index = max(
                0, (now_ns - first_release_ns + period_ns - 1) // period_ns
            )
            next_release_index = max(next_release_index, earliest_index + 1)
            aligned_release_ns = first_release_ns + next_release_index * period_ns
            target_ns = aligned_release_ns - round(args.launch_ahead_ms * 1e6)
            while time.perf_counter_ns() < target_ns:
                time.sleep(0.0002)
            request["latest_start_ns"] = aligned_release_ns + 20_000_000
            next_release_index += 1
        sent_ns = time.perf_counter_ns()
        lost = False
        try:
            rpc(args.socket, request)
        except ConnectionError:
            lost = True
        disconnected_ns = time.perf_counter_ns()
        reconciliation = rpc(args.socket, {"op": "reconcile", **immutable})
        reconciled_ns = time.perf_counter_ns()
        duplicate = rpc(args.socket, {"op": "reconcile", **immutable})
        record = reconciliation["record"]
        records.append({
            "index": index, "fault_mode": fault, "context_length": context,
            "identity": immutable, "sent_ns": sent_ns,
            "disconnected_ns": disconnected_ns,
            "reconciled_ns": reconciled_ns,
            "channel_loss_observed": lost,
            "resolution": reconciliation["resolution"],
            "record": record,
            "aligned_release_ns": aligned_release_ns,
            "duplicate_reconcile_stable": duplicate.get("record") == record,
        })
        if fault == "drop_after_prepare":
            time.sleep(args.pacing_ms / 1000.0)
    stopped = rpc(args.socket, {"op": "stop"})
    value = {
        "schema": "softwall-c164-reconnect-client-v1",
        "host": platform.node(), "worker_epoch": args.worker_epoch,
        "lifecycle_epoch": args.lifecycle_epoch,
        "tokens": args.tokens, "pacing_ms": args.pacing_ms,
        "launch_ahead_ms": args.launch_ahead_ms,
        "mandatory_ready": str(args.mandatory_ready),
        "started_ns": started_ns, "completed_ns": time.perf_counter_ns(),
        "stop_ack": stopped.get("stopped") is True,
        "records": records,
    }
    atomic_json(args.output, value)


if __name__ == "__main__":
    main()
