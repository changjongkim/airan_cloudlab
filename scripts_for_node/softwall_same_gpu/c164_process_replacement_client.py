#!/usr/bin/env python3.11
"""Submit one replacement fault token and require response-channel loss."""

from __future__ import annotations

import argparse
import json
import platform
import socket
import time
from pathlib import Path

from c164_reconnect_state_model_v1 import digest_payload


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", type=Path, required=True)
    parser.add_argument("--worker-epoch", required=True)
    parser.add_argument("--lifecycle-epoch", type=int, required=True)
    parser.add_argument("--token", required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--fault-mode", choices=("drop_after_prepare", "drop_after_launch"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    payload = {"context_length": args.context_length, "token": args.token}
    request = {
        "op": "run_fault", "token": args.token,
        "lifecycle_epoch": args.lifecycle_epoch,
        "payload_sha256": digest_payload(payload),
        "worker_epoch": args.worker_epoch,
        "context_length": args.context_length, "fault_mode": args.fault_mode,
    }
    sent_ns = time.perf_counter_ns()
    client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    client.settimeout(30); client.connect(str(args.socket))
    with client, client.makefile("rwb", buffering=0) as channel:
        channel.write(json.dumps(request).encode() + b"\n")
        raw = channel.readline()
    returned_ns = time.perf_counter_ns()
    value = {
        "schema": "softwall-c164-process-replacement-client-v1",
        "host": platform.node(), "clock": "time.perf_counter_ns",
        "sent_ns": sent_ns, "returned_ns": returned_ns,
        "request": request, "channel_loss_observed": raw == b"",
        "unexpected_response": raw.decode(errors="replace"),
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    raise SystemExit(0 if value["channel_loss_observed"] else 1)


if __name__ == "__main__":
    main()
