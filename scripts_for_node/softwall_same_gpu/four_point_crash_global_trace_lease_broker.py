#!/usr/bin/env python3
"""V16 broker fault harness covering abort as a post-apply crash point.

This is a separate qualification harness so the source hashes frozen for the
V15 C145/C146 evidence remain unchanged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import socketserver
import threading
from pathlib import Path

import control_point_crash_global_trace_lease_broker as base


FAULT_OPERATIONS = ("prepare", "abort", "commit", "complete")


class FourPointBrokerHandler(socketserver.StreamRequestHandler):
    """Apply a broker transition and crash before replying at all four points."""

    def handle(self) -> None:
        while True:
            raw = self.rfile.readline()
            if not raw:
                return
            try:
                request = json.loads(raw)
                op = request["op"]
                if op == "init":
                    result = self.server.state.initialize(
                        request["home"], request["epoch_ns"],
                        request["trace_sha256"],
                    )
                elif op == "prepare":
                    result = self.server.state.prepare(
                        request["home"], request["now_ns"],
                        request["horizon_ns"], request["guard_ns"],
                        request["bounds_ns"],
                    )
                elif op == "commit":
                    result = self.server.state.commit(
                        request["home"], request["token"], request["now_ns"]
                    )
                elif op == "abort":
                    result = self.server.state.abort(
                        request["home"], request["token"], request["now_ns"]
                    )
                elif op == "complete":
                    result = self.server.state.complete(
                        request["home"], request["token"],
                        request["returned_ns"],
                    )
                elif op == "finalize":
                    result = self.server.state.finalize(
                        request["home"], request["now_ns"]
                    )
                elif op == "snapshot":
                    result = self.server.state.snapshot()
                elif op == "stop":
                    result = self.server.state.snapshot()
                    self.server.stop_requested = True
                    threading.Thread(
                        target=self.server.shutdown, daemon=True
                    ).start()
                else:
                    raise ValueError("unknown operation")

                if (op in FAULT_OPERATIONS
                        and self.server.should_crash_after_operation(
                            op, request, result
                        )):
                    os._exit(86)
                response = {"ok": True, "result": result}
            except Exception as error:
                response = {
                    "ok": False,
                    "type": type(error).__name__,
                    "error": str(error),
                }
            self.wfile.write(json.dumps(response).encode() + b"\n")
            self.wfile.flush()
            if request.get("op") == "stop":
                return


class FourPointBrokerServer(base.BrokerServer):
    daemon_threads = True

    def __init__(self, path, state, crash_after_operation=None,
                 crash_after_home=None, crash_after_number=None):
        self.state = state
        self.stop_requested = False
        self.crash_after_operation = crash_after_operation
        self.crash_after_home = crash_after_home
        self.crash_after_number = crash_after_number
        self.operation_counts = {}
        self.faults = []
        self.fault_lock = threading.Lock()
        socketserver.ThreadingUnixStreamServer.__init__(
            self, path, FourPointBrokerHandler
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--socket", required=True)
    parser.add_argument("--stop-only", action="store_true")
    parser.add_argument("--trace")
    parser.add_argument("--trace-sha256")
    parser.add_argument("--participants", type=int)
    parser.add_argument(
        "--policy", choices=("global", "static_partition"), default="global"
    )
    parser.add_argument("--crash-after-operation", choices=FAULT_OPERATIONS)
    parser.add_argument("--crash-after-home", type=int)
    parser.add_argument("--crash-after-number", type=int)
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.stop_only:
        base.stop_broker(args.socket)
        return
    if not all((args.trace, args.trace_sha256, args.participants, args.output)):
        parser.error("serve mode requires trace, hash, participants, and output")
    trace_path = Path(args.trace)
    observed_sha = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    if observed_sha != args.trace_sha256:
        raise RuntimeError("sealed trace hash mismatch")
    socket_path = Path(args.socket)
    socket_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        socket_path.unlink()
    except FileNotFoundError:
        pass
    state = base.GlobalTraceLeaseState(
        json.loads(trace_path.read_text(encoding="utf-8")), observed_sha,
        args.participants, args.policy,
    )
    configured = (
        args.crash_after_operation,
        args.crash_after_home,
        args.crash_after_number,
    )
    if any(value is not None for value in configured) and not all(
            value is not None for value in configured):
        parser.error("all broker-crash fault arguments are required")
    if args.crash_after_number is not None and args.crash_after_number <= 0:
        parser.error("broker-crash fault number must be positive")
    if (args.crash_after_home is not None
            and not 0 <= args.crash_after_home < args.participants):
        parser.error("broker-crash fault home is outside participant range")
    server = FourPointBrokerServer(
        str(socket_path), state, args.crash_after_operation,
        args.crash_after_home, args.crash_after_number,
    )
    try:
        server.serve_forever(poll_interval=0.05)
    finally:
        server.server_close()
        output = state.output()
        output["schema"] = "softwall-global-trace-lease-broker-four-point-fault-v1"
        output["fault_configuration"] = {
            "crash_after_operation": args.crash_after_operation,
            "crash_after_home": args.crash_after_home,
            "crash_after_number": args.crash_after_number,
        }
        output["faults"] = list(server.faults)
        output["operation_counts"] = dict(server.operation_counts)
        base.atomic_json(Path(args.output), output)
        try:
            socket_path.unlink()
        except FileNotFoundError:
            pass


if __name__ == "__main__":
    main()
