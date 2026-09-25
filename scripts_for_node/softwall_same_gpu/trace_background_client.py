#!/usr/bin/env python3
"""Controller-side client and EDF queue for bounded trace Qwen requests."""

from __future__ import annotations

import json
import socket
import time


class TraceRequestQueue:
    def __init__(self, trace: dict, epoch_ns: int) -> None:
        self.epoch_ns = epoch_ns
        self.future = []
        for request in trace["requests"]:
            item = dict(request)
            item["arrival_ns"] = epoch_ns + round(item["arrival_ms"] * 1e6)
            item["deadline_ns"] = epoch_ns + round(item["deadline_ms"] * 1e6)
            item["state"] = "future"
            self.future.append(item)
        self.cursor = 0
        self.ready = []
        self.completed = []
        self.expired = []

    def observe(self, now_ns: int) -> None:
        while self.cursor < len(self.future):
            request = self.future[self.cursor]
            if request["arrival_ns"] > now_ns:
                break
            request["state"] = "ready"
            request["observed_ns"] = now_ns
            self.ready.append(request)
            self.cursor += 1
        keep = []
        for request in self.ready:
            if request["deadline_ns"] <= now_ns:
                request["state"] = "expired"
                request["expired_ns"] = now_ns
                self.expired.append(request)
            else:
                keep.append(request)
        self.ready = keep

    def peek_edf(self, now_ns: int) -> dict | None:
        self.observe(now_ns)
        if not self.ready:
            return None
        return min(
            self.ready,
            key=lambda request: (request["deadline_ns"], request["source_order"]),
        )

    def peek_edf_fitting(
        self,
        now_ns: int,
        horizon_ns: int,
        guard_ns: int,
        bounds_ns: dict[int, int],
    ) -> dict | None:
        """Return a fitting request from the earliest-deadline cohort.

        Requests with a later deadline never jump an earlier cohort.  Within a
        cohort, a long request that cannot fit the currently certified window
        does not block a shorter request with the same deadline.  This rule is
        shared by every system arm and uses no future arrivals or runtimes.
        """
        self.observe(now_ns)
        if not self.ready:
            return None
        earliest = min(request["deadline_ns"] for request in self.ready)
        for request in sorted(
            (request for request in self.ready
             if request["deadline_ns"] == earliest),
            key=lambda request: request["source_order"],
        ):
            bound_ns = bounds_ns.get(request["context_length"])
            if bound_ns is None:
                raise KeyError(
                    f"missing bound for context {request['context_length']}"
                )
            if now_ns + bound_ns + guard_ns <= min(horizon_ns, earliest):
                return request
        return None

    def complete(self, request: dict, returned_ns: int) -> None:
        self.ready.remove(request)
        request["returned_ns"] = returned_ns
        request["state"] = "completed" if returned_ns <= request["deadline_ns"] else "late"
        self.completed.append(request)

    def finalize(self, now_ns: int) -> None:
        self.observe(now_ns)
        for request in self.ready:
            request["state"] = "unfinished"
        for request in self.future[self.cursor:]:
            request["state"] = "not_arrived"

    def summary(self) -> dict:
        offered = self.future[:self.cursor]
        timely = [request for request in self.completed if request["state"] == "completed"]
        return {
            "offered_requests": len(offered),
            "offered_value_tokens": sum(request["value_tokens"] for request in offered),
            "timely_requests": len(timely),
            "timely_value_tokens": sum(request["value_tokens"] for request in timely),
            "late_requests": sum(request["state"] == "late" for request in self.completed),
            "expired_requests": len(self.expired),
            "unfinished_requests": sum(request["state"] == "unfinished" for request in self.ready),
            "not_arrived_requests": len(self.future) - self.cursor,
            "ready_requests": len(self.ready),
        }


class TraceBackgroundClient:
    def __init__(self, path: str, timeout_ms: float) -> None:
        self.socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.socket.connect(path)
        self.socket.settimeout(timeout_ms / 1000.0)
        self.channel = self.socket.makefile("rwb", buffering=0)
        self.enabled = True
        self.records: list[dict] = []
        self.faults: list[dict] = []

    def run_request(self, request: dict, phase: str, bound_ms: float) -> dict | None:
        admitted_ns = time.perf_counter_ns()
        payload = {
            "op": "run",
            "request_id": request["request_id"],
            "context_length": request["context_length"],
        }
        try:
            self.channel.write(json.dumps(payload).encode() + b"\n")
            raw = self.channel.readline()
            if not raw:
                raise ConnectionError("background worker closed the RPC channel")
            response = json.loads(raw)
            if not response.get("ok"):
                raise RuntimeError(f"background worker rejected request: {response}")
        except (BrokenPipeError, ConnectionError, OSError, ValueError, RuntimeError) as error:
            self.enabled = False
            self.faults.append({
                "request_id": request["request_id"],
                "detected_ns": time.perf_counter_ns(),
                "type": type(error).__name__,
                "message": str(error),
            })
            return None
        returned_ns = time.perf_counter_ns()
        execution_ms = (returned_ns - admitted_ns) / 1e6
        record = {
            "request_id": request["request_id"],
            "source_order": request["source_order"],
            "context_length": request["context_length"],
            "value_tokens": request["value_tokens"],
            "arrival_ns": request["arrival_ns"],
            "deadline_ns": request["deadline_ns"],
            "admitted_ns": admitted_ns,
            "returned_ns": returned_ns,
            "gpu_ms": response["gpu_ms"],
            "execution_ms": execution_ms,
            "bound_ms": bound_ms,
            "bound_violation": execution_ms > bound_ms,
            "timely": returned_ns <= request["deadline_ns"],
            "phase": phase,
        }
        self.records.append(record)
        if record["bound_violation"]:
            self.enabled = False
            self.faults.append({
                "request_id": request["request_id"],
                "detected_ns": returned_ns,
                "type": "AIContractViolation",
                "message": json.dumps(record),
            })
        return record

    def close(self) -> None:
        if self.enabled:
            try:
                self.channel.write(b'{"op":"stop"}\n')
                self.channel.readline()
            except (BrokenPipeError, OSError):
                pass
        try:
            self.channel.close()
        except OSError:
            pass
        self.socket.close()
