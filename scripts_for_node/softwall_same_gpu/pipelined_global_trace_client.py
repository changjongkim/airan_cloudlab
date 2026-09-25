#!/usr/bin/env python3
"""Pipelined global trace client with one synchronous launch commit.

The timed RAN executor never waits for prepare, abort, or complete RPCs.  A
dedicated control connection performs those operations in FIFO order.  A
prepared request is returned to the executor only after it is revalidated
against the *current* local horizon.  Commit remains synchronous because a
physical AI launch is forbidden until the broker has acknowledged ownership.

After physical completion, ``complete`` only transfers the request to the
control worker.  The broker token stays inflight until that worker receives the
ACK, so deferred completion cannot permit duplicate execution.
"""

from __future__ import annotations

import queue
import threading
import time

from global_trace_lease_broker import GlobalTraceLeaseClient


class PipelinedGlobalTraceClient:
    """Queue-compatible client that removes non-launch RPCs from the RAN path."""

    completion_ack_deferred = True

    def __init__(self, path: str, home: int, epoch_ns: int, trace_sha256: str,
                 rpc_timeout_ms: float, client_factory=GlobalTraceLeaseClient):
        if rpc_timeout_ms <= 0:
            raise ValueError("global broker RPC timeout must be positive")
        self.path = path
        self.home = home
        self.epoch_ns = epoch_ns
        self.trace_sha256 = trace_sha256
        self.rpc_timeout_ms = rpc_timeout_ms
        self.enabled = True
        self.faults = []
        self.last_summary = None

        # Separate connections prevent a stalled asynchronous operation from
        # holding the lock used by the launch-critical commit.
        self.commit_client = client_factory(path, home, epoch_ns, trace_sha256)
        self.control_client = client_factory(path, home, epoch_ns, trace_sha256)
        for client in (self.commit_client, self.control_client):
            client.socket.settimeout(rpc_timeout_ms / 1000.0)

        self._lock = threading.Lock()
        self._tasks = queue.Queue()
        self._prepare_pending = False
        self._staged = None
        self._offered = {}
        self._inflight = {}
        self._closing = False
        self._records = []
        self._handoff_records = []
        self._worker = threading.Thread(
            target=self._control_loop,
            name=f"softwall-global-control-h{home}",
            daemon=True,
        )
        self._worker.start()

    def _record_rpc(self, operation, begin_ns, request, result, faulted,
                    path="async-control"):
        end_ns = time.perf_counter_ns()
        record = {
            "operation": operation,
            "path": path,
            "begin_ns": begin_ns,
            "end_ns": end_ns,
            "elapsed_ms": (end_ns - begin_ns) / 1e6,
            "timeout_ms": self.rpc_timeout_ms,
            "wall_timeout_exceeded": (
                end_ns - begin_ns > round(self.rpc_timeout_ms * 1e6)
            ),
            "faulted": faulted,
            "request_id": None if request is None else request.get("request_id"),
            "broker_token": None if request is None else request.get("broker_token"),
            "selected": result is not None if operation == "prepare" else None,
            "confirmed": not faulted,
        }
        with self._lock:
            self._records.append(record)
        return result

    def _record_handoff(self, operation, begin_ns, accepted):
        end_ns = time.perf_counter_ns()
        with self._lock:
            self._handoff_records.append({
                "operation": operation,
                "begin_ns": begin_ns,
                "end_ns": end_ns,
                "elapsed_ms": (end_ns - begin_ns) / 1e6,
                "accepted": bool(accepted),
            })

    def _quarantine(self, operation: str, error: Exception, request=None):
        with self._lock:
            self.faults.append({
                "operation": operation,
                "type": type(error).__name__,
                "message": str(error),
                "detected_ns": time.perf_counter_ns(),
                "home": self.home,
                "request_id": None if request is None else request.get("request_id"),
                "broker_token": None if request is None else request.get("broker_token"),
                "rpc_timeout_ms": self.rpc_timeout_ms,
                "action": (
                    "disable-new-global-ai; retain ambiguous broker token; "
                    "local RAN executor never waits"
                ),
            })
            self.enabled = False

    def _control_loop(self):
        while True:
            task = self._tasks.get()
            try:
                if task[0] == "stop":
                    return
                operation, payload = task
                with self._lock:
                    enabled = self.enabled
                if not enabled:
                    if operation == "prepare":
                        with self._lock:
                            self._prepare_pending = False
                    continue
                begin_ns = time.perf_counter_ns()
                request = payload.get("request")
                result = None
                faulted = False
                try:
                    if operation == "prepare":
                        result = self.control_client.peek_edf_fitting(
                            payload["now_ns"], payload["horizon_ns"],
                            payload["guard_ns"], payload["bounds_ns"],
                        )
                    elif operation == "abort":
                        self.control_client.release(request)
                        result = True
                    elif operation == "complete":
                        self.control_client.complete(request, payload["returned_ns"])
                        result = True
                    else:
                        raise RuntimeError(f"unknown asynchronous operation: {operation}")
                except Exception as error:
                    faulted = True
                    self._quarantine(operation, error, request)
                finally:
                    self._record_rpc(
                        operation, begin_ns, request if request is not None else result,
                        result, faulted,
                    )
                with self._lock:
                    if operation == "prepare":
                        self._prepare_pending = False
                        if not faulted and result is not None and self._staged is None:
                            self._staged = result
                    elif operation == "complete" and not faulted:
                        self._inflight.pop(request["broker_token"], None)
            finally:
                self._tasks.task_done()

    @staticmethod
    def _fits(request, now_ns, horizon_ns, guard_ns, bounds_ns):
        bound = bounds_ns.get(request["context_length"])
        if bound is None:
            bound = bounds_ns.get(str(request["context_length"]))
        if bound is None:
            raise KeyError("missing context bound")
        return now_ns + bound + guard_ns <= min(
            horizon_ns, request["deadline_ns"]
        )

    def peek_edf_fitting(self, now_ns, horizon_ns, guard_ns, bounds_ns):
        """Poll a prepared token and submit the next prepare without waiting."""
        begin_ns = time.perf_counter_ns()
        request = None
        abort = None
        submit = False
        with self._lock:
            if self.enabled and not self._closing:
                current_ns = time.perf_counter_ns()
                if self._staged is not None:
                    candidate = self._staged
                    if self._fits(
                            candidate, current_ns, horizon_ns, guard_ns, bounds_ns):
                        self._staged = None
                        request = candidate
                        self._offered[candidate["broker_token"]] = candidate
                    else:
                        bound = bounds_ns.get(candidate["context_length"])
                        if bound is None:
                            bound = bounds_ns.get(str(candidate["context_length"]))
                        # A short current RAN window does not make the global
                        # request stale: a later conditional or post-radio
                        # window may be larger.  Abort only when the request's
                        # own SLO can no longer be met.
                        permanently_stale = (
                            bound is None
                            or current_ns + bound + guard_ns
                            > candidate["deadline_ns"]
                        )
                    if request is None and permanently_stale:
                        self._staged = None
                        abort = candidate
                if request is None and abort is None and not self._prepare_pending:
                    self._prepare_pending = True
                    submit = True
        if abort is not None:
            self._tasks.put(("abort", {"request": abort}))
        elif submit:
            self._tasks.put(("prepare", {
                # Use a fresh submission time.  A second current-time check is
                # performed before any prepared request is returned.
                "now_ns": time.perf_counter_ns(),
                "horizon_ns": horizon_ns,
                "guard_ns": guard_ns,
                "bounds_ns": dict(bounds_ns),
            }))
        self._record_handoff("poll_prepare", begin_ns, request is not None or submit)
        return request

    def commit(self, request):
        """Synchronously confirm global ownership before physical AI launch."""
        token = request["broker_token"]
        with self._lock:
            if not self.enabled or token not in self._offered:
                return False
        begin_ns = time.perf_counter_ns()
        faulted = False
        try:
            self.commit_client.commit(request)
        except Exception as error:
            faulted = True
            self._quarantine("commit", error, request)
        self._record_rpc(
            "commit", begin_ns, request, not faulted, faulted,
            path="synchronous-launch",
        )
        with self._lock:
            self._offered.pop(token, None)
            if not faulted:
                self._inflight[token] = request
        return not faulted

    def release(self, request):
        """Defer abort of a held, never-launched request."""
        begin_ns = time.perf_counter_ns()
        token = request["broker_token"]
        with self._lock:
            accepted = self.enabled and self._offered.pop(token, None) is not None
        if accepted:
            self._tasks.put(("abort", {"request": request}))
        self._record_handoff("submit_abort", begin_ns, accepted)
        return accepted

    def complete(self, request, returned_ns):
        """Defer broker retirement after the physical CUDA completion fence."""
        begin_ns = time.perf_counter_ns()
        token = request["broker_token"]
        with self._lock:
            accepted = self.enabled and token in self._inflight
        if accepted:
            self._tasks.put(("complete", {
                "request": request,
                "returned_ns": returned_ns,
            }))
        self._record_handoff("submit_complete", begin_ns, accepted)
        return accepted

    def _drain(self, timeout_s):
        deadline = time.monotonic() + timeout_s
        while self._tasks.unfinished_tasks:
            if time.monotonic() >= deadline:
                raise TimeoutError("asynchronous broker control drain timed out")
            time.sleep(0.001)

    def finalize(self, now_ns):
        """Drain deferred control after timed traffic, then finalize metadata."""
        with self._lock:
            self._closing = True
        try:
            self._drain(max(1.0, self.rpc_timeout_ms * 0.01 + 1.0))
            # A prepare already executing when closing began may publish a held
            # token during the first drain.  Abort it only after that worker
            # action has reached a terminal state.
            with self._lock:
                staged = self._staged
                self._staged = None
            if staged is not None:
                self._tasks.put(("abort", {"request": staged}))
                self._drain(max(1.0, self.rpc_timeout_ms * 0.01 + 1.0))
        except Exception as error:
            self._quarantine("drain", error)
        if self.enabled:
            try:
                self.commit_client.finalize(now_ns)
                return True
            except Exception as error:
                self._quarantine("metadata_finalize", error)
        return False

    def summary(self):
        try:
            self.last_summary = self.commit_client.summary()
            return self.last_summary
        except Exception as error:
            self._quarantine("metadata_summary", error)
            return self.last_summary or {"unavailable": True}

    def rpc_telemetry(self):
        with self._lock:
            records = list(self._records)
            handoffs = list(self._handoff_records)
            staged = self._staged is not None
            offered = len(self._offered)
            inflight = len(self._inflight)
            prepare_pending = self._prepare_pending
        by_operation = {}
        for record in records:
            item = by_operation.setdefault(record["operation"], {
                "count": 0,
                "faults": 0,
                "wall_timeout_exceeded": 0,
                "max_elapsed_ms": None,
                "path": record["path"],
            })
            item["count"] += 1
            item["faults"] += int(record["faulted"])
            item["wall_timeout_exceeded"] += int(record["wall_timeout_exceeded"])
            elapsed = record["elapsed_ms"]
            item["max_elapsed_ms"] = (
                elapsed if item["max_elapsed_ms"] is None
                else max(item["max_elapsed_ms"], elapsed)
            )
        return {
            "schema": "softwall-pipelined-broker-rpc-telemetry-v1",
            "timeout_ms": self.rpc_timeout_ms,
            "critical_path_operations": ["commit"],
            "deferred_operations": ["prepare", "abort", "complete"],
            "records": records,
            "handoffs": handoffs,
            "by_operation": by_operation,
            "state": {
                "staged": staged,
                "offered": offered,
                "inflight": inflight,
                "prepare_pending": prepare_pending,
            },
        }

    def close(self):
        with self._lock:
            already_closing = self._closing
            self._closing = True
        if not already_closing:
            try:
                self._drain(1.0)
            except Exception:
                pass
        self._tasks.put(("stop", {}))
        self._worker.join(timeout=1.0)
        for client in (self.control_client, self.commit_client):
            try:
                client.close()
            except Exception:
                pass
