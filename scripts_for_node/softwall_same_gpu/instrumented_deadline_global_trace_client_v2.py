#!/usr/bin/env python3
"""RPC telemetry that records only calls attempted while the broker is enabled."""

from __future__ import annotations

import time

from deadline_fault_contained_global_trace_client import (
    DeadlineFaultContainedGlobalTraceClient,
)


class InstrumentedDeadlineGlobalTraceClientV2(DeadlineFaultContainedGlobalTraceClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rpc_records = []

    def _record(self, operation, begin_ns, faults_before, request, result):
        end_ns = time.perf_counter_ns()
        faulted = len(self.faults) > faults_before
        self.rpc_records.append({
            "operation": operation,
            "begin_ns": begin_ns,
            "end_ns": end_ns,
            "elapsed_ms": (end_ns - begin_ns) / 1e6,
            "timeout_ms": self.rpc_timeout_ms,
            "wall_timeout_exceeded": end_ns - begin_ns > round(self.rpc_timeout_ms * 1e6),
            "faulted": faulted,
            "request_id": None if request is None else request.get("request_id"),
            "broker_token": None if request is None else request.get("broker_token"),
            "selected": result is not None if operation == "prepare" else None,
            "confirmed": bool(result) if operation != "prepare" else not faulted,
        })
        return result

    def peek_edf_fitting(self, now_ns, horizon_ns, guard_ns, bounds_ns):
        if not self.enabled:
            return None
        begin_ns = time.perf_counter_ns(); faults_before = len(self.faults)
        result = super().peek_edf_fitting(now_ns, horizon_ns, guard_ns, bounds_ns)
        return self._record("prepare", begin_ns, faults_before, result, result)

    def commit(self, request):
        if not self.enabled:
            return False
        begin_ns = time.perf_counter_ns(); faults_before = len(self.faults)
        result = super().commit(request)
        return self._record("commit", begin_ns, faults_before, request, result)

    def release(self, request):
        if not self.enabled:
            return False
        begin_ns = time.perf_counter_ns(); faults_before = len(self.faults)
        result = super().release(request)
        return self._record("abort", begin_ns, faults_before, request, result)

    def complete(self, request, returned_ns):
        if not self.enabled:
            return False
        begin_ns = time.perf_counter_ns(); faults_before = len(self.faults)
        result = super().complete(request, returned_ns)
        return self._record("complete", begin_ns, faults_before, request, result)

    def rpc_telemetry(self):
        by_operation = {}
        for record in self.rpc_records:
            item = by_operation.setdefault(record["operation"], {
                "count": 0, "faults": 0, "wall_timeout_exceeded": 0,
                "max_elapsed_ms": None,
            })
            item["count"] += 1
            item["faults"] += int(record["faulted"])
            item["wall_timeout_exceeded"] += int(record["wall_timeout_exceeded"])
            value = record["elapsed_ms"]
            item["max_elapsed_ms"] = value if item["max_elapsed_ms"] is None else max(
                item["max_elapsed_ms"], value
            )
        return {"schema": "softwall-broker-rpc-telemetry-v2",
                "timeout_ms": self.rpc_timeout_ms,
                "records": list(self.rpc_records), "by_operation": by_operation}
