#!/usr/bin/env python3

import threading
import time
import unittest

from single_token_pipelined_global_trace_client import (
    SingleTokenPipelinedGlobalTraceClient,
)


class FakeSocket:
    def settimeout(self, value):
        self.timeout = value


class TwoRequestBackend:
    def __init__(self):
        now = time.perf_counter_ns()
        self.requests = [
            {
                "request_id": "r%d" % index,
                "source_order": index,
                "context_length": 64,
                "value_tokens": 64,
                "arrival_ns": now - 1,
                "deadline_ns": now + 2_000_000_000,
                "broker_token": "t%d" % index,
                "state": "ready",
            }
            for index in range(2)
        ]
        self.prepare_calls = 0
        self.lock = threading.Lock()


class TwoRequestClient:
    def __init__(self, backend):
        self.backend = backend
        self.socket = FakeSocket()

    def peek_edf_fitting(self, now_ns, horizon_ns, guard_ns, bounds_ns):
        del now_ns, horizon_ns, guard_ns, bounds_ns
        with self.backend.lock:
            self.backend.prepare_calls += 1
            for request in self.backend.requests:
                if request["state"] == "ready":
                    request["state"] = "held"
                    return {key: value for key, value in request.items()
                            if key != "state"}
        return None

    def commit(self, request):
        with self.backend.lock:
            target = self.backend.requests[request["source_order"]]
            if target["state"] != "held":
                raise ValueError("not held")
            target["state"] = "inflight"

    def release(self, request):
        with self.backend.lock:
            target = self.backend.requests[request["source_order"]]
            if target["state"] != "held":
                raise ValueError("not held")
            target["state"] = "ready"

    def complete(self, request, returned_ns):
        del returned_ns
        with self.backend.lock:
            target = self.backend.requests[request["source_order"]]
            if target["state"] != "inflight":
                raise ValueError("not inflight")
            target["state"] = "completed"

    def finalize(self, now_ns):
        del now_ns

    def summary(self):
        return {}

    def close(self):
        pass


class Factory:
    def __init__(self, backend):
        self.backend = backend

    def __call__(self, path, home, epoch_ns, trace_sha256):
        del path, home, epoch_ns, trace_sha256
        return TwoRequestClient(self.backend)


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.001)


class SingleTokenPipelinedClientTest(unittest.TestCase):
    def test_short_horizon_does_not_prepare_second_token(self):
        backend = TwoRequestBackend()
        client = SingleTokenPipelinedGlobalTraceClient(
            "unused", 0, 1, "sealed", 5,
            client_factory=Factory(backend),
        )
        now = time.perf_counter_ns()
        client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        )
        wait_for(lambda: client.rpc_telemetry()["state"]["staged"])
        self.assertEqual(backend.prepare_calls, 1)

        for _ in range(20):
            now = time.perf_counter_ns()
            self.assertIsNone(client.peek_edf_fitting(
                now, now + 1_000_000, 9_000_000, {64: 35_000_000}
            ))
        time.sleep(0.02)
        self.assertEqual(backend.prepare_calls, 1)
        self.assertEqual(
            [request["state"] for request in backend.requests],
            ["held", "ready"],
        )
        telemetry = client.rpc_telemetry()
        self.assertTrue(telemetry["state"]["ownership_invariant_holds"])
        self.assertEqual(telemetry["state"]["unlaunched_tokens"], 1)
        self.assertGreater(
            telemetry["ownership_evidence"]["suppressed_prepare_due_owned_token"],
            0,
        )
        self.assertEqual(
            telemetry["ownership_evidence"]["maximum_unlaunched_tokens"], 1
        )

        now = time.perf_counter_ns()
        request = client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        )
        self.assertEqual(request["request_id"], "r0")
        self.assertTrue(client.commit(request))
        self.assertTrue(client.complete(request, time.perf_counter_ns()))
        client.finalize(time.perf_counter_ns())
        client.close()

    def test_offered_token_suppresses_speculative_prepare(self):
        backend = TwoRequestBackend()
        client = SingleTokenPipelinedGlobalTraceClient(
            "unused", 0, 1, "sealed", 5,
            client_factory=Factory(backend),
        )
        now = time.perf_counter_ns()
        client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        )
        wait_for(lambda: client.rpc_telemetry()["state"]["staged"])
        now = time.perf_counter_ns()
        request = client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        )
        self.assertIsNotNone(request)
        now = time.perf_counter_ns()
        self.assertIsNone(client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        ))
        time.sleep(0.02)
        self.assertEqual(backend.prepare_calls, 1)
        self.assertTrue(client.release(request))
        client.finalize(time.perf_counter_ns())
        client.close()


if __name__ == "__main__":
    unittest.main()
