#!/usr/bin/env python3

import threading
import time
import unittest

from pipelined_global_trace_client import PipelinedGlobalTraceClient


class FakeSocket:
    def settimeout(self, value):
        self.timeout = value


class FakeBackend:
    def __init__(self, prepare_delay=0.0, complete_delay=0.0,
                 fail_complete=False):
        now = time.perf_counter_ns()
        self.request = {
            "request_id": "r0",
            "source_order": 0,
            "context_length": 64,
            "value_tokens": 64,
            "arrival_ns": now - 1,
            "deadline_ns": now + 2_000_000_000,
            "broker_token": "t0",
        }
        self.prepare_delay = prepare_delay
        self.complete_delay = complete_delay
        self.fail_complete = fail_complete
        self.state = "ready"
        self.commits = 0
        self.completes = 0
        self.aborts = 0
        self.lock = threading.Lock()


class FakeClient:
    def __init__(self, backend):
        self.backend = backend
        self.socket = FakeSocket()

    def peek_edf_fitting(self, now_ns, horizon_ns, guard_ns, bounds_ns):
        time.sleep(self.backend.prepare_delay)
        with self.backend.lock:
            if self.backend.state != "ready":
                return None
            self.backend.state = "held"
            return dict(self.backend.request)

    def commit(self, request):
        with self.backend.lock:
            if self.backend.state != "held":
                raise ValueError("not held")
            self.backend.state = "inflight"
            self.backend.commits += 1

    def release(self, request):
        with self.backend.lock:
            if self.backend.state != "held":
                raise ValueError("not held")
            self.backend.state = "ready"
            self.backend.aborts += 1

    def complete(self, request, returned_ns):
        time.sleep(self.backend.complete_delay)
        if self.backend.fail_complete:
            raise TimeoutError("injected complete timeout")
        with self.backend.lock:
            if self.backend.state != "inflight":
                raise ValueError("not inflight")
            self.backend.state = "completed"
            self.backend.completes += 1

    def finalize(self, now_ns):
        return {}

    def summary(self):
        return {"state": self.backend.state}

    def close(self):
        pass


class Factory:
    def __init__(self, backend):
        self.backend = backend

    def __call__(self, path, home, epoch_ns, trace_sha256):
        return FakeClient(self.backend)


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            raise AssertionError("condition did not become true")
        time.sleep(0.001)


class PipelinedGlobalTraceClientTest(unittest.TestCase):
    def make_client(self, backend):
        return PipelinedGlobalTraceClient(
            "unused", 0, 1, "sealed", 5,
            client_factory=Factory(backend),
        )

    def test_prepare_and_complete_do_not_block_ran_path(self):
        backend = FakeBackend(prepare_delay=0.1, complete_delay=0.1)
        client = self.make_client(backend)
        now = time.perf_counter_ns()
        begin = time.monotonic()
        self.assertIsNone(client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        ))
        self.assertLess((time.monotonic() - begin) * 1000, 20)
        wait_for(lambda: client.rpc_telemetry()["state"]["staged"])
        request = client.peek_edf_fitting(
            time.perf_counter_ns(), time.perf_counter_ns() + 1_000_000_000,
            9_000_000, {64: 35_000_000},
        )
        self.assertIsNotNone(request)
        self.assertTrue(client.commit(request))
        begin = time.monotonic()
        self.assertTrue(client.complete(request, time.perf_counter_ns()))
        self.assertLess((time.monotonic() - begin) * 1000, 20)
        client.finalize(time.perf_counter_ns())
        self.assertEqual((backend.commits, backend.completes), (1, 1))
        telemetry = client.rpc_telemetry()
        self.assertEqual(telemetry["critical_path_operations"], ["commit"])
        self.assertEqual(telemetry["state"]["inflight"], 0)
        client.close()

    def test_short_window_retains_token_then_later_window_uses_it(self):
        backend = FakeBackend()
        client = self.make_client(backend)
        now = time.perf_counter_ns()
        client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        )
        wait_for(lambda: client.rpc_telemetry()["state"]["staged"])
        now = time.perf_counter_ns()
        self.assertIsNone(client.peek_edf_fitting(
            now, now + 1_000_000, 9_000_000, {64: 35_000_000}
        ))
        self.assertTrue(client.rpc_telemetry()["state"]["staged"])
        now = time.perf_counter_ns()
        request = client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        )
        self.assertIsNotNone(request)
        self.assertTrue(client.commit(request))
        self.assertTrue(client.complete(request, time.perf_counter_ns()))
        client.finalize(time.perf_counter_ns())
        client.close()

    def test_request_past_own_slo_is_aborted_without_launch(self):
        backend = FakeBackend()
        backend.request["deadline_ns"] = time.perf_counter_ns() + 1_000_000
        client = self.make_client(backend)
        now = time.perf_counter_ns()
        client.peek_edf_fitting(
            now, now + 1_000_000_000, 0, {64: 1}
        )
        wait_for(lambda: client.rpc_telemetry()["state"]["staged"])
        time.sleep(0.005)
        now = time.perf_counter_ns()
        self.assertIsNone(client.peek_edf_fitting(
            now, now + 1_000_000_000, 0, {64: 1}
        ))
        wait_for(lambda: backend.aborts == 1)
        self.assertEqual(backend.commits, 0)
        client.finalize(time.perf_counter_ns())
        client.close()

    def test_ambiguous_deferred_complete_quarantines_without_reuse(self):
        backend = FakeBackend(fail_complete=True)
        client = self.make_client(backend)
        now = time.perf_counter_ns()
        client.peek_edf_fitting(
            now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
        )
        wait_for(lambda: client.rpc_telemetry()["state"]["staged"])
        request = client.peek_edf_fitting(
            time.perf_counter_ns(), time.perf_counter_ns() + 1_000_000_000,
            9_000_000, {64: 35_000_000},
        )
        self.assertTrue(client.commit(request))
        self.assertTrue(client.complete(request, time.perf_counter_ns()))
        wait_for(lambda: not client.enabled)
        telemetry = client.rpc_telemetry()
        self.assertEqual(telemetry["state"]["inflight"], 1)
        self.assertEqual(client.faults[0]["operation"], "complete")
        self.assertIsNone(client.peek_edf_fitting(
            time.perf_counter_ns(), time.perf_counter_ns() + 1_000_000_000,
            9_000_000, {64: 35_000_000},
        ))
        client.close()


if __name__ == "__main__":
    unittest.main()
