#!/usr/bin/env python3
"""Deterministically reproduce V14's staged-token leak and verify V15's fix."""

import argparse
import json
import threading
import time
from pathlib import Path

from pipelined_global_trace_client import PipelinedGlobalTraceClient
from single_token_pipelined_global_trace_client import (
    SingleTokenPipelinedGlobalTraceClient,
)


class FakeSocket:
    def settimeout(self, value):
        self.timeout = value


class Backend:
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


class Client:
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
        del request

    def release(self, request):
        with self.backend.lock:
            self.backend.requests[request["source_order"]]["state"] = "ready"

    def complete(self, request, returned_ns):
        del request, returned_ns

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
        return Client(self.backend)


def wait_for(predicate, timeout=1.0):
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() >= deadline:
            return False
        time.sleep(0.001)
    return True


def exercise(client_class):
    backend = Backend()
    client = client_class(
        "unused", 0, 1, "sealed", 5, client_factory=Factory(backend)
    )
    now = time.perf_counter_ns()
    client.peek_edf_fitting(
        now, now + 1_000_000_000, 9_000_000, {64: 35_000_000}
    )
    if not wait_for(lambda: client.rpc_telemetry()["state"]["staged"]):
        raise RuntimeError("first token did not become staged")
    for _ in range(20):
        now = time.perf_counter_ns()
        client.peek_edf_fitting(
            now, now + 1_000_000, 9_000_000, {64: 35_000_000}
        )
        time.sleep(0.001)
    time.sleep(0.02)
    telemetry = client.rpc_telemetry()
    states = [request["state"] for request in backend.requests]
    broker_held = states.count("held")
    client_tracked = int(telemetry["state"]["staged"]) + telemetry["state"]["offered"]
    result = {
        "prepare_calls": backend.prepare_calls,
        "broker_request_states": states,
        "broker_held_tokens": broker_held,
        "client_tracked_unlaunched_tokens": client_tracked,
        "untracked_held_tokens": broker_held - client_tracked,
    }
    client.close()
    return result


def verify():
    v14 = exercise(PipelinedGlobalTraceClient)
    v15 = exercise(SingleTokenPipelinedGlobalTraceClient)
    gates = {
        "v14_regression_reproduced": (
            v14["prepare_calls"] >= 2 and v14["untracked_held_tokens"] == 1
        ),
        "v15_one_prepare_while_staged": v15["prepare_calls"] == 1,
        "v15_no_untracked_held_token": v15["untracked_held_tokens"] == 0,
    }
    return {
        "schema": "softwall-v14-v15-staged-ownership-regression-v1",
        "scope": "deterministic two-request fake broker; no CUDA or timing claim",
        "v14": v14,
        "v15": v15,
        "gates": gates,
        "all_pass": all(gates.values()),
        "interpretation": (
            "V14 physical campaigns ended with zero outstanding tokens, but the "
            "implementation admitted an unexercised second-prepare branch. V15 "
            "suppresses prepare while a staged/offered token is owned."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = verify()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    if not result["all_pass"]:
        raise SystemExit("staged ownership regression gate failed")


if __name__ == "__main__":
    main()
