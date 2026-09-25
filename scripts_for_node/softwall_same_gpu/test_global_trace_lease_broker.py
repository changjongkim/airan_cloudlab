#!/usr/bin/env python3

import threading
import unittest

from global_trace_lease_broker import GlobalTraceLeaseState


def trace_with(count=3):
    return {
        "requests": [
            {
                "request_id": index,
                "source_order": index,
                "context_length": 16,
                "value_tokens": 16,
                "arrival_ms": 0,
                "deadline_ms": 1000,
            }
            for index in range(count)
        ]
    }


class GlobalTraceLeaseStateTest(unittest.TestCase):
    def state(self, count=3, policy="global"):
        value = GlobalTraceLeaseState(trace_with(count), "sealed", 2, policy)
        value.initialize(0, 1_000_000_000, "sealed")
        value.initialize(1, 1_000_000_000, "sealed")
        return value

    def prepare(self, state, home):
        return state.prepare(
            home, 1_000_000_001, 1_500_000_000, 1_000,
            {"16": 10_000_000},
        )

    def test_epoch_and_trace_must_match(self):
        state = GlobalTraceLeaseState(trace_with(), "sealed", 2, "global")
        state.initialize(0, 123, "sealed")
        with self.assertRaises(ValueError):
            state.initialize(1, 124, "sealed")
        with self.assertRaises(ValueError):
            state.initialize(1, 123, "wrong")

    def test_prepare_is_exclusive_under_concurrent_homes(self):
        state = self.state(count=1)
        barrier = threading.Barrier(2)
        results = []

        def attempt(home):
            barrier.wait()
            results.append(self.prepare(state, home))

        threads = [threading.Thread(target=attempt, args=(home,)) for home in (0, 1)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(result is not None for result in results), 1)
        self.assertEqual(state.snapshot()["states"], {"held": 1})

    def test_abort_returns_request_to_ready(self):
        state = self.state(count=1)
        first = self.prepare(state, 0)
        state.abort(0, first["broker_token"], 1_000_000_002)
        second = self.prepare(state, 1)
        self.assertEqual(first["request_id"], second["request_id"])
        self.assertNotEqual(first["broker_token"], second["broker_token"])

    def test_commit_complete_is_single_owner(self):
        state = self.state(count=1)
        request = self.prepare(state, 0)
        state.commit(0, request["broker_token"], 1_000_000_002)
        with self.assertRaises(ValueError):
            state.commit(1, request["broker_token"], 1_000_000_003)
        state.complete(0, request["broker_token"], 1_010_000_000)
        summary = state.snapshot()
        self.assertEqual(summary["timely_requests"], 1)
        self.assertEqual(summary["outstanding_tokens"], 0)
        self.assertEqual(summary["duplicate_commit_count"], 0)

    def test_request_that_does_not_fit_is_not_held(self):
        state = self.state(count=1)
        request = state.prepare(
            0, 1_000_000_001, 1_005_000_000, 1_000,
            {"16": 10_000_000},
        )
        self.assertIsNone(request)
        self.assertEqual(state.snapshot()["states"], {"ready": 1})

    def test_static_partition_restricts_owner(self):
        state = self.state(count=2, policy="static_partition")
        home1 = self.prepare(state, 1)
        home0 = self.prepare(state, 0)
        self.assertEqual(home1["source_order"] % 2, 1)
        self.assertEqual(home0["source_order"] % 2, 0)

    def test_finalize_tracks_every_home(self):
        state = self.state(count=1)
        state.finalize(0, 1_100_000_000)
        state.finalize(1, 1_100_000_000)
        self.assertEqual(state.snapshot()["finalized_homes"], [0, 1])


if __name__ == "__main__":
    unittest.main()
