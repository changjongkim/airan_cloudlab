#!/usr/bin/env python3

import unittest

from trace_background_client import TraceRequestQueue


class TraceRequestQueueTest(unittest.TestCase):
    def setUp(self):
        self.trace = {"requests": [
            {"request_id": "a", "arrival_ms": 0, "deadline_ms": 100,
             "source_order": 0, "context_length": 16, "value_tokens": 10},
            {"request_id": "b", "arrival_ms": 50, "deadline_ms": 80,
             "source_order": 1, "context_length": 32, "value_tokens": 20},
        ]}

    def test_visibility_and_edf(self):
        queue = TraceRequestQueue(self.trace, 1_000_000_000)
        self.assertEqual(queue.peek_edf(1_000_000_000)["request_id"], "a")
        self.assertEqual(queue.peek_edf(1_060_000_000)["request_id"], "b")

    def test_expiry_and_completion(self):
        queue = TraceRequestQueue(self.trace, 0)
        first = queue.peek_edf(0)
        queue.complete(first, 10_000_000)
        queue.observe(90_000_000)
        self.assertEqual(len(queue.expired), 1)
        self.assertEqual(queue.summary()["timely_requests"], 1)

    def test_edf_fitting_skips_long_request_only_within_same_cohort(self):
        trace = {"requests": [
            {"request_id": "long", "arrival_ms": 0, "deadline_ms": 100,
             "source_order": 0, "context_length": 512, "value_tokens": 10},
            {"request_id": "short", "arrival_ms": 0, "deadline_ms": 100,
             "source_order": 1, "context_length": 256, "value_tokens": 20},
            {"request_id": "later", "arrival_ms": 0, "deadline_ms": 200,
             "source_order": 2, "context_length": 16, "value_tokens": 30},
        ]}
        queue = TraceRequestQueue(trace, 0)
        selected = queue.peek_edf_fitting(
            0, 60_000_000, 2_000_000,
            {16: 10_000_000, 256: 40_000_000, 512: 80_000_000},
        )
        self.assertEqual(selected["request_id"], "short")
        self.assertNotEqual(selected["request_id"], "later")


if __name__ == "__main__":
    unittest.main()
