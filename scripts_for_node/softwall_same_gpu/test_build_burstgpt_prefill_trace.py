#!/usr/bin/env python3

import unittest

from build_burstgpt_prefill_trace import (
    bucket_tokens, densest_window, within_second_offsets_ms,
)


class BurstGptTraceBuilderTest(unittest.TestCase):
    def test_bucket_tokens(self):
        self.assertEqual(bucket_tokens(1), (16, False))
        self.assertEqual(bucket_tokens(17), (32, False))
        self.assertEqual(bucket_tokens(512), (512, False))
        self.assertEqual(bucket_tokens(513), (512, True))

    def test_densest_window_uses_earliest_tie(self):
        rows = [{"timestamp_s": value} for value in (0, 1, 10, 11)]
        self.assertEqual(densest_window(rows, 2), (0, 2))

    def test_uniform_within_second_preserves_order_and_counts(self):
        rows = [
            {"timestamp_s": 7}, {"timestamp_s": 7},
            {"timestamp_s": 8}, {"timestamp_s": 8}, {"timestamp_s": 8},
        ]
        self.assertEqual(within_second_offsets_ms(rows, "burst"), [0.0] * 5)
        self.assertEqual(
            within_second_offsets_ms(rows, "uniform"),
            [250.0, 750.0, 1000.0 / 6, 500.0, 5000.0 / 6],
        )


if __name__ == "__main__":
    unittest.main()
