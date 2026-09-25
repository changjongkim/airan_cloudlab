#!/usr/bin/env python3

import unittest

from build_c159_partitioned_trace import (
    bucket_tokens,
    select_nonoverlapping_windows,
    within_second_offsets_ms,
)


class C159TraceBuilderTest(unittest.TestCase):
    def test_bucket_tokens(self):
        self.assertEqual(bucket_tokens(1), (16, False))
        self.assertEqual(bucket_tokens(17), (32, False))
        self.assertEqual(bucket_tokens(513), (512, True))

    def test_density_order_and_nonoverlap(self):
        rows = []
        source_row = 0
        for timestamp, count in ((0, 4), (1, 4), (10, 5), (11, 5), (20, 3)):
            for _ in range(count):
                rows.append({"timestamp_s": timestamp, "source_row": source_row})
                source_row += 1
        selected = select_nonoverlapping_windows(rows, window_s=2, count=2)
        self.assertEqual(selected[0], {"start_s": 10, "end_s": 12, "requests": 10})
        self.assertEqual(selected[1], {"start_s": 0, "end_s": 2, "requests": 8})

    def test_earliest_tie(self):
        rows = [
            {"timestamp_s": timestamp, "source_row": index}
            for index, timestamp in enumerate((0, 1, 10, 11))
        ]
        selected = select_nonoverlapping_windows(rows, window_s=2, count=2)
        self.assertEqual([item["start_s"] for item in selected], [0, 10])

    def test_uniform_offsets_preserve_count(self):
        rows = [
            {"timestamp_s": 7, "source_row": 0},
            {"timestamp_s": 7, "source_row": 1},
            {"timestamp_s": 8, "source_row": 2},
        ]
        self.assertEqual(within_second_offsets_ms(rows, "burst"), [0.0, 0.0, 0.0])
        self.assertEqual(within_second_offsets_ms(rows, "uniform"), [250.0, 750.0, 500.0])


if __name__ == "__main__":
    unittest.main()
