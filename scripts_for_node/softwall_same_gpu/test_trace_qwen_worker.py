#!/usr/bin/env python3

import argparse
import unittest

from trace_qwen_worker import parse_lengths, summarize


class TraceQwenWorkerUtilitiesTest(unittest.TestCase):
    def test_parse_lengths_sorts_and_deduplicates(self):
        self.assertEqual(parse_lengths("128,16,64,16"), (16, 64, 128))

    def test_parse_lengths_rejects_nonpositive(self):
        with self.assertRaises(argparse.ArgumentTypeError):
            parse_lengths("0,16")

    def test_summary(self):
        self.assertEqual(summarize([])["count"], 0)
        result = summarize([3.0, 1.0, 2.0])
        self.assertEqual(result["p50"], 2.0)
        self.assertEqual(result["max"], 3.0)


if __name__ == "__main__":
    unittest.main()
