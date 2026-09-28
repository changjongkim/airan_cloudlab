#!/usr/bin/env python3.11
"""Checks for C172 two-node aggregation helpers."""

from __future__ import annotations

import unittest

from analyze_c172_two_node import rule_of_three_zero_upper


class C172TwoNodeTest(unittest.TestCase):
    def test_rule_of_three(self) -> None:
        self.assertEqual(rule_of_three_zero_upper(120), 0.025)
        self.assertAlmostEqual(rule_of_three_zero_upper(119), 3 / 119)
        self.assertIsNone(rule_of_three_zero_upper(0))


if __name__ == "__main__":
    unittest.main()
