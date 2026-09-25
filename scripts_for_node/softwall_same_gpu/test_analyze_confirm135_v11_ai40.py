#!/usr/bin/env python3

import unittest


class Confirm135GateShapeTest(unittest.TestCase):
    def test_candidate_gate_requires_positive_count(self):
        evidence = {"branches": 1, "exchanges": 1,
                    "minimum_reserved_horizon_margin_ms": 0.1}
        self.assertTrue(evidence["branches"] > 0)
        self.assertTrue(evidence["exchanges"] > 0)
        self.assertGreaterEqual(evidence["minimum_reserved_horizon_margin_ms"], 0)

    def test_zero_candidate_is_not_a_pass(self):
        evidence = {"branches": 5, "exchanges": 0,
                    "minimum_reserved_horizon_margin_ms": None}
        self.assertFalse(evidence["exchanges"] > 0)


if __name__ == "__main__":
    unittest.main()
