#!/usr/bin/env python3

import math
import unittest

from analyze_confirm136_v12_requalification import zero_failure_upper


class ZeroFailureBoundTest(unittest.TestCase):
    def test_exact_zero_failure_upper(self):
        upper = zero_failure_upper(100, 0.95)
        self.assertAlmostEqual((1.0 - upper) ** 100, 0.05, places=12)

    def test_more_trials_tighten_bound(self):
        self.assertLess(zero_failure_upper(1000), zero_failure_upper(100))

    def test_rejects_invalid_inputs(self):
        for trials, confidence in ((0, 0.95), (1, 0), (1, 1)):
            with self.assertRaises(ValueError):
                zero_failure_upper(trials, confidence)

    def test_rule_of_three_scale(self):
        self.assertTrue(math.isclose(
            zero_failure_upper(10000), 0.00029953, rel_tol=1e-3
        ))


if __name__ == "__main__":
    unittest.main()
