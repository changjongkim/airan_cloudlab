#!/usr/bin/env python3.11

import unittest

from c161_phase1_faults import (
    ARMS,
    arm_for_index,
    effective_successes,
    parse_arm_order,
)


class C161Phase1FaultTests(unittest.TestCase):
    def test_each_arm_has_exact_frozen_length(self):
        arms = parse_arm_order(",".join(ARMS), 90, 30)
        self.assertEqual([arm_for_index(i, arms, 30) for i in (0, 29, 30, 59, 60, 89)],
                         [ARMS[0], ARMS[0], ARMS[1], ARMS[1], ARMS[2], ARMS[2]])

    def test_reverse_holdout_order_is_valid(self):
        self.assertEqual(
            parse_arm_order(",".join(reversed(ARMS)), 90, 30),
            tuple(reversed(ARMS)),
        )

    def test_missing_duplicate_or_wrong_length_is_rejected(self):
        for text, iterations, length in (
            ("no_fault,correlated_all_fail", 60, 30),
            ("no_fault,no_fault,latest_start_nonlaunch", 90, 30),
            (",".join(ARMS), 89, 30),
        ):
            with self.assertRaises(ValueError):
                parse_arm_order(text, iterations, length)

    def test_only_correlated_arm_suppresses_successes(self):
        actual = (("home0", "r0"), ("home1", "r0"))
        self.assertEqual(effective_successes("correlated_all_fail", actual), ())
        self.assertEqual(effective_successes("no_fault", actual), actual)
        self.assertEqual(effective_successes("latest_start_nonlaunch", actual), actual)


if __name__ == "__main__":
    unittest.main()
