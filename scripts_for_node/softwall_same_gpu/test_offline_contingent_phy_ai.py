#!/usr/bin/env python3.11
"""Structural tests for the small two-stage decision screen."""

import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_contingent_phy_ai import (
    AiUnit, Cell, Problem, exact_joint_policy, greedy_radio_then_ai,
)


class ConditionalPhyAiScreenTest(unittest.TestCase):
    def test_all_fail_obligation_can_make_every_plan_infeasible(self) -> None:
        problem = Problem(
            cells=(
                Cell("a", 10, 6, 5, 0.8, 0.4),
                Cell("b", 10, 6, 5, 0.8, 0.4),
            ),
            ai=(), min_expected_radio_rescues=0.4,
            nrx_endpoint_capacity=2,
        )
        self.assertIsNone(exact_joint_policy(problem))
        self.assertIsNone(greedy_radio_then_ai(problem))

    def test_early_ai_needs_a_valid_all_fail_certificate(self) -> None:
        problem = Problem(
            cells=(Cell("a", 20, 6, 5, 0.5, 0.25),),
            ai=(AiUnit("x", 8, 4, 4, 1),),
            min_expected_radio_rescues=0.2,
            nrx_endpoint_capacity=1,
        )
        exact = exact_joint_policy(problem)
        self.assertIsNotNone(exact)
        self.assertEqual(exact["selected_nrx"], ["a"])
        self.assertEqual(exact["early_ai"], ["x"])
        self.assertEqual(exact["expected_ai_value"], 1)
        self.assertEqual(
            exact["all_fail_recovery_certificate"],
            [{"job": "conv_a", "start_ms": 5, "finish_ms": 11}],
        )
        self.assertEqual(greedy_radio_then_ai(problem)["expected_ai_value"], 1)

    def test_incremental_rescue_cannot_exceed_neural_success(self) -> None:
        with self.assertRaises(ValueError):
            Cell("a", 20, 6, 5, 0.4, 0.5)

    def test_greedy_takes_equal_ai_radio_improvement(self) -> None:
        problem = Problem(
            cells=(
                Cell("a", 130, 25, 50, 0.5, 0.10),
                Cell("b", 130, 25, 50, 0.8, 0.13),
                Cell("c", 130, 25, 50, 0.6, 0.12),
            ),
            ai=(AiUnit("x", 50, 15, 15, 1),
                AiUnit("y", 90, 15, 15, 1)),
            min_expected_radio_rescues=0.10,
            nrx_endpoint_capacity=2,
        )
        exact = exact_joint_policy(problem)
        greedy = greedy_radio_then_ai(problem)
        self.assertEqual(exact["expected_ai_value"], greedy["expected_ai_value"])
        self.assertEqual(exact["radio_gain"], greedy["radio_gain"])

    def test_floating_point_ai_tie_does_not_override_radio_gain(self) -> None:
        # A held-out feature scenario once produced expected AI values
        # 3.0000000000000004 and 3.0 for two physically equivalent plans.
        leaves = {
            0: (0.31647940074906367, 0.1017478152309613),
            5: (0.0006242197253433209, 0.0006242197253433209),
            6: (0.799625468164794, 0.13420724094881398),
        }
        ids = (6, 6, 5, 0)
        deadlines = (119, 140, 145, 136)
        problem = Problem(
            cells=tuple(Cell(chr(97 + i), deadlines[i], 25, 50,
                             *leaves[ids[i]]) for i in range(4)),
            ai=(AiUnit("ai0", 70, 15, 15, 3),
                AiUnit("ai1", 53, 30, 30, 1)),
            min_expected_radio_rescues=0.1,
            nrx_endpoint_capacity=2,
        )
        exact = exact_joint_policy(problem)
        greedy = greedy_radio_then_ai(problem)
        self.assertAlmostEqual(exact["expected_ai_value"], greedy["expected_ai_value"])
        self.assertGreaterEqual(exact["radio_gain"] + 1e-12,
                                greedy["radio_gain"])


if __name__ == "__main__":
    unittest.main()
