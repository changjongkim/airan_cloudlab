#!/usr/bin/env python3.11
"""Cross-check the faster decision screen against the frozen v1 oracle."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_contingent_phy_ai_v2 import (
    AiUnit, Cell, Problem, exact_joint_policy, feasible_schedule,
)
from offline_recovery_exchange import Job, feasible_schedule as permutation_schedule


class QualifiedDurationScreenTest(unittest.TestCase):
    def test_subset_dp_matches_permutation_feasibility_and_finish(self) -> None:
        rng = random.Random(20922001)
        for _ in range(160):
            now = rng.randint(0, 10)
            jobs = tuple(
                Job(f"j{i}", rng.randint(0, now), rng.randint(1, 8),
                    rng.randint(now + 2, now + 32))
                for i in range(rng.randint(0, 7))
            )
            got = feasible_schedule(now, jobs)
            oracle = permutation_schedule(now, jobs)
            self.assertEqual(got is None, oracle is None)
            if got is not None:
                self.assertEqual(got[-1]["finish_ms"] if got else now,
                                 oracle[-1]["finish_ms"] if oracle else now)

    def test_no_more_than_one_early_ai_unit(self) -> None:
        problem = Problem(
            cells=(Cell("a", 60, 12, 30, 0.8, 0.4),),
            ai=(AiUnit("x", 20, 15, 15, 1),
                AiUnit("y", 25, 15, 15, 1)),
            min_expected_radio_rescues=0.2,
            nrx_endpoint_capacity=1,
        )
        result = exact_joint_policy(problem)
        self.assertIsNotNone(result)
        self.assertLessEqual(len(result["early_ai"]), 1)


if __name__ == "__main__":
    unittest.main()
