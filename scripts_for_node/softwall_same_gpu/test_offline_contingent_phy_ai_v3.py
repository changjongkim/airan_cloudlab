#!/usr/bin/env python3.11
"""Independent recourse and conditional-capacity checks for the seven-AI screen."""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_recovery_exchange import Job, State
from offline_contingent_phy_ai_v2 import exact_joint as subset_oracle
from offline_contingent_phy_ai_v3 import exact_joint, feasible_schedule


class SevenAiCapacityTest(unittest.TestCase):
    def test_optional_job_dp_matches_independent_subset_oracle(self) -> None:
        rng = random.Random(20993001)
        for _ in range(100):
            recovery = tuple(Job(f"r{j}", 0, rng.randint(2, 8),
                                 rng.randint(8, 40))
                             for j in range(rng.randint(0, 4)))
            ai = tuple(Job(f"a{j}", 0, rng.randint(2, 8),
                           rng.randint(8, 40), rng.randint(1, 3))
                       for j in range(rng.randint(0, 4)))
            state = State(0, recovery, ai)
            got = exact_joint(state)
            oracle = subset_oracle(state)
            self.assertEqual((got["value"], got["admitted"]),
                             (oracle["value"], oracle["admitted"]))

    def test_seventh_ai_requires_conditional_recovery_release(self) -> None:
        # One AI and one conventional unit completed before the 30 ms event.
        # Three all-fail conventional obligations remain, but only one after
        # both selected NeuralRx results succeed.
        all_fail = tuple(Job(f"conv{j}", 30, 12, 153)
                         for j in ("a", "b", "d"))
        two_success = (Job("convd", 30, 12, 153),)
        six_remaining_ai = tuple(Job(f"ai{i}", 30, 15, 153, 1)
                                 for i in range(1, 7))
        self.assertIsNone(feasible_schedule(30, all_fail + six_remaining_ai))
        success_schedule = feasible_schedule(30, two_success + six_remaining_ai)
        self.assertIsNotNone(success_schedule)
        self.assertEqual(success_schedule[-1]["finish_ms"], 132)


if __name__ == "__main__":
    unittest.main()
