#!/usr/bin/env python3.11

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from offline_online_recovery_v4 import (
    AiJob, Cell, Config, branch_expectation, exact_joint_policy,
    joint_one_swap_policy, radio_guarded_joint_policy,
    radio_guarded_one_swap_policy, replay_branch,
)


CFG = Config(153, 15, 45, 12, 6, 2)
AI = tuple(AiJob(f"ai{i}", 0 if i < 5 else 60 + 30 * (i - 5),
                 153, 50, 22, 1) for i in range(7))
CELLS = tuple(Cell(chr(97 + i), 0.45 + i * 0.05,
                   0.08 + i * 0.02) for i in range(4))


class OnlineRecoveryV4Test(unittest.TestCase):
    def test_every_branch_completes_all_mandatory_radio(self) -> None:
        selected = frozenset(("b", "d"))
        for b in (False, True):
            for d in (False, True):
                replay = replay_branch(CELLS, selected, {"b": b, "d": d},
                                       AI, CFG)
                self.assertTrue(replay["safe"])
                expected = {"a", "c"} | ({"b"} if not b else set()) \
                    | ({"d"} if not d else set())
                self.assertEqual(set(replay["completed_conv"]), expected)
                self.assertTrue(all(action["finish_ms"] <= 153 + 1e-9
                                    for action in replay["actions"]))

    def test_recovery_release_can_increase_ai_completion(self) -> None:
        selected = frozenset(("a", "b"))
        failed = replay_branch(CELLS, selected, {"a": False, "b": False},
                               AI, CFG)
        succeeded = replay_branch(CELLS, selected, {"a": True, "b": True},
                                  AI, CFG)
        self.assertGreater(succeeded["completed_ai_value"],
                           failed["completed_ai_value"])

    def test_force_all_mandatory_removes_subset_capacity_effect(self) -> None:
        values = []
        for selected in (frozenset(("a",)), frozenset(("b", "d"))):
            result = branch_expectation(CELLS, selected, AI, CFG,
                                        force_all_mandatory=True)
            self.assertTrue(result["safe"])
            values.append(result["expected_ai_value"])
        self.assertAlmostEqual(values[0], values[1])

    def test_one_swap_matches_exact_on_small_random_states(self) -> None:
        rng = random.Random(21002001)
        for _ in range(100):
            cells = tuple(Cell(chr(97 + i), p := rng.uniform(0.2, 0.9),
                               rng.uniform(0.0, p)) for i in range(4))
            floor = rng.choice((0.05, 0.1, 0.2))
            exact = exact_joint_policy(cells, AI[:5], CFG, floor)
            greedy = joint_one_swap_policy(cells, AI[:5], CFG, floor)
            self.assertEqual(exact is None, greedy is None)
            if exact is not None:
                self.assertAlmostEqual(exact["expected_visible_ai_value"],
                                       greedy["expected_visible_ai_value"])

    def test_radio_guard_is_enforced_and_greedy_matches(self) -> None:
        exact = radio_guarded_joint_policy(CELLS, AI[:5], CFG, 0.1, 0.02)
        greedy = radio_guarded_one_swap_policy(CELLS, AI[:5], CFG, 0.1, 0.02)
        self.assertIsNotNone(exact)
        self.assertGreaterEqual(exact["radio_gain"] + 1e-12,
                                exact["max_radio_gain"] - 0.02)
        self.assertEqual(exact["selected_nrx"], greedy["selected_nrx"])


if __name__ == "__main__":
    unittest.main()
