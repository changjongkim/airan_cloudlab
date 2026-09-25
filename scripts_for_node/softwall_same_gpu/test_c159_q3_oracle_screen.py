#!/usr/bin/env python3.11

import unittest

from c159_q3_oracle_screen import (
    exact_weighted_matching,
    policy_slot,
)


class Q3OracleScreenTests(unittest.TestCase):
    def test_exact_matching_can_reassign_an_earlier_choice(self):
        requests = [
            {"context_length": 16, "arrival_ms": 0, "deadline_ms": 1000,
             "value_tokens": 10},
            {"context_length": 512, "arrival_ms": 0, "deadline_ms": 1000,
             "value_tokens": 100},
        ]
        slots = [
            {"decision_ms": 10, "p": {"completion_by_context_ms": {16: 20, 512: 20}}},
            {"decision_ms": 20, "p": {"completion_by_context_ms": {16: 30}}},
        ]
        result = exact_weighted_matching(requests, slots, "p")
        self.assertEqual(result["timely_value_tokens"], 110)
        self.assertEqual(result["timely_requests"], 2)

    def test_two_recoveries_accept_short_and_reject_long(self):
        state = {"unresolved": 2, "empirical_recovery_ms": 8.0}
        slot = policy_slot(0, state)
        self.assertIn(128, slot["softwall"]["completion_by_context_ms"])
        self.assertNotIn(256, slot["softwall"]["completion_by_context_ms"])
        self.assertNotIn(512, slot["event_empirical"]["completion_by_context_ms"])

    def test_one_recovery_has_same_admission_but_event_finishes_later(self):
        state = {"unresolved": 1, "empirical_recovery_ms": 4.0}
        slot = policy_slot(0, state)
        self.assertEqual(
            set(slot["softwall"]["completion_by_context_ms"]),
            set(slot["event_empirical"]["completion_by_context_ms"]),
        )
        self.assertEqual(
            slot["event_empirical"]["completion_by_context_ms"][512]
            - slot["softwall"]["completion_by_context_ms"][512],
            4.0,
        )


if __name__ == "__main__":
    unittest.main()
