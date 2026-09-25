#!/usr/bin/env python3

import unittest

from softwall_envelope_checker_v8 import evaluate


class EnvelopeCheckerV8GuardTest(unittest.TestCase):
    def scenario(self):
        return {
            "id": "full-control-candidate",
            "gpu_count": 1,
            "cells": 4,
            "home_cell_counts": [4],
            "deadline_ms": 155,
            "guard_ms": 2,
            "recovery_bound_ms": 25,
            "home_memory_feasible": True,
            "timing_bounds_qualified": True,
            "home_endpoint_bounds_ms": [[45, 45]],
            "home_endpoint_ring_depths": [[1, 1]],
            "rejected_recovery_before_exchange": False,
            "ai_bounds_ms": [40],
            "control_fault_model": "broker_fail_stop",
            "control_fault_qualified": True,
            "control_rpc_budget_required": True,
            "control_rpc_bound_ms": 5,
            "control_rpc_count": 3,
            "control_transaction_budget_ms": 15,
            "control_budget_accounted_in_admission": True,
            "ai_completion_guard_ms": 2,
        }

    def test_complete_guard_preserves_candidate_with_one_ms_margin(self):
        result = evaluate(self.scenario())
        ai = result["ai_classes"][0]
        self.assertEqual(result["base_all_fail_slack_ms"], 53)
        self.assertEqual(result["home_max_safe_exchange_transaction_ms"], [58])
        self.assertEqual(ai["effective_transaction_bound_ms"], 57)
        self.assertFalse(ai["static_admissible"])
        self.assertTrue(ai["exchange_only_candidate"])
        self.assertEqual(58 - ai["effective_transaction_bound_ms"], 1)

    def test_completion_guard_can_remove_a_boundary_candidate(self):
        scenario = self.scenario()
        scenario["ai_bounds_ms"] = [42]
        result = evaluate(scenario)
        ai = result["ai_classes"][0]
        self.assertEqual(ai["effective_transaction_bound_ms"], 59)
        self.assertFalse(ai["exchange_only_candidate"])

    def test_negative_completion_guard_is_invalid(self):
        scenario = self.scenario()
        scenario["ai_completion_guard_ms"] = -1
        with self.assertRaises(ValueError):
            evaluate(scenario)

    def test_missing_completion_guard_is_invalid(self):
        scenario = self.scenario()
        scenario.pop("ai_completion_guard_ms")
        with self.assertRaises(ValueError):
            evaluate(scenario)


if __name__ == "__main__":
    unittest.main()
