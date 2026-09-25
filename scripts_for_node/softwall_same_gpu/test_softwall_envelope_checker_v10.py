#!/usr/bin/env python3

import unittest

from softwall_envelope_checker_v10 import evaluate


def scenario(qualified):
    return {
        "id": "pipelined",
        "gpu_count": 2,
        "cells": 8,
        "home_cell_counts": [4, 4],
        "deadline_ms": 155.0,
        "guard_ms": 2.0,
        "recovery_bound_ms": 25.0,
        "home_memory_feasible": True,
        "timing_bounds_qualified": True,
        "ai_bounds_ms": [45.0],
        "evidence": [],
        "home_endpoint_bounds_ms": [[45.0, 45.0], [45.0, 45.0]],
        "home_endpoint_ring_depths": [[1, 1], [1, 1]],
        "rejected_recovery_before_exchange": False,
        "control_fault_model": "pipelined",
        "control_fault_qualified": True,
        "control_rpc_budget_required": True,
        "control_rpc_bound_ms": 7,
        "control_rpc_count": 1,
        "control_transaction_budget_ms": 7,
        "control_budget_accounted_in_admission": True,
        "ai_completion_guard_ms": 2.0,
        "single_token_ownership_required": True,
        "single_token_ownership_qualified": qualified,
        "ownership_contract": "at-most-one-staged-or-offered-token-per-home",
    }


class EnvelopeCheckerV10Test(unittest.TestCase):
    def test_missing_ownership_qualification_demotes_mode(self):
        result = evaluate(scenario(False))
        self.assertEqual(result["status"], "UQ")
        self.assertIn("single-token ownership", result["reason"])

    def test_qualified_ownership_restores_qsu(self):
        result = evaluate(scenario(True))
        self.assertEqual(result["status"], "QSU")
        ai45 = result["ai_classes"][0]
        self.assertEqual(ai45["effective_transaction_bound_ms"], 54.0)
        self.assertTrue(ai45["exchange_only_candidate"])


if __name__ == "__main__":
    unittest.main()
