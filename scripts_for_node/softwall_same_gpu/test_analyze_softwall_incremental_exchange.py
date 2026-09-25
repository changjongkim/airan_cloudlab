#!/usr/bin/env python3

import unittest

from analyze_softwall_incremental_exchange import (
    audit_scenario,
    enumerate_home_prefix_branches,
)
from softwall_envelope_checker_v7 import evaluate


class IncrementalExchangeTest(unittest.TestCase):
    def scenario(self):
        return {
            "id": "full-control",
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
            "ai_bounds_ms": [35, 40, 65, 75],
            "control_fault_model": "broker_fail_stop",
            "control_fault_qualified": True,
            "control_rpc_budget_required": True,
            "control_rpc_bound_ms": 5,
            "control_rpc_count": 3,
            "control_transaction_budget_ms": 15,
            "control_budget_accounted_in_admission": True,
        }

    def test_two_early_successes_open_fifty_eight_ms(self):
        scenario = self.scenario()
        result = evaluate(scenario)
        branches = enumerate_home_prefix_branches(scenario, result, 0)
        best = max(branches, key=lambda item: item["max_safe_transaction_ms"])
        self.assertEqual(best["observed_at_ms"], 45)
        self.assertEqual(best["successes"], 2)
        self.assertEqual(best["remaining_recovery_obligations"], 2)
        self.assertEqual(best["earliest_recovery_start_ms"], 103)
        self.assertEqual(best["max_safe_transaction_ms"], 58)

    def test_current_phase_already_admits_effective_fifty_five_ms_class(self):
        scenario = self.scenario()
        result = evaluate(scenario)
        audit = audit_scenario(scenario, result)
        by_bound = {item["bound_ms"]: item for item in audit["ai_classes"]}
        self.assertEqual(by_bound[35]["new_incremental_homes"], [])
        self.assertEqual(by_bound[40]["effective_transaction_bound_ms"], 55)
        self.assertEqual(by_bound[40]["observe_all_exchange_homes"], [0])
        self.assertEqual(by_bound[40]["incremental_exchange_homes"], [0])
        self.assertEqual(by_bound[40]["new_incremental_homes"], [])
        self.assertEqual(by_bound[65]["incremental_exchange_homes"], [])

    def test_all_admitted_success_still_leaves_rejected_recoveries(self):
        scenario = self.scenario()
        result = evaluate(scenario)
        branches = enumerate_home_prefix_branches(scenario, result, 0)
        witness = next(item for item in branches if item["successes"] == 2)
        self.assertEqual(witness["remaining_recovery_obligations"], 2)

    def test_no_success_branch_never_releases_capacity(self):
        scenario = self.scenario()
        result = evaluate(scenario)
        branches = enumerate_home_prefix_branches(scenario, result, 0)
        self.assertTrue(branches)
        self.assertTrue(all(item["successes"] > 0 for item in branches))


if __name__ == "__main__":
    unittest.main()
