#!/usr/bin/env python3

import unittest

from analyze_softwall_decision_time_envelope import (
    audit_scenario,
    home_decision_window,
)
from softwall_envelope_checker_v5 import evaluate


class DecisionTimeEnvelopeTest(unittest.TestCase):
    def scenario(self):
        return {
            "id": "conv25",
            "gpu_count": 1,
            "cells": 4,
            "home_cell_counts": [4],
            "deadline_ms": 155,
            "guard_ms": 2,
            "recovery_bound_ms": 25,
            "home_memory_feasible": True,
            "timing_bounds_qualified": True,
            "home_endpoint_bounds_ms": [[45, 45]],
            "ai_bounds_ms": [35, 40, 65, 75],
        }

    def test_observe_all_window_charges_elapsed_nrx_time(self):
        scenario = self.scenario()
        result = evaluate(scenario)
        window = home_decision_window(scenario, result, 0)
        self.assertEqual(window["latest_nrx_observation_bound_ms"], 90)
        self.assertEqual(window["one_pending_recovery_start_ms"], 128)
        self.assertEqual(window["max_safe_exchange_transaction_ms"], 38)

    def test_geometry_only_candidates_are_removed(self):
        scenario = self.scenario()
        result = evaluate(scenario)
        audit = audit_scenario(scenario, result)
        by_bound = {item["bound_ms"]: item for item in audit["ai_classes"]}
        self.assertEqual(by_bound[65]["geometry_exchange_candidate_homes"], [0])
        self.assertEqual(by_bound[65]["decision_time_exchange_candidate_homes"], [])
        self.assertEqual(by_bound[65]["geometry_only_homes"], [0])

    def test_rejected_nrx_jobs_charge_conventional_time(self):
        scenario = self.scenario()
        scenario.update(home_endpoint_bounds_ms=[[60]])
        result = evaluate(scenario)
        window = home_decision_window(scenario, result, 0)
        self.assertEqual(window["admitted_nrx"], 2)
        self.assertEqual(window["rejected_nrx"], 2)
        self.assertEqual(window["rejected_conventional_charge_ms"], 50)
        self.assertEqual(window["decision_time_bound_ms"], 170)
        self.assertEqual(window["max_safe_exchange_transaction_ms"], 0)

    def test_control_budget_is_in_effective_transaction_bound(self):
        scenario = self.scenario()
        scenario.update(
            control_fault_model="broker_fail_stop",
            control_fault_qualified=True,
            control_rpc_budget_required=True,
            control_rpc_bound_ms=5,
            control_rpc_count=3,
            control_transaction_budget_ms=15,
            control_budget_accounted_in_admission=True,
        )
        result = evaluate(scenario)
        audit = audit_scenario(scenario, result)
        by_bound = {item["bound_ms"]: item for item in audit["ai_classes"]}
        self.assertEqual(by_bound[40]["effective_transaction_bound_ms"], 55)
        self.assertEqual(by_bound[40]["decision_time_exchange_candidate_homes"], [])


if __name__ == "__main__":
    unittest.main()
