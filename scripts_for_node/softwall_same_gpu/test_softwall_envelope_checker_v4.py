#!/usr/bin/env python3

import unittest

from softwall_envelope_checker_v4 import (
    endpoint_admission,
    endpoint_admission_for_disjoint_homes,
    evaluate,
)


class EnvelopeCheckerTest(unittest.TestCase):
    def base(self):
        return {
            "id": "test",
            "gpu_count": 1,
            "cells": 4,
            "deadline_ms": 155,
            "guard_ms": 2,
            "recovery_bound_ms": 12,
            "home_memory_feasible": True,
            "timing_bounds_qualified": True,
            "endpoint_path_bounds_ms": [45, 45],
            "ai_bounds_ms": [35, 75],
        }

    def test_current_four_cell_mode_is_safe_and_static_ai_fits(self):
        result = evaluate(self.base())
        self.assertEqual(result["status"], "QSU")
        self.assertEqual(result["base_all_fail_slack_ms"], 105)
        self.assertEqual(result["endpoint_admitted_cells"], 4)
        self.assertTrue(all(item["static_admissible"] for item in result["ai_classes"]))

    def test_memory_boundary_dominates_timing_arithmetic(self):
        scenario = self.base()
        scenario.update(cells=8, home_memory_feasible=False)
        result = evaluate(scenario)
        self.assertEqual(result["status"], "MI")
        self.assertIn("residency", result["reason"])

    def test_mandatory_arithmetic_boundary(self):
        scenario = self.base()
        scenario.update(cells=13)
        result = evaluate(scenario)
        self.assertEqual(result["status"], "MI")
        self.assertGreater(result["mandatory_demand_ms"], result["mandatory_capacity_ms"])

    def test_exchange_only_window(self):
        scenario = self.base()
        scenario.update(deadline_ms=80, ai_bounds_ms=[35])
        result = evaluate(scenario)
        self.assertFalse(result["ai_classes"][0]["static_admissible"])
        self.assertTrue(result["ai_classes"][0]["exchange_only_candidate"])

    def test_endpoint_cutoff_rejects_early_cells(self):
        decisions = endpoint_admission(12, 155, 2, 12, [25, 25, 25, 25])
        self.assertFalse(decisions[0]["admitted"])
        self.assertFalse(decisions[1]["admitted"])
        self.assertGreater(sum(item["admitted"] for item in decisions), 0)

    def test_sharded_homes_compose_independent_recovery_capacity(self):
        scenario = self.base()
        scenario.update(
            id="sharded",
            gpu_count=2,
            cells=8,
            home_cell_counts=[4, 4],
            home_memory_feasible=True,
            timing_bounds_qualified=False,
        )
        result = evaluate(scenario)
        self.assertEqual(result["status"], "UQ")
        self.assertEqual(result["home_mandatory_demand_ms"], [48, 48])
        self.assertEqual(result["home_base_all_fail_slack_ms"], [105, 105])

    def test_disjoint_endpoint_pools_preserve_home_local_capacity(self):
        decisions = endpoint_admission_for_disjoint_homes(
            [4, 4], 155, 2, 12, [[45, 45], [45, 45]]
        )
        self.assertEqual(len(decisions), 8)
        self.assertTrue(all(item["admitted"] for item in decisions))
        self.assertEqual(
            {item["endpoint"].split(":")[0] for item in decisions},
            {"h0", "h1"},
        )

    def test_disjoint_scenario_reports_endpoint_scope(self):
        scenario = self.base()
        scenario.update(
            id="disjoint",
            gpu_count=2,
            cells=8,
            home_cell_counts=[4, 4],
            home_endpoint_bounds_ms=[[45, 45], [45, 45]],
        )
        scenario.pop("endpoint_path_bounds_ms")
        result = evaluate(scenario)
        self.assertEqual(result["status"], "QSU")
        self.assertEqual(result["endpoint_scope"], "disjoint-per-home")
        self.assertEqual(result["endpoint_admitted_cells"], 8)

    def test_unqualified_control_fault_demotes_feasible_mode(self):
        scenario = self.base()
        scenario.update(
            control_fault_model="post_apply_commit_reply_loss",
            control_fault_qualified=False,
            ambiguous_token_policy="retry-unspecified",
        )
        result = evaluate(scenario)
        self.assertEqual(result["status"], "UQ")
        self.assertIn("control-plane", result["reason"])

    def test_quarantined_commit_fault_preserves_qualified_mode(self):
        scenario = self.base()
        scenario.update(
            control_fault_model="post_apply_commit_reply_loss",
            control_fault_qualified=True,
            ambiguous_token_policy="no-retry; affected-home-global-AI-disabled",
        )
        result = evaluate(scenario)
        self.assertEqual(result["status"], "QSU")
        self.assertEqual(
            result["ambiguous_token_policy"],
            "no-retry; affected-home-global-AI-disabled",
        )

    def test_bounded_broker_fault_without_admission_charge_is_unqualified(self):
        scenario = self.base()
        scenario.update(
            control_fault_model="broker_fail_stop",
            control_fault_qualified=True,
            control_rpc_budget_required=True,
            control_rpc_bound_ms=5,
            control_rpc_count=3,
            control_transaction_budget_ms=0,
            control_budget_accounted_in_admission=False,
        )
        result = evaluate(scenario)
        self.assertEqual(result["status"], "UQ")
        self.assertFalse(result["control_budget_qualified"])

    def test_partial_control_budget_is_unqualified(self):
        scenario = self.base()
        scenario.update(
            control_fault_model="broker_fail_stop",
            control_fault_qualified=True,
            control_rpc_budget_required=True,
            control_rpc_bound_ms=5,
            control_rpc_count=3,
            control_transaction_budget_ms=10,
            control_budget_accounted_in_admission=True,
        )
        result = evaluate(scenario)
        self.assertEqual(result["status"], "UQ")
        self.assertEqual(result["required_control_transaction_budget_ms"], 15)

    def test_full_control_budget_qualifies_and_reduces_ai_slack(self):
        scenario = self.base()
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
        self.assertEqual(result["status"], "QSU")
        self.assertTrue(result["control_budget_qualified"])
        self.assertEqual(
            result["ai_classes"][0]["effective_transaction_bound_ms"], 50
        )


if __name__ == "__main__":
    unittest.main()
