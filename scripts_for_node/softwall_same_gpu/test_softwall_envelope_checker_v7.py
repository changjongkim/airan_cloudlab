#!/usr/bin/env python3

import unittest

from softwall_envelope_checker_v7 import (
    decision_time_exchange_windows,
    endpoint_admission,
    endpoint_admission_for_disjoint_homes,
    evaluate,
    max_conditional_release_by_home,
)


class EnvelopeCheckerV7Test(unittest.TestCase):
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
        self.assertEqual(result["endpoint_admitted_cells_by_home"], [4])
        self.assertEqual(result["home_max_released_recovery_slack_ms"], [36])
        self.assertTrue(all(item["static_admissible"]
                            for item in result["ai_classes"]))

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
        self.assertGreater(result["mandatory_demand_ms"],
                           result["mandatory_capacity_ms"])

    def test_exchange_only_window(self):
        scenario = self.base()
        scenario.update(
            deadline_ms=80,
            endpoint_path_bounds_ms=[5, 5],
            ai_bounds_ms=[35],
        )
        result = evaluate(scenario)
        self.assertFalse(result["ai_classes"][0]["static_admissible"])
        self.assertTrue(
            result["ai_classes"][0]["exchange_only_geometry_candidate"]
        )
        self.assertTrue(result["ai_classes"][0]["exchange_only_candidate"])

    def test_geometry_does_not_imply_decision_time_exchange(self):
        scenario = self.base()
        scenario.update(deadline_ms=80, ai_bounds_ms=[35])
        result = evaluate(scenario)
        ai = result["ai_classes"][0]
        self.assertTrue(ai["exchange_only_geometry_candidate"])
        self.assertFalse(ai["exchange_only_candidate"])
        self.assertEqual(result["status"], "QSN")

    def test_endpoint_cutoff_rejects_early_cells(self):
        decisions = endpoint_admission(12, 155, 2, 12, [25, 25, 25, 25])
        self.assertFalse(decisions[0]["admitted"])
        self.assertFalse(decisions[1]["admitted"])
        self.assertGreater(sum(item["admitted"] for item in decisions), 0)

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

    def test_home_local_delta_does_not_cross_home_boundary(self):
        scenario = self.base()
        scenario.update(
            id="asymmetric",
            gpu_count=2,
            cells=6,
            home_cell_counts=[4, 2],
            deadline_ms=102,
            recovery_bound_ms=25,
            home_endpoint_bounds_ms=[[200], [20, 20]],
            ai_bounds_ms=[55],
        )
        scenario.pop("endpoint_path_bounds_ms")
        result = evaluate(scenario)
        self.assertEqual(result["endpoint_admitted_cells_by_home"], [0, 2])
        self.assertEqual(result["home_max_released_recovery_slack_ms"], [0, 25])
        self.assertEqual(
            result["ai_classes"][0]["exchange_only_candidate_homes"], [1]
        )

    def test_decision_window_charges_observation_and_rejected_recovery(self):
        admission = [
            {
                "home": 0,
                "admitted": True,
                "predicted_finish_ms": 20,
            },
            {
                "home": 0,
                "admitted": True,
                "predicted_finish_ms": 40,
            },
            {
                "home": 0,
                "admitted": False,
                "predicted_finish_ms": None,
            },
        ]
        window = decision_time_exchange_windows(
            admission, [3], 100, 2, 12, [True]
        )[0]
        self.assertEqual(window["latest_nrx_observation_bound_ms"], 40)
        self.assertEqual(window["rejected_conventional_charge_ms"], 12)
        self.assertEqual(window["decision_time_bound_ms"], 52)
        self.assertEqual(window["one_pending_recovery_start_ms"], 86)
        self.assertEqual(window["max_safe_exchange_transaction_ms"], 34)

    def test_ring_depth_caps_synchronous_admission(self):
        decisions = endpoint_admission(
            4, 155, 2, 25, [45, 45], [1, 1]
        )
        self.assertEqual(sum(item["admitted"] for item in decisions), 2)

    def test_rejected_recovery_phase_changes_released_credit(self):
        self.assertEqual(
            max_conditional_release_by_home(
                [2], 25, [4], [True]
            ),
            [25],
        )
        self.assertEqual(
            max_conditional_release_by_home(
                [2], 25, [4], [False]
            ),
            [50],
        )

    def test_global_phase_exposes_fifty_eight_ms_window(self):
        scenario = self.base()
        scenario.update(
            deadline_ms=155,
            recovery_bound_ms=25,
            endpoint_path_bounds_ms=[45, 45],
            endpoint_ring_depths=[1, 1],
            rejected_recovery_before_exchange=False,
            ai_bounds_ms=[40],
            control_fault_model="broker_fail_stop",
            control_fault_qualified=True,
            control_rpc_budget_required=True,
            control_rpc_bound_ms=5,
            control_rpc_count=3,
            control_transaction_budget_ms=15,
            control_budget_accounted_in_admission=True,
        )
        result = evaluate(scenario)
        self.assertEqual(result["endpoint_admitted_cells_by_home"], [2])
        self.assertEqual(result["home_max_released_recovery_slack_ms"], [50])
        self.assertEqual(result["home_max_safe_exchange_transaction_ms"], [58])
        self.assertEqual(
            result["ai_classes"][0]["exchange_only_candidate_homes"], [0]
        )

    def test_partial_admission_caps_delta_by_admitted_jobs(self):
        scenario = self.base()
        scenario.update(
            deadline_ms=105,
            recovery_bound_ms=20,
            endpoint_path_bounds_ms=[60],
            ai_bounds_ms=[30],
        )
        result = evaluate(scenario)
        self.assertEqual(result["endpoint_admitted_cells_by_home"], [1])
        self.assertEqual(result["home_max_released_recovery_slack_ms"], [0])

    def test_max_conditional_release_requires_one_pending_recovery(self):
        self.assertEqual(
            max_conditional_release_by_home([0, 1, 2, 4], 12),
            [0, 0, 12, 36],
        )

    def test_unqualified_control_fault_demotes_feasible_mode(self):
        scenario = self.base()
        scenario.update(
            control_fault_model="post_apply_commit_reply_loss",
            control_fault_qualified=False,
            ambiguous_token_policy="retry-unspecified",
        )
        result = evaluate(scenario)
        self.assertEqual(result["status"], "UQ")

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
