#!/usr/bin/env python3.11
"""CPU checks for the frozen C172 diagnostic design."""

from __future__ import annotations

import math
import unittest

from c159_q2_classes import CLASS_BOUNDS_MS
from c162_boundary_plan import build_boundary_plan
from c172_debt_blind_cases import SCENARIOS, timing_cell_contains
from integrated_shared_recovery_plan_v1 import IntegratedScenarioConfig, REQUEST_ORDER


class C172DesignTest(unittest.TestCase):
    def test_scenario_balance_and_integer_cells(self) -> None:
        self.assertEqual(len(SCENARIOS), 5)
        self.assertEqual(len({row.scenario_id for row in SCENARIOS}), 5)
        for row in SCENARIOS:
            self.assertEqual(round(row.target_decision_ms), row.integer_decision_ms)

    def test_softwall_decisions_match_frozen_expectation(self) -> None:
        for row in SCENARIOS:
            config = IntegratedScenarioConfig(
                period_ms=180,
                ai_bound_ms=CLASS_BOUNDS_MS[row.context_length],
            )
            plan = build_boundary_plan(
                REQUEST_ORDER[:row.success_count],
                round(row.target_decision_ms * 1e6),
                config,
                offer_ai=True,
            )
            self.assertEqual(
                plan["lease_decision"]["accepted"],
                row.expected_softwall_lease,
                row.scenario_id,
            )

    def test_declared_service_vector_has_expected_witnesses(self) -> None:
        completion = {
            "E4": 45 + 5 + 65 + 2 * 25,
            "E6a": 88 + 5 + 35 + 25,
            "E6b": 89 + 5 + 35 + 25,
        }
        self.assertEqual(completion["E4"], 165)
        self.assertEqual(completion["E4"] - 153, 12)
        self.assertEqual(completion["E4"] - 155, 10)
        self.assertEqual(completion["E6a"], 153)
        self.assertEqual(completion["E6b"], 154)

    def test_exact_timing_cells(self) -> None:
        by_case = {row.case_id: row for row in SCENARIOS}
        self.assertTrue(timing_cell_contains(by_case["E6a"], 88.0))
        self.assertFalse(timing_cell_contains(by_case["E6a"], 88.000001))
        self.assertTrue(timing_cell_contains(by_case["E6b"], 89.0))
        self.assertTrue(timing_cell_contains(by_case["E6b"], 89.499999))
        self.assertFalse(timing_cell_contains(by_case["E6b"], 89.5))


if __name__ == "__main__":
    unittest.main()
