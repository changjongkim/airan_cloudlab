#!/usr/bin/env python3.11

import math
import unittest

from c159_q2_classes import CLASS_BOUNDS_MS
from c162_boundary_cases import CASES, case_for_sequence
from c162_boundary_plan import build_boundary_plan
from c162_feasibility_model_v1 import EnvelopePoint, predict
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


class BoundaryPlanTest(unittest.TestCase):
    def test_prespecified_cases_match_model_and_plan(self):
        accepted = REQUEST_ORDER[:4]
        for case in CASES:
            decision_ms = 88 if case.case_id.startswith("E6a") else (
                89 if case.case_id.startswith("E6b") else case.target_decision_ms
            )
            point = EnvelopePoint(
                4, 4 - case.success_count, decision_ms, case.context_length
            )
            self.assertEqual(predict(point).state, case.expected_state, case.case_id)
            ai_bound = CLASS_BOUNDS_MS.get(case.context_length, 35)
            config = IntegratedScenarioConfig(ai_bound_ms=ai_bound)
            plan = build_boundary_plan(
                accepted[:case.success_count], decision_ms * 1_000_000,
                config, offer_ai=case.context_length is not None,
            )
            self.assertEqual(plan["lease_decision"]["accepted"], case.expected_lease)
            self.assertEqual(len(plan["physical_recovery_keys"]), 4 - case.success_count)
            self.assertEqual(plan["rejected_keys"], (REQUEST_ORDER[4],))

    def test_case_order_can_be_reversed(self):
        self.assertEqual(case_for_sequence(1), CASES[0])
        self.assertEqual(case_for_sequence(1, reverse=True), CASES[-1])
        self.assertEqual(case_for_sequence(7), CASES[0])

    def test_conservative_observed_time_mapping(self):
        self.assertEqual(math.ceil(87_000_001 / 1_000_000), 88)
        self.assertEqual(math.ceil(88_000_001 / 1_000_000), 89)


if __name__ == "__main__":
    unittest.main()
