#!/usr/bin/env python3.11
"""Prospective certificate tests for C159-Q2 context-class admission."""

import unittest

from actual_nrx_shared_recovery_plan_v1 import build_actual_outcome_plan
from c159_q2_classes import CLASS_BOUNDS_MS, CONTEXT_LENGTHS
from integrated_shared_recovery_plan_v1 import (
    MS,
    REQUEST_ORDER,
    IntegratedScenarioConfig,
)


class VariableContextPlanTests(unittest.TestCase):
    def test_class_admission_matches_serial_capacity(self):
        accepted = REQUEST_ORDER[:4]
        for context_length in CONTEXT_LENGTHS:
            bound_ms = CLASS_BOUNDS_MS[context_length]
            for success_count in range(5):
                config = IntegratedScenarioConfig(ai_bound_ms=bound_ms)
                plan = build_actual_outcome_plan(
                    accepted[:success_count],
                    config.recovery_release_ns,
                    config,
                )
                remaining = len(accepted) - success_count
                required_ms = (
                    remaining * config.conventional_bound_ms
                    + 5
                    + bound_ms
                )
                available_ms = (
                    config.recovery_deadline_ns - config.recovery_release_ns
                ) / MS
                self.assertEqual(
                    plan["ai_expected"],
                    required_ms <= available_ms,
                    (context_length, success_count, required_ms, available_ms),
                )

    def test_long_classes_need_at_most_one_remaining_recovery(self):
        for context_length in (256, 512):
            config = IntegratedScenarioConfig(
                ai_bound_ms=CLASS_BOUNDS_MS[context_length]
            )
            two_remaining = build_actual_outcome_plan(
                REQUEST_ORDER[:2], config.recovery_release_ns, config
            )
            one_remaining = build_actual_outcome_plan(
                REQUEST_ORDER[:3], config.recovery_release_ns, config
            )
            self.assertFalse(two_remaining["ai_expected"])
            self.assertTrue(one_remaining["ai_expected"])


if __name__ == "__main__":
    unittest.main()
