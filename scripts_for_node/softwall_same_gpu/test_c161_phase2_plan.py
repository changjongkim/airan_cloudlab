#!/usr/bin/env python3.11

import unittest

from c161_phase2_plan import build_phase2_plan
from integrated_shared_recovery_plan_v1 import IntegratedScenarioConfig, REQUEST_ORDER


class C161Phase2PlanTests(unittest.TestCase):
    def test_quarantine_blocks_ai_without_changing_recovery_set(self):
        config = IntegratedScenarioConfig(period_ms=180, ai_bound_ms=35)
        successes = REQUEST_ORDER[:2]
        normal = build_phase2_plan(
            successes, config.recovery_release_ns, config, ai_enabled=True
        )
        blocked = build_phase2_plan(
            successes, config.recovery_release_ns, config, ai_enabled=False
        )
        self.assertTrue(normal["ai_expected"])
        self.assertFalse(blocked["ai_expected"])
        self.assertTrue(blocked["ai_quarantined"])
        self.assertIsNone(blocked["lease_interval"])
        self.assertEqual(
            normal["physical_recovery_keys"], blocked["physical_recovery_keys"]
        )
        self.assertEqual(
            blocked["lease_decision"]["reason"],
            "ai_quarantined_before_admission",
        )

    def test_quarantine_blocks_ai_when_every_debt_succeeds(self):
        config = IntegratedScenarioConfig(period_ms=180, ai_bound_ms=35)
        blocked = build_phase2_plan(
            REQUEST_ORDER[:4], config.recovery_release_ns, config,
            ai_enabled=False,
        )
        self.assertFalse(blocked["ai_expected"])
        self.assertEqual(blocked["physical_recovery_keys"], ())
        self.assertEqual(blocked["coordinator"].snapshot()["leases"], [])


if __name__ == "__main__":
    unittest.main()
