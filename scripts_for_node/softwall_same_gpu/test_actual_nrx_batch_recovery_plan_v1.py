#!/usr/bin/env python3.11

import unittest

from actual_nrx_batch_recovery_plan_v1 import build_actual_outcome_batch_plan
from actual_nrx_shared_recovery_plan_v1 import build_actual_outcome_plan
from c159_q2_classes import CLASS_BOUNDS_MS, CONTEXT_LENGTHS
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


class BatchOutcomePlanTests(unittest.TestCase):
    def test_matches_sequential_plan_at_normal_cutoff(self):
        accepted = REQUEST_ORDER[:4]
        for length in CONTEXT_LENGTHS:
            config = IntegratedScenarioConfig(ai_bound_ms=CLASS_BOUNDS_MS[length])
            for success_count in range(5):
                successes = accepted[:success_count]
                sequential = build_actual_outcome_plan(
                    successes, config.recovery_release_ns, config
                )
                batch = build_actual_outcome_batch_plan(
                    successes, config.recovery_release_ns, config
                )
                self.assertEqual(batch["ai_expected"], sequential["ai_expected"])
                self.assertEqual(
                    set(batch["physical_recovery_keys"]),
                    set(sequential["physical_recovery_keys"]),
                )
                self.assertEqual(batch["rejected_keys"], sequential["rejected_keys"])

    def test_late_all_success_has_no_artificial_intermediate_debt(self):
        config = IntegratedScenarioConfig(ai_bound_ms=35)
        late_now_ns = 100_000_000
        with self.assertRaises(RuntimeError):
            build_actual_outcome_plan(REQUEST_ORDER[:4], late_now_ns, config)
        batch = build_actual_outcome_batch_plan(
            REQUEST_ORDER[:4], late_now_ns, config
        )
        self.assertTrue(batch["ai_expected"])
        self.assertEqual(batch["physical_recovery_keys"], ())
        self.assertEqual(
            batch["outcome_transition"]["kind"],
            "atomic_observed_success_batch",
        )

    def test_late_three_successes_rebuild_one_real_debt(self):
        config = IntegratedScenarioConfig(ai_bound_ms=35)
        late_now_ns = 80_000_000
        with self.assertRaises(RuntimeError):
            build_actual_outcome_plan(REQUEST_ORDER[:3], late_now_ns, config)
        batch = build_actual_outcome_batch_plan(
            REQUEST_ORDER[:3], late_now_ns, config
        )
        self.assertTrue(batch["ai_expected"])
        self.assertEqual(
            batch["physical_recovery_keys"], (REQUEST_ORDER[3],)
        )


if __name__ == "__main__":
    unittest.main()
