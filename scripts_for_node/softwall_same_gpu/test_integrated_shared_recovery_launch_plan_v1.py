#!/usr/bin/env python3.11

import unittest

from integrated_shared_recovery_launch_plan_v1 import build_launch_branch


class IntegratedSharedRecoveryLaunchPlanTest(unittest.TestCase):
    def test_conditional_lease_retimes_to_actual_launch(self):
        value = build_launch_branch("conditional_open", 49_000_000)
        self.assertTrue(value["lease_decision"]["accepted"])
        self.assertEqual(value["lease_interval"]["start_ns"], 49_000_000)
        self.assertEqual(value["lease_interval"]["finish_ns"], 89_000_000)
        starts = [
            row["start_ns"] for row in value["coordinator"].snapshot()["placements"]
        ]
        self.assertGreaterEqual(min(starts), 89_000_000)

    def test_too_late_conditional_lease_is_rejected_atomically(self):
        value = build_launch_branch("conditional_open", 64_000_000)
        self.assertFalse(value["lease_decision"]["accepted"])
        self.assertEqual(
            value["lease_decision"]["reason"], "lease_breaks_global_certificate"
        )
        self.assertTrue(value["lease_decision"]["state_unchanged_on_reject"])
        self.assertEqual(value["coordinator"].snapshot()["leases"], [])

    def test_all_fail_never_opens_ai_at_outcome_time(self):
        value = build_launch_branch("all_fail", 45_000_000)
        self.assertFalse(value["lease_decision"]["accepted"])
        self.assertEqual(len(value["physical_recovery_keys"]), 4)


if __name__ == "__main__":
    unittest.main()
