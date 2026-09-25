#!/usr/bin/env python3

import unittest

from integrated_shared_recovery_plan_v1 import (
    LEASE_ID,
    PHYSICAL_RECOVERY_KEYS,
    REJECTED_KEY,
    IntegratedScenarioConfig,
    local_certificates,
    placement_keys,
    reserve_global,
    resolve_successes_and_lease,
)


class IntegratedSharedRecoveryPlanTest(unittest.TestCase):
    def setUp(self):
        self.config = IntegratedScenarioConfig()

    def test_local_feasible_global_fifth_rejected_atomically(self):
        self.assertEqual(local_certificates(self.config), {
            "home0": True,
            "home1": True,
        })
        coordinator, decisions = reserve_global(self.config)
        self.assertEqual(sum(row["accepted"] for row in decisions), 4)
        rejected = [row for row in decisions if not row["accepted"]]
        self.assertEqual(
            [(row["home_id"], row["request_id"]) for row in rejected],
            [REJECTED_KEY],
        )
        self.assertEqual(rejected[0]["reason"], "global_all_fail_infeasible")
        self.assertTrue(rejected[0]["state_unchanged_on_reject"])
        self.assertEqual(coordinator.generation, 4)

    def test_ai_opens_only_after_two_successes(self):
        coordinator, _ = reserve_global(self.config)
        transition = resolve_successes_and_lease(coordinator, self.config)
        first = transition["lease_after_one_success"]
        second = transition["lease_after_two_successes"]
        self.assertFalse(first["accepted"])
        self.assertEqual(first["reason"], "lease_breaks_global_certificate")
        self.assertTrue(first["state_unchanged"])
        self.assertTrue(second["accepted"])
        self.assertEqual(second["reason"], "lease_committed")
        snapshot = coordinator.snapshot()
        self.assertEqual([row["lease_id"] for row in snapshot["leases"]], [LEASE_ID])
        self.assertEqual(set(placement_keys(snapshot)), set(PHYSICAL_RECOVERY_KEYS))

    def test_physical_fence_and_current_time_replan(self):
        coordinator, _ = reserve_global(self.config)
        resolve_successes_and_lease(coordinator, self.config)
        before = coordinator.snapshot()
        rejected = coordinator.retire_lease(
            LEASE_ID,
            gpu_fence_confirmed=False,
            expected_generation=coordinator.generation,
            now_ns=70_000_000,
        )
        self.assertFalse(rejected.accepted)
        self.assertEqual(before, coordinator.snapshot())
        accepted = coordinator.retire_lease(
            LEASE_ID,
            gpu_fence_confirmed=True,
            expected_generation=coordinator.generation,
            now_ns=70_000_000,
        )
        self.assertTrue(accepted.accepted)
        self.assertEqual(placement_keys(coordinator.snapshot()), PHYSICAL_RECOVERY_KEYS)
        self.assertGreaterEqual(
            min(row["start_ns"] for row in coordinator.snapshot()["placements"]),
            70_000_000,
        )


if __name__ == "__main__":
    unittest.main()

