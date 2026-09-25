#!/usr/bin/env python3

import threading
import unittest

from shared_recovery_certificate_v1 import (
    RecoveryObligation,
    SharedAILease,
    SharedRecoveryCoordinator,
    exact_certificate,
)


MS = 1_000_000


def job(home, number, deadline=100):
    return RecoveryObligation(
        home, f"r{number}", 0, deadline * MS, 25 * MS
    )


class SharedRecoveryCertificateTest(unittest.TestCase):
    def test_local_safe_calendars_can_be_globally_infeasible(self):
        home0 = [job("h0", i) for i in range(3)]
        home1 = [job("h1", i) for i in range(2)]
        self.assertIsNotNone(exact_certificate(home0, 1))
        self.assertIsNotNone(exact_certificate(home1, 1))
        self.assertIsNone(exact_certificate(home0 + home1, 1))

    def test_rejected_update_is_atomic(self):
        coordinator = SharedRecoveryCoordinator(1)
        for home in ("h0", "h1"):
            for index in range(2):
                self.assertTrue(coordinator.reserve_mandatory(
                    job(home, index)
                ).accepted)
        before = coordinator.snapshot()
        rejected = coordinator.reserve_mandatory(job("h0", 2))
        self.assertFalse(rejected.accepted)
        self.assertEqual(rejected.reason, "global_all_fail_infeasible")
        self.assertEqual(coordinator.snapshot(), before)

    def test_success_credit_atomically_opens_conditional_ai_lease(self):
        coordinator = SharedRecoveryCoordinator(1)
        for home in ("h0", "h1"):
            for index in range(2):
                self.assertTrue(coordinator.reserve_mandatory(
                    job(home, index)
                ).accepted)
        full_generation = coordinator.generation
        blocked = coordinator.replan_and_lease(
            SharedAILease("ai0", "h0", 0, 25 * MS), full_generation
        )
        self.assertFalse(blocked.accepted)
        released = coordinator.resolve_success(
            ("h1", "r0"), coordinator.generation
        )
        self.assertTrue(released.accepted)
        admitted = coordinator.replan_and_lease(
            SharedAILease("ai0", "h0", 0, 25 * MS), released.generation
        )
        self.assertTrue(admitted.accepted)
        self.assertEqual(admitted.reason, "lease_committed")

    def test_stale_generation_cannot_publish_or_retire(self):
        coordinator = SharedRecoveryCoordinator(1)
        first = coordinator.reserve_mandatory(job("h0", 0))
        second = coordinator.reserve_mandatory(job("h1", 0), first.generation)
        self.assertTrue(second.accepted)
        before = coordinator.snapshot()
        stale = coordinator.replan_and_lease(
            SharedAILease("ai0", "h0", 0, 25 * MS), first.generation
        )
        self.assertFalse(stale.accepted)
        self.assertEqual(stale.reason, "stale_generation")
        self.assertEqual(coordinator.snapshot(), before)

    def test_physical_fence_is_required_to_retire_shared_lease(self):
        coordinator = SharedRecoveryCoordinator(1)
        reserved = coordinator.reserve_mandatory(job("h0", 0))
        leased = coordinator.replan_and_lease(
            SharedAILease("ai0", "h0", 0, 25 * MS), reserved.generation
        )
        self.assertTrue(leased.accepted)
        before = coordinator.snapshot()
        self.assertFalse(coordinator.retire_lease(
            "ai0", False, leased.generation
        ).accepted)
        self.assertEqual(coordinator.snapshot(), before)
        self.assertTrue(coordinator.retire_lease(
            "ai0", True, leased.generation
        ).accepted)

    def test_concurrent_same_generation_only_one_update_commits(self):
        coordinator = SharedRecoveryCoordinator(1)
        generation = coordinator.generation
        barrier = threading.Barrier(3)
        outcomes = []
        lock = threading.Lock()

        def reserve(home):
            barrier.wait()
            result = coordinator.reserve_mandatory(
                job(home, 0), generation
            )
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=reserve, args=(home,))
                   for home in ("h0", "h1")]
        for thread in threads:
            thread.start()
        barrier.wait()
        for thread in threads:
            thread.join()
        self.assertEqual(sum(result.accepted for result in outcomes), 1)
        self.assertEqual(
            sorted(result.reason for result in outcomes),
            ["reserved", "stale_generation"],
        )


if __name__ == "__main__":
    unittest.main()
