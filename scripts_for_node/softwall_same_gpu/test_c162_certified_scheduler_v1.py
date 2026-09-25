#!/usr/bin/env python3.11

import dataclasses
import unittest

from c162_certified_scheduler_v1 import certified_schedule, verify_certificate
from shared_recovery_certificate_v1 import RecoveryObligation, SharedAILease


class CertifiedSchedulerTest(unittest.TestCase):
    def setUp(self):
        self.jobs = tuple(
            RecoveryObligation("h0", f"r{i}", 10, 100, 15)
            for i in range(4)
        )

    def test_generated_certificate_verifies(self):
        rows = certified_schedule(self.jobs, 1, now_ns=10)
        self.assertIsNotNone(rows)
        self.assertEqual(verify_certificate(self.jobs, rows, 1, now_ns=10),
                         (True, "verified"))

    def test_ai_blackout_is_respected(self):
        lease = SharedAILease("a", "global", 10, 40)
        rows = certified_schedule(self.jobs, 1, (lease,), now_ns=10)
        self.assertTrue(all(row.start_ns >= 40 for row in rows))
        self.assertTrue(verify_certificate(self.jobs, rows, 1, (lease,), 10)[0])

    def test_infeasible_returns_none(self):
        jobs = tuple(dataclasses.replace(job, deadline_ns=50) for job in self.jobs)
        self.assertIsNone(certified_schedule(jobs, 1, now_ns=10))

    def test_verifier_rejects_mutated_deadline(self):
        rows = list(certified_schedule(self.jobs, 1, now_ns=10))
        rows[-1] = dataclasses.replace(rows[-1], finish_ns=101)
        self.assertFalse(verify_certificate(self.jobs, rows, 1, now_ns=10)[0])

    def test_verifier_rejects_missing_key(self):
        rows = certified_schedule(self.jobs, 1, now_ns=10)
        self.assertEqual(verify_certificate(self.jobs, rows[:-1], 1, now_ns=10)[1],
                         "placement_key_mismatch")


if __name__ == "__main__":
    unittest.main()
