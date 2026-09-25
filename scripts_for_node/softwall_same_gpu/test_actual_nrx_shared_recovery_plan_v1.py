#!/usr/bin/env python3.11

import unittest

from actual_nrx_shared_recovery_plan_v1 import build_actual_outcome_plan
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER


MS = 1_000_000


class ActualNrxSharedRecoveryPlanTest(unittest.TestCase):
    def test_two_actual_successes_open_ai_and_leave_two_recoveries(self):
        successes = (REQUEST_ORDER[0], REQUEST_ORDER[3])
        plan = build_actual_outcome_plan(successes, 49 * MS)
        self.assertEqual(plan["local_certificates"], {
            "home0": True,
            "home1": True,
        })
        self.assertEqual(sum(
            row["accepted"] for row in plan["reservation_decisions"]
        ), 4)
        self.assertEqual(plan["rejected_keys"], (REQUEST_ORDER[4],))
        self.assertTrue(plan["ai_expected"])
        self.assertEqual(
            set(plan["physical_recovery_keys"]),
            {REQUEST_ORDER[1], REQUEST_ORDER[2]},
        )

    def test_no_success_keeps_four_debts_and_rejects_ai(self):
        plan = build_actual_outcome_plan((), 49 * MS)
        self.assertFalse(plan["ai_expected"])
        self.assertEqual(
            set(plan["physical_recovery_keys"]), set(REQUEST_ORDER[:4])
        )
        self.assertEqual(
            plan["lease_decision"]["reason"],
            "lease_breaks_global_certificate",
        )

    def test_all_accepted_successes_leave_no_recovery(self):
        plan = build_actual_outcome_plan(REQUEST_ORDER[:4], 49 * MS)
        self.assertTrue(plan["ai_expected"])
        self.assertEqual(plan["physical_recovery_keys"], ())

    def test_rejected_fifth_request_cannot_report_success(self):
        with self.assertRaisesRegex(ValueError, "not an accepted obligation"):
            build_actual_outcome_plan((REQUEST_ORDER[4],), 49 * MS)

    def test_late_two_success_decision_rejects_ai_atomically(self):
        plan = build_actual_outcome_plan(
            (REQUEST_ORDER[0], REQUEST_ORDER[3]), 64 * MS
        )
        self.assertFalse(plan["ai_expected"])
        self.assertTrue(
            plan["lease_decision"]["state_unchanged_on_reject"]
        )


if __name__ == "__main__":
    unittest.main()
