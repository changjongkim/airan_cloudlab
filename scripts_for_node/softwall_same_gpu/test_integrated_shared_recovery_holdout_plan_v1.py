#!/usr/bin/env python3

import unittest

from integrated_shared_recovery_holdout_plan_v1 import (
    ACCEPTED_KEYS,
    BRANCHES,
    build_branch,
)
from integrated_shared_recovery_plan_v1 import REJECTED_KEY


class IntegratedSharedRecoveryHoldoutPlanTest(unittest.TestCase):
    def test_all_four_branches_have_expected_decisions(self):
        expected = {
            "all_fail": (False, 4, ()),
            "conditional_open": (True, 2, (REJECTED_KEY,)),
            "all_success": (True, 0, ()),
            "overload": (False, 4, (REJECTED_KEY,)),
        }
        for branch in BRANCHES:
            with self.subTest(branch=branch):
                value = build_branch(branch)
                ai, recoveries, rejected = expected[branch]
                self.assertEqual(value["ai_expected"], ai)
                self.assertEqual(len(value["physical_recovery_keys"]), recoveries)
                self.assertEqual(value["rejected_keys"], rejected)
                self.assertEqual(value["local_certificates"], {
                    "home0": True,
                    "home1": True,
                })
                self.assertEqual(
                    sum(row["accepted"] for row in value["reservation_decisions"]),
                    4,
                )
                self.assertTrue(all(
                    row["state_unchanged_on_reject"]
                    for row in value["reservation_decisions"]
                ))

    def test_only_accepted_keys_can_reach_physical_plan(self):
        accepted = set(ACCEPTED_KEYS)
        for branch in BRANCHES:
            value = build_branch(branch)
            self.assertLessEqual(set(value["physical_recovery_keys"]), accepted)
            self.assertTrue(
                set(value["physical_recovery_keys"]).isdisjoint(
                    set(value["success_keys"])
                )
            )


if __name__ == "__main__":
    unittest.main()

