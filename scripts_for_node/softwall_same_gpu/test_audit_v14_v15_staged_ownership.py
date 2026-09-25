#!/usr/bin/env python3

import unittest

from audit_v14_v15_staged_ownership import verify


class StagedOwnershipAuditTest(unittest.TestCase):
    def test_v14_regression_and_v15_fix(self):
        result = verify()
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["v14"]["untracked_held_tokens"], 1)
        self.assertEqual(result["v15"]["untracked_held_tokens"], 0)


if __name__ == "__main__":
    unittest.main()
