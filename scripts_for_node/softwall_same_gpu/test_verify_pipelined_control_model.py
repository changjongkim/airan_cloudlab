#!/usr/bin/env python3

import unittest

from verify_pipelined_control_model import verify


class PipelinedControlModelTest(unittest.TestCase):
    def test_all_rpc_failure_branches_preserve_ownership_invariants(self):
        result = verify()
        self.assertTrue(result["all_pass"])
        self.assertFalse(result["violations"])
        transitions = {
            row["transition"] for row in result["transition_values"]
        }
        for operation in ("prepare", "abort", "commit", "complete"):
            self.assertIn(f"{operation}_fail_before_apply", transitions)
            self.assertIn(f"{operation}_fail_after_apply", transitions)
        critical = {
            row["transition"] for row in result["transition_values"]
            if row["ran_critical_path"]
        }
        self.assertEqual(critical, {
            "commit_ack", "commit_fail_before_apply",
            "commit_fail_after_apply",
        })


if __name__ == "__main__":
    unittest.main()
