#!/usr/bin/env python3.11

import unittest

from verify_shared_recovery_certificate_v1 import build_result


class SharedRecoveryVerifierTest(unittest.TestCase):
    def test_finite_state_campaign_passes(self):
        result = build_result()
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["status"], "MODEL_PASS_PHYSICAL_UQ")
        self.assertEqual(
            result["equal_deadline_grid"]["analytic_mismatch_count"], 0
        )
        self.assertGreater(
            result["equal_deadline_grid"][
                "local_safe_global_unsafe_count"
            ],
            0,
        )
        self.assertEqual(
            result["variable_window_grid"]["reference_mismatch_count"], 0
        )


if __name__ == "__main__":
    unittest.main()
