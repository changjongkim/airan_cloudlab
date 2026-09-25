#!/usr/bin/env python3

import unittest

from analyze_confirm136_v12_requalification import zero_failure_upper


class CombinedQualificationSensitivityTest(unittest.TestCase):
    def test_combined_units_show_dependence_sensitivity(self):
        tb = zero_failure_upper(10240)
        release = zero_failure_upper(2560)
        arm = zero_failure_upper(8)
        node = zero_failure_upper(2)
        self.assertLess(tb, release)
        self.assertLess(release, arm)
        self.assertLess(arm, node)


if __name__ == "__main__":
    unittest.main()
