#!/usr/bin/env python3

import unittest

from analyze_softwall_service_bound_qualification_v2 import sum_totals


class ServiceBoundQualificationV2Test(unittest.TestCase):
    def test_sum_totals_keeps_units_separate(self):
        original = {
            "nodes": 2, "arms": 8, "radio_records": 10240,
            "home_releases": 2560, "atomic_exchanges": 72,
            "candidate_branches": 1196, "ai40_candidate_exchanges": 38,
            "declared_safety_violations": 0,
        }
        telemetry = {
            "nodes": ["n3"], "arms": 6, "radio_records": 7680,
            "home_releases": 1920, "atomic_exchanges": 52,
            "candidate_branches": 904, "ai40_candidate_exchanges": 28,
        }
        result = sum_totals(original, telemetry)
        self.assertEqual(result["nodes"], 3)
        self.assertEqual(result["arms"], 14)
        self.assertEqual(result["radio_records"], 17920)
        self.assertEqual(result["ai40_candidate_exchanges"], 66)


if __name__ == "__main__":
    unittest.main()
