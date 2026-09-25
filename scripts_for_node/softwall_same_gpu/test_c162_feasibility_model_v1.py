#!/usr/bin/env python3.11

import unittest

from c162_feasibility_model_v1 import EnvelopePoint, predict


class C162FeasibilityModelTests(unittest.TestCase):
    def test_prespecified_boundary_points(self):
        cases = (
            (EnvelopePoint(4, 4, 45, None), "QSN"),
            (EnvelopePoint(5, 5, 45, None), "MI"),
            (EnvelopePoint(4, 2, 45, 128), "QSU"),
            (EnvelopePoint(4, 2, 45, 256), "QSN"),
            (EnvelopePoint(4, 1, 45, 512), "QSU"),
            (EnvelopePoint(4, 1, 88, 64), "QSU"),
            (EnvelopePoint(4, 1, 89, 64), "QSN"),
        )
        for point, expected in cases:
            with self.subTest(point=point):
                self.assertEqual(predict(point).state, expected)

    def test_provenance_prevents_false_promotion(self):
        self.assertEqual(predict(EnvelopePoint(
            4, 2, 45, 128, recovery_bound_ms=12
        )).state, "UQ")
        self.assertEqual(predict(EnvelopePoint(
            4, 2, 45, 128, lifecycle="cold"
        )).state, "UQ")
        self.assertEqual(predict(EnvelopePoint(
            4, 2, 45, 128, node_bound_status="failed"
        )).state, "UQ")

    def test_fault_quarantine_is_safe_no_slack(self):
        for fault in ("post_fence_reply_delay", "pre_fence_channel_loss"):
            self.assertEqual(predict(EnvelopePoint(
                4, 2, 94, 128, fault_class=fault
            )).state, "QSN")

    def test_memory_boundary(self):
        self.assertEqual(predict(EnvelopePoint(
            4, 2, 45, 128, resident_receivers_per_home=8
        )).state, "MI")
        self.assertEqual(predict(EnvelopePoint(
            4, 2, 45, 128, resident_receivers_per_home=6
        )).state, "UQ")


if __name__ == "__main__":
    unittest.main()
