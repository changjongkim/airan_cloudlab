#!/usr/bin/env python3

import unittest

from analyze_confirm137_service_bound_telemetry import aggregate_rpc_evidence


def arm(maximum=1.0, faulted=0, exceeded=0, include_complete=True):
    operations = ["prepare", "commit"] + (["complete"] if include_complete else [])
    records = [
        {"operation": operation, "elapsed_ms": maximum,
         "faulted": bool(faulted), "wall_timeout_exceeded": bool(exceeded)}
        for operation in operations
    ]
    return {
        "control_evidence": {
            "records": len(records),
            "by_operation": {operation: 1 for operation in operations},
            "faulted": faulted,
            "wall_timeout_exceeded": exceeded,
        },
        "homes": [{"rpc_telemetry": {"records": records}}],
    }


class Confirm137AnalyzerTest(unittest.TestCase):
    def test_pass(self):
        result = aggregate_rpc_evidence([arm(), arm(2.0)], 5.0)
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["records"], 6)
        self.assertEqual(result["maximum_elapsed_ms"], 2.0)

    def test_fails_bound(self):
        result = aggregate_rpc_evidence([arm(5.1, exceeded=1)], 5.0)
        self.assertFalse(result["all_pass"])
        self.assertFalse(result["gates"]["maximum_within_declared_timeout"])

    def test_fails_missing_operation(self):
        result = aggregate_rpc_evidence([arm(include_complete=False)], 5.0)
        self.assertFalse(result["all_pass"])
        self.assertFalse(result["gates"]["every_arm_prepare_commit_complete_coverage"])


if __name__ == "__main__":
    unittest.main()
