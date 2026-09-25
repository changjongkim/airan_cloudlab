#!/usr/bin/env python3

import unittest

from analyze_confirm138_service_bound_fault_telemetry import summarize_rpc_records


def arm(operation="prepare", elapsed=4.0, exceeded=False, include_target=True):
    records = [
        {"operation": "prepare", "elapsed_ms": 1.0,
         "faulted": include_target and operation == "prepare",
         "wall_timeout_exceeded": False},
        {"operation": "commit", "elapsed_ms": 1.0,
         "faulted": include_target and operation == "commit",
         "wall_timeout_exceeded": False},
        {"operation": "complete", "elapsed_ms": 1.0,
         "faulted": include_target and operation == "complete",
         "wall_timeout_exceeded": False},
    ]
    for record in records:
        if record["operation"] == operation:
            record["elapsed_ms"] = elapsed
            record["wall_timeout_exceeded"] = exceeded
    return {
        "crash_operation": operation,
        "homes": [
            {"home": 0, "rpc_telemetry": {
                "schema": "softwall-broker-rpc-telemetry-v2", "records": records,
            }},
            {"home": 1, "rpc_telemetry": {
                "schema": "softwall-broker-rpc-telemetry-v2", "records": [
                    {"operation": name, "elapsed_ms": 0.5, "faulted": False,
                     "wall_timeout_exceeded": False}
                    for name in ("prepare", "commit", "complete")
                ],
            }},
        ],
    }


class Confirm138AnalyzerTest(unittest.TestCase):
    def test_passes_all_control_points(self):
        result = summarize_rpc_records(
            [arm("prepare"), arm("commit"), arm("complete")], 5.0
        )
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["faulted_by_operation"], {
            "commit": 1, "complete": 1, "prepare": 1,
        })

    def test_fails_prospective_bound(self):
        result = summarize_rpc_records([arm("commit", 5.2, True)], 5.0)
        self.assertFalse(result["all_pass"])
        self.assertFalse(result["gates"]["no_wall_timeout_exceedance"])
        self.assertFalse(result["gates"]["maximum_within_declared_timeout"])

    def test_requires_faulting_call(self):
        result = summarize_rpc_records(
            [arm("prepare", include_target=False)], 5.0
        )
        self.assertFalse(result["all_pass"])
        self.assertFalse(result["gates"]["faulted_call_each_arm"])


if __name__ == "__main__":
    unittest.main()
