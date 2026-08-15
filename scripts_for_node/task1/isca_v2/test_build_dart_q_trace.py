#!/usr/bin/env python3
"""Tests for actual-hardware-artifact to DART-Q trace conversion."""

from __future__ import annotations

import unittest

from build_dart_q_trace import convert, select_run


class BuildDartQTraceTests(unittest.TestCase):
    def artifact(self):
        return {
            "schema": "dart-nrx-policy-hardware-v0",
            "scope": "actual NRx compute/queue; no transport",
            "runs": [{
                "policy": "predicted_finish",
                "rate_slots_s": 1000.0,
                "requests": 2,
                "raw": {
                    "arrival_ms": [0.0, 1.0],
                    "completion_ms": [1.5, 2.7],
                    "assignment_device": [0, 1],
                },
            }],
        }

    def test_select_run_is_exact(self):
        run = select_run(self.artifact(), "predicted_finish", 1000.0)
        self.assertEqual(run["requests"], 2)

    def test_conversion_preserves_measured_readiness(self):
        trace = convert(
            self.artifact(),
            "predicted_finish",
            1000.0,
            deadline_ns=5_000_000,
            conventional_ns=1_000_000,
            commit_guard_ns=50_000,
        )
        self.assertEqual(len(trace.transactions), 2)
        self.assertEqual(trace.transactions[0].remote_ready_ns, 1_500_000)
        self.assertEqual(trace.transactions[1].release_ns, 1_000_000)
        self.assertEqual(trace.transactions[1].remote_ready_ns, 2_700_000)
        self.assertEqual(
            trace.transactions[0].fallback_latest_start_ns,
            3_950_000,
        )
        self.assertFalse(trace.metadata["transport_included"])

    def test_invalid_timing_is_rejected(self):
        with self.assertRaises(ValueError):
            convert(
                self.artifact(),
                "predicted_finish",
                1000.0,
                deadline_ns=1_000_000,
                conventional_ns=1_000_000,
                commit_guard_ns=50_000,
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)

