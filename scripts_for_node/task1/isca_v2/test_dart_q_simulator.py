#!/usr/bin/env python3
"""Deterministic tests for the DART-Q trace schema and control simulator."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from dart_q_simulator import DEFAULT_PROFILES, simulate, simulate_one
from dart_transaction_trace import (
    TransactionSample,
    TransactionTrace,
    read_trace,
    write_trace,
)


class DartQSimulatorTests(unittest.TestCase):
    def make_sample(self, **overrides):
        values = {
            "slot_id": 1,
            "epoch": 9,
            "release_ns": 0,
            "deadline_ns": 5_000_000,
            "fallback_latest_start_ns": 3_900_000,
            "conventional_service_ns": 1_000_000,
            "remote_ready_ns": 2_000_000,
        }
        values.update(overrides)
        return TransactionSample(**values)

    def test_trace_round_trip(self):
        trace = TransactionTrace.build(
            [self.make_sample()], {"provenance": "unit-test"}
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.json"
            write_trace(path, trace)
            self.assertEqual(read_trace(path), trace)

    def test_early_remote_completion_avoids_fallback(self):
        row = simulate_one(self.make_sample(), DEFAULT_PROFILES["dartq"])
        self.assertEqual(row["winner"], "nrx")
        self.assertFalse(row["fallback_started"])

    def test_invisible_remote_uses_reserved_fallback(self):
        row = simulate_one(
            self.make_sample(payload_visible=False),
            DEFAULT_PROFILES["dartq"],
        )
        self.assertEqual(row["winner"], "conventional")
        self.assertTrue(row["fallback_started"])
        self.assertTrue(row["timely"])

    def test_unreserved_fallback_queue_delay_can_miss(self):
        row = simulate_one(
            self.make_sample(
                remote_ready_ns=None,
                fallback_reserved=False,
                unreserved_fallback_delay_ns=500_000,
            ),
            DEFAULT_PROFILES["dartq"],
        )
        self.assertEqual(row["winner"], "deadline_miss")

    def test_single_winner_when_remote_and_fallback_race(self):
        row = simulate_one(
            self.make_sample(remote_ready_ns=4_950_000),
            DEFAULT_PROFILES["dartq"],
        )
        self.assertEqual(row["winner"], "conventional")
        self.assertTrue(row["fallback_started"])

    def test_dartq_control_tail_is_lower_than_placeholder_host(self):
        transactions = [
            self.make_sample(slot_id=index, epoch=index + 1, remote_ready_ns=3_880_000)
            for index in range(32)
        ]
        host = simulate(transactions, DEFAULT_PROFILES["host"])
        dartq = simulate(transactions, DEFAULT_PROFILES["dartq"])
        self.assertLess(
            dartq["latency_us"]["p99"],
            host["latency_us"]["p99"],
        )


if __name__ == "__main__":
    unittest.main(verbosity=2)

