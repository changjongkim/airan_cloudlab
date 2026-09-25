#!/usr/bin/env python3

from __future__ import annotations

import unittest

from global_trace_certificate_executor_controller import certificate_order
from isca_v2.dart_runtime import (
    DartRequest,
    DartRuntime,
    EndpointState,
    FallbackCalendar,
    ProfileTable,
    ServiceProfile,
)


class CertificateExecutorOrderTest(unittest.TestCase):
    def setUp(self):
        self.calendar = FallbackCalendar(
            capacity=1, service_ns=25_000_000, commit_guard_ns=2_000_000
        )
        endpoint = EndpointState("nrx0", ring_depth=1)
        profiles = ProfileTable({
            ("nrx0", 0, 0, "isolated"): ServiceProfile(0, 1, 0),
        })
        self.runtime = DartRuntime(
            [endpoint], profiles, 2_000_000,
            fallback_calendar=self.calendar,
        )
        release = 1_000_000_000
        deadline = release + 155_000_000
        self.transactions = []
        for cell, offset_ms in enumerate((53, 78, 103, 128)):
            request = DartRequest(
                slot_id=cell,
                epoch=1,
                graph_id=0,
                tensor_class=0,
                release_ns=release,
                deadline_ns=deadline,
                fallback_latest_start_ns=release + offset_ms * 1_000_000,
            )
            tx = self.runtime.reserve_mandatory(request, release)
            self.assertIsNotNone(tx)
            self.transactions.append(tx)
        self.radio = [{"commit_kind": None} for _ in self.transactions]

    def test_later_ready_class_cannot_jump_over_earlier_credit(self):
        # This is the exact C125 geometry.  Starting cell 2 at +33 ms would
        # occupy [33,58] and collide with cell 0's live [53,78] credit.
        self.assertFalse(
            self.runtime.start_fallback_early(
                self.transactions[2], 1_033_000_000
            )
        )

    def test_merged_ready_work_follows_live_certificate(self):
        ordered = certificate_order(self.transactions, self.radio)
        self.assertEqual([cell for cell, _ in ordered], [0, 1, 2, 3])
        for _, tx in ordered:
            start = tx.fallback_reservation.start_ns
            self.assertTrue(self.runtime.start_fallback(tx, start))
            self.assertTrue(self.runtime.complete_conventional(tx, start + 1))
        self.assertEqual(self.calendar.snapshot()["outstanding"], 0)


if __name__ == "__main__":
    unittest.main()
