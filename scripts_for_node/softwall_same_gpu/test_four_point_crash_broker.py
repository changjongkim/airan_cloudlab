#!/usr/bin/env python3

import json
import socket
import threading
import unittest

from control_point_crash_global_trace_lease_broker import GlobalTraceLeaseState
from four_point_crash_global_trace_lease_broker import (
    FAULT_OPERATIONS,
    FourPointBrokerServer,
)


class FourPointCrashBrokerTest(unittest.TestCase):
    def test_abort_is_a_configured_fault_operation(self):
        self.assertEqual(
            set(FAULT_OPERATIONS), {"prepare", "abort", "commit", "complete"}
        )

    def test_abort_transition_is_counted_after_apply(self):
        server = FourPointBrokerServer.__new__(FourPointBrokerServer)
        server.crash_after_operation = "abort"
        server.crash_after_home = 0
        server.crash_after_number = 1
        server.operation_counts = {}
        server.faults = []
        server.fault_lock = threading.Lock()
        self.assertTrue(server.should_crash_after_operation(
            "abort", {"home": 0, "token": "t0"},
            {"request_id": "r0", "state": "ready"},
        ))
        self.assertEqual(server.faults[0]["request_id"], "r0")


if __name__ == "__main__":
    unittest.main()
