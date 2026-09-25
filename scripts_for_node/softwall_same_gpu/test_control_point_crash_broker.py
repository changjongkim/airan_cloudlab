#!/usr/bin/env python3

import threading
import unittest

from control_point_crash_global_trace_lease_broker import BrokerServer


class ControlPointCrashBrokerTest(unittest.TestCase):
    def server(self, operation, number):
        server = BrokerServer.__new__(BrokerServer)
        server.crash_after_operation = operation
        server.crash_after_home = 0
        server.crash_after_number = number
        server.operation_counts = {}
        server.faults = []
        server.fault_lock = threading.Lock()
        return server

    def test_empty_prepare_is_not_an_applied_fault_point(self):
        server = self.server("prepare", 1)
        self.assertFalse(server.should_crash_after_operation(
            "prepare", {"home": 0}, None
        ))
        self.assertEqual(server.operation_counts, {})

    def test_each_operation_counts_only_its_home_and_class(self):
        server = self.server("complete", 2)
        result = {"request_id": "r0", "broker_token": "t0"}
        self.assertFalse(server.should_crash_after_operation(
            "complete", {"home": 1, "token": "t1"}, result
        ))
        self.assertFalse(server.should_crash_after_operation(
            "commit", {"home": 0, "token": "t0"}, result
        ))
        self.assertFalse(server.should_crash_after_operation(
            "complete", {"home": 0, "token": "t0"}, result
        ))
        self.assertTrue(server.should_crash_after_operation(
            "complete", {"home": 0, "token": "t0"}, result
        ))
        self.assertEqual(server.faults[0]["operation"], "complete")
        self.assertEqual(server.faults[0]["operation_number"], 2)


if __name__ == "__main__":
    unittest.main()
