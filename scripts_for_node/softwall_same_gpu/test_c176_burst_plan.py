#!/usr/bin/env python3.11
"""CPU tests for the C176 admission policies."""

from __future__ import annotations

import unittest

from c176_burst_plan import MS, lease_ns, plan_period
from integrated_shared_recovery_plan_v1 import IntegratedScenarioConfig

ALL = (("home0", "h0-r0"), ("home0", "h0-r1"), ("home0", "h0-r2"), ("home1", "h1-r0"))
CONFIG = IntegratedScenarioConfig(period_ms=180)
NOW = 45 * MS


def request(name, context, deadline_ms=1000.0):
    return {"request_id": name, "context_length": context, "deadline_rel_ns": round(deadline_ms * MS)}


class PlanTest(unittest.TestCase):
    def test_all_fail_admission_rejects_fifth_request(self):
        plan = plan_period("backstop", ALL, NOW, [], CONFIG, 4)
        self.assertEqual(len(plan["accepted_keys"]), 4)
        self.assertEqual(plan["rejected_keys"], [("home1", "h1-r1")])
        self.assertEqual(plan["unresolved_keys"], [])

    def test_backstop_rejects_lease_that_breaks_two_recoveries(self):
        # Two required recoveries: a 256-token lease (70 ms) + 50 ms > 108 ms window.
        plan = plan_period("backstop", ALL[:2], NOW, [request("a", 256)], CONFIG, 4)
        self.assertEqual(len(plan["unresolved_keys"]), 2)
        self.assertEqual(plan["leases"], [])
        plan = plan_period("backstop", ALL[:2], NOW, [request("a", 128)], CONFIG, 4)
        self.assertEqual([lease["request_id"] for lease in plan["leases"]], ["a"])
        self.assertLessEqual(plan["bound_end_ns"], CONFIG.recovery_deadline_ns)

    def test_idle_time_breaks_the_contract_backstop_does_not(self):
        queue = [request("a", 256)]
        idle = plan_period("idle_time", ALL[:2], NOW, queue, CONFIG, 4)
        self.assertEqual(len(idle["leases"]), 1)
        self.assertTrue(idle["contract_broken"])
        self.assertGreater(idle["bound_end_ns"], CONFIG.recovery_deadline_ns)
        backstop = plan_period("backstop", ALL[:2], NOW, queue, CONFIG, 4)
        self.assertFalse(backstop["contract_broken"])

    def test_recovery_first_places_leases_after_recoveries(self):
        plan = plan_period("recovery_first", ALL[:3], NOW, [request("a", 64)], CONFIG, 4)
        self.assertEqual(plan["leases"][0]["start_ns"], NOW + 25 * MS)
        self.assertLessEqual(plan["bound_end_ns"], CONFIG.recovery_deadline_ns)

    def test_deadline_separates_backstop_and_recovery_first(self):
        # One required recovery; the lease must end by 45 + 5 + 35 + 10 = 95 ms.
        queue = [request("a", 64, deadline_ms=95.0)]
        self.assertEqual(len(plan_period("backstop", ALL[:3], NOW, queue, CONFIG, 4)["leases"]), 1)
        self.assertEqual(len(plan_period("recovery_first", ALL[:3], NOW, queue, CONFIG, 4)["leases"]), 0)

    def test_multiple_leases_fill_the_window_in_fifo_order(self):
        queue = [request(name, 64) for name in "abcd"]
        plan = plan_period("backstop", ALL, NOW, queue, CONFIG, 4)
        # No required recovery: 108 ms window holds two 40 ms leases (80 ms), not three (120 ms).
        self.assertEqual([lease["request_id"] for lease in plan["leases"]], ["a", "b"])
        self.assertEqual(plan["leases"][1]["start_ns"], plan["leases"][0]["finish_ns"])
        self.assertEqual(lease_ns(64), 40 * MS)

    def test_planning_does_not_mutate_queue(self):
        queue = [request("a", 64)]
        plan_period("backstop", ALL, NOW, queue, CONFIG, 4)
        self.assertEqual(queue, [request("a", 64)])


if __name__ == "__main__":
    unittest.main()
