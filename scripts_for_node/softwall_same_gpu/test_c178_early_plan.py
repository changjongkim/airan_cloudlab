#!/usr/bin/env python3.11
"""CPU tests for C178 early AI slots."""

from __future__ import annotations

import unittest

from c176_burst_plan import MS
from c178_early_plan import mandatory_infeasible_plan, plan_early
from integrated_shared_recovery_plan_v1 import IntegratedScenarioConfig

CONFIG = IntegratedScenarioConfig(period_ms=180)


def request(name, context, deadline_ms=1000.0):
    return {"request_id": name, "context_length": context, "deadline_rel_ns": round(deadline_ms * MS)}


class EarlyPlanTest(unittest.TestCase):
    def test_short_slot_fits_before_four_recoveries(self):
        # 16 tokens: 5 ms launch budget + 35 ms bound ends at 40 ms, and the
        # four pending recoveries (100 ms) still end by the 153 ms deadline.
        plan = plan_early(0, [request("a", 16)], CONFIG, 4)
        self.assertEqual(len(plan["accepted_keys"]), 4)
        self.assertEqual([(l["start_ns"], l["finish_ns"]) for l in plan["leases"]], [(0, 40 * MS)])
        self.assertLessEqual(plan["all_fail_end_ns"], CONFIG.recovery_deadline_ns)

    def test_slack_decides_the_latest_early_start(self):
        # The window holds 108 ms and four recoveries 100 ms: a 128-token slot
        # (45 ms) may end at 53 ms, so it may start at 8 ms but not at 9 ms.
        self.assertEqual(len(plan_early(8 * MS, [request("a", 128)], CONFIG, 4)["leases"]), 1)
        self.assertEqual(plan_early(9 * MS, [request("a", 128)], CONFIG, 4)["leases"], [])

    def test_long_slots_are_rejected(self):
        for context in (256, 512):
            self.assertEqual(plan_early(0, [request("a", context)], CONFIG, 4)["leases"], [])

    def test_second_slot_would_break_the_schedule(self):
        plan = plan_early(0, [request("a", 16), request("b", 16)], CONFIG, 4)
        self.assertEqual([l["request_id"] for l in plan["leases"]], ["a"])

    def test_fifo_skips_a_rejected_request(self):
        plan = plan_early(0, [request("a", 512), request("b", 64)], CONFIG, 4)
        self.assertEqual([l["request_id"] for l in plan["leases"]], ["b"])

    def test_request_deadline_is_respected(self):
        self.assertEqual(plan_early(0, [request("a", 16, deadline_ms=39.0)], CONFIG, 4)["leases"], [])
        self.assertEqual(len(plan_early(0, [request("a", 16, deadline_ms=40.0)], CONFIG, 4)["leases"]), 1)

    def test_max_leases_caps_the_slots(self):
        self.assertEqual(plan_early(0, [request("a", 16)], CONFIG, 0)["leases"], [])


    def test_infeasible_period_runs_every_required_recovery_without_ai(self):
        successes = [("home0", "h0-r0")]
        plan = mandatory_infeasible_plan(successes, 120 * MS, CONFIG)
        self.assertTrue(plan["mandatory_infeasible"])
        self.assertEqual(plan["leases"], [])
        self.assertEqual(plan["unresolved_keys"], [("home0", "h0-r1"), ("home0", "h0-r2"), ("home1", "h1-r0")])
        self.assertEqual(plan["rejected_keys"], [("home1", "h1-r1")])
        self.assertEqual(plan["bound_end_ns"], 195 * MS)
        self.assertFalse(plan["contract_broken"])


    def test_prefilter_keeps_every_decision_of_the_verifier(self):
        import random
        rng = random.Random(20260929)
        contexts = (16, 32, 64, 128, 256, 512)
        for _ in range(300):
            queue = [request(f"r{i}", rng.choice(contexts), deadline_ms=rng.uniform(30, 400))
                     for i in range(rng.randint(0, 12))]
            now = rng.randint(0, 20) * MS
            fast = plan_early(now, queue, CONFIG, 4)
            slow = plan_early(now, queue, CONFIG, 4, prefilter=False)
            self.assertEqual(fast["leases"], slow["leases"])

    def test_prefilter_bounds_the_verifier_calls_of_a_burst(self):
        import time
        queue = [request(f"r{i}", 512) for i in range(400)] + [request("short", 16)]
        started = time.perf_counter()
        plan = plan_early(0, queue, CONFIG, 4)
        self.assertLess(time.perf_counter() - started, 0.005)
        self.assertEqual([l["request_id"] for l in plan["leases"]], ["short"])


if __name__ == "__main__":
    unittest.main()
