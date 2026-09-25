#!/usr/bin/env python3

import unittest

from global_trace_budgeted_fault_contained_controller import composed_ai_guard_ns


class GlobalControlBudgetTest(unittest.TestCase):
    def test_two_rpc_budgets_are_charged_to_global_ai_admission(self):
        total, broker = composed_ai_guard_ns(2_000_000, 5, True)
        self.assertEqual(broker, 10_000_000)
        self.assertEqual(total, 12_000_000)

    def test_local_queue_does_not_pay_global_broker_budget(self):
        total, broker = composed_ai_guard_ns(2_000_000, 5, False)
        self.assertEqual(broker, 0)
        self.assertEqual(total, 2_000_000)


if __name__ == "__main__":
    unittest.main()
