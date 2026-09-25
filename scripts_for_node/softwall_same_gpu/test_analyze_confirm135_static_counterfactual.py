#!/usr/bin/env python3

import unittest

from analyze_confirm135_static_counterfactual import counterfactual_terms


class Confirm135CounterfactualTest(unittest.TestCase):
    def protocol(self):
        return {
            "deadline_ms": 155,
            "guard_ms": 2,
            "conv_bound_ms": 25,
            "nrx_bound_ms": 45,
            "global_broker_transaction_budget_ms": 15,
            "admission_ai_guard_ms": 17,
        }

    def test_c135_complete_transaction_is_exchange_only(self):
        terms = counterfactual_terms(self.protocol(), 4, 2, 2, 40)
        self.assertEqual(terms["all_fail_base_slack_ms"], 53)
        self.assertEqual(terms["conditional_transaction_window_ms"], 58)
        self.assertEqual(terms["effective_transaction_bound_ms"], 57)
        self.assertEqual(terms["static_margin_ms"], -4)
        self.assertEqual(terms["conditional_margin_ms"], 1)
        self.assertEqual(terms["ai_completion_guard_ms"], 2)

    def test_inconsistent_guard_decomposition_is_rejected(self):
        protocol = self.protocol()
        protocol["admission_ai_guard_ms"] = 14
        with self.assertRaises(ValueError):
            counterfactual_terms(protocol, 4, 2, 2, 40)


if __name__ == "__main__":
    unittest.main()
