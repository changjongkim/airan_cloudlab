#!/usr/bin/env python3

import unittest
from types import SimpleNamespace

from four_cell_trace_baseline_controller import fallback_is_early


class TraceBaselineControllerTest(unittest.TestCase):
    def test_retimed_credit_controls_early_dispatch_branch(self):
        transaction = SimpleNamespace(
            request=SimpleNamespace(fallback_latest_start_ns=105),
            fallback_reservation=SimpleNamespace(start_ns=141),
        )
        # This is later than the immutable NRx cutoff but still earlier than
        # the credit after tail compaction.
        self.assertTrue(fallback_is_early(transaction, 116))
        self.assertFalse(fallback_is_early(transaction, 141))


if __name__ == "__main__":
    unittest.main()
