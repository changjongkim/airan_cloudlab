#!/usr/bin/env python3

import unittest

from build_softwall_envelope_v13 import build, NEW_ID, OLD_ID


class EnvelopeV13BuilderTest(unittest.TestCase):
    def test_invalidates_old_bound_and_adds_corrected_mode(self):
        old = {
            "id": OLD_ID, "control_fault_qualified": True,
            "control_rpc_bound_ms": 5, "control_rpc_count": 3,
            "control_transaction_budget_ms": 15,
            "control_budget_accounted_in_admission": True,
            "qualification_nodes": ["old"], "evidence": [],
        }
        value = build({"schema": "v12", "status": "old", "scope": "scope.",
                       "scenarios": [old]})
        rows = {row["id"]: row for row in value["scenarios"]}
        self.assertFalse(rows[OLD_ID]["control_fault_qualified"])
        self.assertEqual(rows[NEW_ID]["control_rpc_bound_ms"], 7)
        self.assertEqual(rows[NEW_ID]["control_transaction_budget_ms"], 21)
        self.assertTrue(rows[NEW_ID]["control_fault_qualified"])


if __name__ == "__main__":
    unittest.main()
