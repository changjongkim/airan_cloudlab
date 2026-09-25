#!/usr/bin/env python3

import unittest

from build_softwall_envelope_v14 import build, V13_ID, V14_ID
from softwall_envelope_checker_v9 import evaluate


class EnvelopeV14BuilderTest(unittest.TestCase):
    def test_pipelined_mode_charges_only_launch_commit(self):
        old = {
            "id": V13_ID,
            "architecture": "two homes",
            "gpu_count": 2,
            "cells": 8,
            "home_cell_counts": [4, 4],
            "deadline_ms": 155.0,
            "guard_ms": 2.0,
            "recovery_bound_ms": 25.0,
            "home_memory_feasible": True,
            "timing_bounds_qualified": True,
            "ai_bounds_ms": [35.0],
            "evidence": [],
            "home_endpoint_bounds_ms": [[45.0, 45.0], [45.0, 45.0]],
            "home_endpoint_ring_depths": [[1, 1], [1, 1]],
            "rejected_recovery_before_exchange": False,
            "control_fault_model": "sync",
            "control_fault_qualified": True,
            "control_rpc_budget_required": True,
            "control_rpc_bound_ms": 7,
            "control_rpc_count": 3,
            "control_transaction_budget_ms": 21,
            "control_budget_accounted_in_admission": True,
            "ai_completion_guard_ms": 2.0,
        }
        value = build({"schema": "v13", "status": "old", "scope": "scope.",
                       "classification": {}, "scenarios": [old]})
        rows = {row["id"]: row for row in value["scenarios"]}
        new = rows[V14_ID]
        self.assertEqual(new["control_rpc_count"], 1)
        self.assertEqual(new["control_transaction_budget_ms"], 7)
        self.assertEqual(new["control_critical_operations"], ["commit"])
        result = evaluate(new)
        ai45 = next(row for row in result["ai_classes"]
                    if row["bound_ms"] == 45.0)
        self.assertEqual(ai45["effective_transaction_bound_ms"], 54.0)
        self.assertTrue(ai45["exchange_only_candidate"])


if __name__ == "__main__":
    unittest.main()
