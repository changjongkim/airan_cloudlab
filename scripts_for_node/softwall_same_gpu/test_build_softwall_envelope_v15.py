#!/usr/bin/env python3

import unittest

from build_softwall_envelope_v15 import build, V14_ID, V15_ID
from softwall_envelope_checker_v10 import evaluate


def old_mode():
    return {
        "id": V14_ID,
        "gpu_count": 2,
        "cells": 8,
        "home_cell_counts": [4, 4],
        "deadline_ms": 155.0,
        "guard_ms": 2.0,
        "recovery_bound_ms": 25.0,
        "home_memory_feasible": True,
        "timing_bounds_qualified": True,
        "ai_bounds_ms": [45.0],
        "evidence": [],
        "home_endpoint_bounds_ms": [[45.0, 45.0], [45.0, 45.0]],
        "home_endpoint_ring_depths": [[1, 1], [1, 1]],
        "rejected_recovery_before_exchange": False,
        "control_fault_qualified": True,
        "control_rpc_budget_required": True,
        "control_rpc_bound_ms": 7,
        "control_rpc_count": 1,
        "control_transaction_budget_ms": 7,
        "control_budget_accounted_in_admission": True,
        "ai_completion_guard_ms": 2.0,
    }


def node(node):
    return {
        "all_pass": True,
        "nodes": [node],
        "totals": {
            "ai45_exchanges": 1,
            "fault_arms": 3,
            "radio_records": 6080,
            "radio_after_fault": 3000,
            "suppressed_prepare_due_owned_token": 10,
            "maximum_unlaunched_tokens": 1,
        },
    }


class EnvelopeV15BuilderTest(unittest.TestCase):
    def test_v14_demoted_and_two_node_v15_qualified(self):
        grid = {"schema": "v14", "status": "old", "scope": "scope.",
                "classification": {}, "scenarios": [old_mode()]}
        value = build(grid, [node("n1"), node("n2")])
        rows = {row["id"]: evaluate(row) for row in value["scenarios"]}
        self.assertEqual(rows[V14_ID]["status"], "UQ")
        self.assertEqual(rows[V15_ID]["status"], "QSU")
        self.assertTrue(rows[V15_ID]["single_token_ownership_qualified"])

    def test_v15_stays_unqualified_without_two_nodes(self):
        grid = {"schema": "v14", "status": "old", "scope": "scope.",
                "classification": {}, "scenarios": [old_mode()]}
        value = build(grid, [node("n1")])
        fixed = next(row for row in value["scenarios"] if row["id"] == V15_ID)
        self.assertEqual(evaluate(fixed)["status"], "UQ")


if __name__ == "__main__":
    unittest.main()
