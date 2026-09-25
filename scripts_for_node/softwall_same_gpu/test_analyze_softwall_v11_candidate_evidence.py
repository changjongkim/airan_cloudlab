#!/usr/bin/env python3

import unittest

from analyze_softwall_v11_candidate_evidence import audit_controller


def controller(bound_ms):
    record = {
        "index": 0, "cell": 0, "admitted": True, "nrx_commit": True,
    }
    rows = [record, dict(record, cell=1)]
    rows += [
        {"index": 0, "cell": 2, "admitted": False, "nrx_commit": False},
        {"index": 0, "cell": 3, "admitted": False, "nrx_commit": False},
    ]
    return {
        "records": rows,
        "recovery_decisions": [{
            "release_index": 0, "selected_request_id": "r0",
            "ai_lease": True, "pending_cells": [2, 3],
        }],
        "background_records": [{
            "request_id": "r0", "bound_ms": bound_ms,
            "horizon_ns": 103_000_000, "returned_ns": 80_000_000,
            "execution_ms": 20,
        }],
        "admission_ai_guard_ms": 17,
        "deadline_misses": 0, "nrx_bound_violations": 0,
        "conv_bound_violations": 0, "conv_path_bound_violations": 0,
        "background_budget_violations": 0,
        "background_horizon_violations": 0,
    }


class V11CandidateEvidenceTest(unittest.TestCase):
    def test_exact_ai40_occurrence_is_counted(self):
        result = audit_controller(controller(40), "synthetic")
        self.assertEqual(result["ai40_candidate_occurrences"], 1)
        self.assertEqual(
            result["candidate_examples"][0]["reserved_horizon_margin_ms"], 6
        )

    def test_larger_ai_is_not_mislabeled(self):
        result = audit_controller(controller(65), "synthetic")
        self.assertEqual(result["two_admitted_two_success_two_rejected_branches"], 1)
        self.assertEqual(result["ai40_candidate_occurrences"], 0)

    def test_nonzero_safety_is_reported(self):
        data = controller(40)
        data["deadline_misses"] = 1
        result = audit_controller(data, "synthetic")
        self.assertFalse(result["all_declared_safety_zero"])


if __name__ == "__main__":
    unittest.main()
