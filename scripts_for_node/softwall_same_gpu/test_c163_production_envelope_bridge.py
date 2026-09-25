#!/usr/bin/env python3.11

import copy
import unittest

from c163_production_envelope_bridge import classify_document


def bridge_document():
    records = []
    for index, outcome in enumerate(("timely_success", "failed_or_late")):
        records.append({
            "request_id": f"r{index}", "batch_id": "b0",
            "home_id": f"h{index}", "cell_id": index, "slot_id": 7,
            "release_ns": 10 + index, "expiry_ns": 200 + index,
            "nrx_admitted": True, "nrx_outcome_at_decision": outcome,
        })
    return {
        "clock": {"calibration": {"max_error_ns": 1}},
        "mode": {
            "lifecycle": "warm", "node_bound_status": "qualified",
            "recovery_capacity": 1, "recovery_bound_ns": 25,
            "nrx_bound_ns": 45, "control_bound_ns": 5,
            "guard_ns": 2, "ai_class_bounds_ns": {"128": 40},
        },
        "records": records,
        "decisions": [{
            "batch_id": "b0", "decision_ns": 60,
            "mandatory_request_ids": ["r0", "r1"],
            "unresolved_request_ids": ["r1"],
            "ai_context_length": 128, "ai_deadline_ns": 150,
            "ai_lease_accepted": True,
        }],
    }


class ProductionEnvelopeBridgeTest(unittest.TestCase):
    def test_safe_ai_and_recovery_is_qsu(self):
        result = classify_document(bridge_document())[0]
        self.assertEqual(result.state, "QSU")
        self.assertTrue(result.mandatory_all_fail_safe)
        self.assertTrue(result.ai_safe)
        self.assertTrue(result.decision_matches)

    def test_ai_deadline_can_make_mandatory_safe_state_qsn(self):
        value = bridge_document()
        value["decisions"][0]["ai_deadline_ns"] = 100
        value["decisions"][0]["ai_lease_accepted"] = False
        result = classify_document(value)[0]
        self.assertEqual(result.state, "QSN")
        self.assertTrue(result.current_mandatory_safe)
        self.assertFalse(result.ai_safe)
        self.assertTrue(result.decision_matches)

    def test_five_debts_are_mandatory_infeasible(self):
        value = bridge_document()
        value["records"] = []
        for index in range(5):
            value["records"].append({
                "request_id": f"r{index}", "batch_id": "b0",
                "home_id": f"h{index}", "cell_id": index, "slot_id": 7,
                "release_ns": 10, "expiry_ns": 150,
                "nrx_admitted": True,
                "nrx_outcome_at_decision": "failed_or_late",
            })
        ids = [row["request_id"] for row in value["records"]]
        value["decisions"][0].update({
            "mandatory_request_ids": ids, "unresolved_request_ids": ids,
            "ai_lease_accepted": False,
        })
        result = classify_document(value)[0]
        self.assertEqual(result.state, "MI")
        self.assertFalse(result.mandatory_all_fail_safe)
        self.assertTrue(result.decision_matches)

    def test_nonwarm_or_unqualified_mode_is_uq(self):
        for field, replacement in (("lifecycle", "cold"),
                                   ("node_bound_status", "failed")):
            value = bridge_document()
            value["mode"][field] = replacement
            value["decisions"][0]["ai_lease_accepted"] = False
            with self.subTest(field=field):
                result = classify_document(value)[0]
                self.assertEqual(result.state, "UQ")
                self.assertTrue(result.decision_matches)

    def test_no_ai_request_is_qsn(self):
        value = bridge_document()
        value["decisions"][0].update({
            "ai_context_length": None, "ai_deadline_ns": None,
            "ai_lease_accepted": False,
        })
        result = classify_document(value)[0]
        self.assertEqual(result.state, "QSN")
        self.assertEqual(result.reason, "no_ai_request")

    def test_not_admitted_nrx_owes_recovery_at_release(self):
        value = bridge_document()
        value["records"] = [copy.deepcopy(value["records"][1])]
        row = value["records"][0]
        row.update({
            "request_id": "r0", "nrx_admitted": False,
            "nrx_outcome_at_decision": "not_admitted",
            "release_ns": 10, "expiry_ns": 50,
        })
        value["decisions"][0].update({
            "decision_ns": 20, "mandatory_request_ids": ["r0"],
            "unresolved_request_ids": ["r0"], "ai_context_length": None,
            "ai_deadline_ns": None, "ai_lease_accepted": False,
        })
        result = classify_document(value)[0]
        self.assertEqual(result.state, "QSN")
        self.assertEqual(result.all_fail_finish_ns, 35)

    def test_large_feasible_state_uses_verified_certified_schedule(self):
        value = bridge_document()
        value["mode"].update({
            "recovery_capacity": 2, "recovery_bound_ns": 10,
            "nrx_bound_ns": 20,
        })
        value["records"] = []
        for index in range(11):
            value["records"].append({
                "request_id": f"r{index}", "batch_id": "b0",
                "home_id": f"h{index % 2}", "cell_id": index, "slot_id": 7,
                "release_ns": 10, "expiry_ns": 200,
                "nrx_admitted": True,
                "nrx_outcome_at_decision": "failed_or_late",
            })
        ids = [row["request_id"] for row in value["records"]]
        value["decisions"][0].update({
            "decision_ns": 30, "mandatory_request_ids": ids,
            "unresolved_request_ids": ids, "ai_context_length": None,
            "ai_deadline_ns": None, "ai_lease_accepted": False,
        })
        result = classify_document(value)[0]
        self.assertEqual(result.state, "QSN")
        self.assertTrue(result.mandatory_all_fail_safe)
        self.assertEqual(result.all_fail_certificate_method, "certified")


if __name__ == "__main__":
    unittest.main()
