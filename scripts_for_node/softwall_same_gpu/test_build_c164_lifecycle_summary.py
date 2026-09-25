#!/usr/bin/env python3.11

import copy
import unittest

from build_c164_lifecycle_summary import build


def passed(status):
    return {"status": status, "all_pass": True}


class LifecycleSummaryTests(unittest.TestCase):
    def fixtures(self):
        return {
            "warm": passed("C162_TWO_NODE_BOUNDARY_PASS"),
            "model": passed("C164_LIFECYCLE_MODEL_PASS"),
            "idle": passed("C164_IDLE30_TWO_NODE_PASS"),
            "restart": passed("C164_MPS_RESTART_TWO_NODE_PASS"),
            "reload": passed("C164_QWEN_RELOAD_TWO_NODE_PASS"),
            "reconnect_model": passed("C164_RECONNECT_MODEL_PASS"),
            "reconnect_physical": passed("C164_RECONNECT_TWO_NODE_PASS"),
            "process_replacement": {
                "status": "C164_PROCESS_REPLACEMENT_PHYSICAL_PASS",
                "all_pass": True,
                "campaign": "development",
            },
            "gc": {"all_contract_gates_pass": False,
                   "arm_gates": {"a_on": {"sample_safety": False}}},
        }

    def test_scoped_matrix_is_consistent(self):
        value = build(**self.fixtures())
        self.assertTrue(value["all_inputs_consistent"])
        self.assertFalse(value["complete"])
        self.assertTrue(value["claim_scope_complete"])
        self.assertEqual(
            value["paper_gate_status"],
            "C164_CLAIM_SCOPED_LIFECYCLE_COMPLETE",
        )
        self.assertEqual(value["counts"]["qualified_or_partial_modes"], 5)
        self.assertEqual(value["counts"]["unqualified_modes"], 5)
        self.assertIn("audited manuscript draft", value["next_internal_gate"])
        self.assertIn("venue submission", value["next_internal_gate"])
        self.assertNotIn("process replacement", value["next_internal_gate"])

    def test_reload_is_not_optional_availability(self):
        value = build(**self.fixtures())
        row = value["modes"]["qwen_reload_first"]
        self.assertTrue(row["mandatory_continuity"])
        self.assertFalse(row["optional_boundary"])
        self.assertFalse(row["availability_during_transition"])

    def test_restart_is_after_requalification_only(self):
        value = build(**self.fixtures())
        row = value["modes"]["mps_restart_first"]
        self.assertTrue(row["optional_boundary"])
        self.assertFalse(row["availability_during_transition"])

    def test_channel_reconnect_does_not_promote_process_replacement(self):
        value = build(**self.fixtures())
        channel = value["modes"]["worker_channel_reconnect_same_epoch"]
        replacement = value["modes"]["worker_process_replacement"]
        self.assertTrue(channel["status"].startswith("QUALIFIED"))
        self.assertFalse(channel["optional_boundary"])
        self.assertTrue(replacement["status"].startswith("UQ_"))
        self.assertEqual(replacement["status"], "UQ_SINGLE_NODE_DEVELOPMENT_ONLY")

    def test_failed_input_makes_matrix_inconsistent(self):
        inputs = self.fixtures(); inputs["reload"] = copy.deepcopy(inputs["reload"])
        inputs["reload"]["all_pass"] = False
        self.assertFalse(build(**inputs)["all_inputs_consistent"])


if __name__ == "__main__":
    unittest.main()
