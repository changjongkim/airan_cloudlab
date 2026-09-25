#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from analyze_confirm141_v13_independent_requalification import combine


class Confirm141AnalyzerTest(unittest.TestCase):
    def make_inputs(self, root, node="new"):
        control = {
            "all_pass": True, "nodes": [node],
            "totals": {"arms": 6, "radio_records": 60, "radio_records_after_fault_detection": 50},
            "control_evidence": {"declared_per_rpc_admission_bound_ms": 7.0,
                                 "admission_bound_exceeded": 0, "records": 20},
        }
        ai35 = {
            "all_pass": True, "nodes": [node], "arms": [{}, {}],
            "totals": {"radio_records": 20, "ai35_candidate_exchanges": 3},
            "model_counterfactual": {"static_all_fail_slack_ms": 53.0,
                                     "effective_transaction_bound_ms": 58.0,
                                     "conditional_decision_window_ms": 58.0,
                                     "static_margin_ms": -5.0,
                                     "conditional_margin_ms": 0.0},
        }
        cp = root / "control.json"; cp.write_text(json.dumps(control))
        ap = root / "ai.json"; ap.write_text(json.dumps(ai35))
        return control, ai35, cp, ap

    def test_independent_node_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control, ai35, cp, ap = self.make_inputs(root)
            result = combine(control, ai35, cp, ap, ["old"])
            self.assertTrue(result["all_pass"])
            self.assertEqual(result["totals"]["arms"], 8)

    def test_reused_node_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            control, ai35, cp, ap = self.make_inputs(root, node="old")
            result = combine(control, ai35, cp, ap, ["old"])
            self.assertFalse(result["all_pass"])
            self.assertFalse(result["gates"]["node_is_independent"])


if __name__ == "__main__":
    unittest.main()
