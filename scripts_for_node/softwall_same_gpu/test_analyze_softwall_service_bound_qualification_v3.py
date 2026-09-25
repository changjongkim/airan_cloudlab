#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from analyze_softwall_service_bound_qualification_v3 import qualification


class ServiceBoundQualificationV3Test(unittest.TestCase):
    def write(self, root, name, value):
        path = root / name
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def test_timeout_exceedance_forces_supersession(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2 = self.write(root, "v2.json", {"all_pass": True})
            c138 = self.write(root, "c138.json", {
                "all_pass": False,
                "gates": {"system_safety": True, "rpc_wall_bound": False},
                "control_evidence": {"max_elapsed_ms": 5.8, "records": 2, "wall_timeout_exceeded": 1},
            })
            c139 = self.write(root, "c139.json", {
                "all_pass": True, "nodes": ["n"],
                "totals": {"arms": 6, "radio_records": 10, "radio_records_after_fault_detection": 8},
                "control_evidence": {
                    "declared_per_rpc_admission_bound_ms": 7.0, "socket_timeout_ms": 5.0,
                    "admission_bound_exceeded": 0, "maximum_elapsed_ms": 5.9, "records": 12,
                },
            })
            c140 = self.write(root, "c140.json", {
                "all_pass": True, "nodes": ["n"], "arms": [{}, {}],
                "totals": {"radio_records": 20, "candidate_branches": 3,
                           "ai35_candidate_exchanges": 8,
                           "minimum_physical_guarded_horizon_margin_ms": 1.0},
                "model_counterfactual": {"effective_transaction_bound_ms": 58.0,
                                         "static_margin_ms": -5.0,
                                         "conditional_margin_ms": 0.0},
            })
            v13 = self.write(root, "v13.json", {"all_pass": True})
            result = qualification(root, v2, c138, c139, c140, v13)
            self.assertTrue(result["all_pass"])
            self.assertEqual(result["status"], "FINITE_SAMPLE_CORRECTED_CONTROL_PASS")
            self.assertEqual(result["control_contract"]["control_transaction_bound_ms"], 21.0)

    def test_missing_falsification_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            v2 = self.write(root, "v2.json", {"all_pass": True})
            c138 = self.write(root, "c138.json", {
                "all_pass": True,
                "gates": {"system_safety": True, "rpc_wall_bound": True},
                "control_evidence": {"max_elapsed_ms": 4.8, "records": 2, "wall_timeout_exceeded": 0},
            })
            c139 = self.write(root, "c139.json", {
                "all_pass": True, "nodes": ["n"],
                "totals": {"arms": 6, "radio_records": 10, "radio_records_after_fault_detection": 8},
                "control_evidence": {
                    "declared_per_rpc_admission_bound_ms": 7.0, "socket_timeout_ms": 5.0,
                    "admission_bound_exceeded": 0, "maximum_elapsed_ms": 5.9, "records": 12,
                },
            })
            c140 = self.write(root, "c140.json", {
                "all_pass": True, "nodes": ["n"], "arms": [{}, {}],
                "totals": {"radio_records": 20, "candidate_branches": 3,
                           "ai35_candidate_exchanges": 8,
                           "minimum_physical_guarded_horizon_margin_ms": 1.0},
                "model_counterfactual": {"effective_transaction_bound_ms": 58.0,
                                         "static_margin_ms": -5.0,
                                         "conditional_margin_ms": 0.0},
            })
            v13 = self.write(root, "v13.json", {"all_pass": True})
            result = qualification(root, v2, c138, c139, c140, v13)
            self.assertFalse(result["all_pass"])
            self.assertEqual(result["status"], "UNQUALIFIED")


if __name__ == "__main__":
    unittest.main()
