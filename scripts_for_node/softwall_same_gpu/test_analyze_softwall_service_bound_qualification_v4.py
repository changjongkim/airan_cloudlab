#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from analyze_softwall_service_bound_qualification_v4 import aggregate


class ServiceBoundQualificationV4Test(unittest.TestCase):
    def test_two_distinct_nodes_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            paths = {}
            for name in ("v3", "c139", "c140", "c141"):
                paths[name] = root / (name + ".json")
                paths[name].write_text("{}")
            v3 = {"all_pass": True}
            c139 = {"nodes": ["a"], "totals": {"arms": 6, "radio_records": 16320,
                    "radio_records_after_fault_detection": 14919},
                    "control_evidence": {"declared_per_rpc_admission_bound_ms": 7.0,
                    "admission_bound_exceeded": 0, "records": 2020, "maximum_elapsed_ms": 5.6}}
            c140 = {"nodes": ["a"], "arms": [{}, {}],
                    "totals": {"radio_records": 2560, "candidate_branches": 309,
                    "ai35_candidate_exchanges": 8,
                    "minimum_physical_guarded_horizon_margin_ms": 43.6},
                    "model_counterfactual": {"effective_transaction_bound_ms": 58.0,
                    "static_margin_ms": -5.0, "conditional_margin_ms": 0.0}}
            c141 = {"all_pass": True, "nodes": ["b"],
                    "totals": {"radio_records_after_fault_detection": 14914,
                               "ai35_candidate_exchanges": 7},
                    "control_campaign": {"arms": 6, "radio_records": 16320,
                                         "radio_records_after_fault_detection": 14914},
                    "ai35_campaign": {"arms": 2, "radio_records": 2560,
                                      "candidate_branches": 293,
                                      "ai35_candidate_exchanges": 7,
                                      "minimum_physical_guarded_horizon_margin_ms": 44.1},
                    "control_evidence": {"declared_per_rpc_admission_bound_ms": 7.0,
                    "admission_bound_exceeded": 0, "records": 2010, "maximum_elapsed_ms": 5.8},
                    "ai35_counterfactual": {"effective_transaction_bound_ms": 58.0,
                    "static_margin_ms": -5.0, "conditional_margin_ms": 0.0}}
            result = aggregate(v3, c139, c140, c141, paths)
            self.assertTrue(result["all_pass"])
            self.assertEqual(result["control_fault_evidence"]["attempted_rpc_records"], 4030)
            self.assertEqual(result["conditional_ai35_evidence"]["ai35_candidate_exchanges"], 15)


if __name__ == "__main__":
    unittest.main()
