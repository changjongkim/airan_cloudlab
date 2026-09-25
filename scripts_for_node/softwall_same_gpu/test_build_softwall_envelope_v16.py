#!/usr/bin/env python3

import unittest

from build_softwall_envelope_v16 import V15_ID, V16_ID, build


class BuildEnvelopeV16Test(unittest.TestCase):
    def test_demotes_v15_and_adds_qualified_v16(self):
        base = {
            "schema": "old",
            "status": "old",
            "scope": "scope",
            "scenarios": [{
                "id": V15_ID,
                "qualification_nodes": ["n1", "n2"],
                "evidence": ["old"],
            }],
        }
        result = lambda node: {
            "all_pass": True,
            "nodes": [node],
            "gates": {"abort_fault_injected": True},
            "totals": {
                "radio_records": 1600,
                "radio_after_fault": 1000,
                "suppressed_prepare_due_owned_token": 10,
                "maximum_unlaunched_tokens": 1,
            },
        }
        value = build(base, [result("n3"), result("n4")])
        rows = {row["id"]: row for row in value["scenarios"]}
        self.assertFalse(rows[V15_ID]["four_point_control_fault_qualified"])
        self.assertTrue(rows[V16_ID]["four_point_control_fault_qualified"])
        self.assertEqual(rows[V16_ID]["qualification_nodes"],
                         ["n1", "n2", "n3", "n4"])


if __name__ == "__main__":
    unittest.main()
