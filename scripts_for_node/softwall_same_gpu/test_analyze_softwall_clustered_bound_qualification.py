import json
import tempfile
import unittest
from pathlib import Path

from analyze_softwall_clustered_bound_qualification import analyze, zero_failure_upper


class ClusteredBoundQualificationTest(unittest.TestCase):
    def make_fixture(self, exceed_ai=False, mismatch=False):
        temporary = tempfile.TemporaryDirectory()
        root = Path(temporary.name)
        result_dir = root / "results/softwall_multigpu"
        result_dir.mkdir(parents=True)
        campaigns = {}
        for campaign, count, node in (("c135", 2, "n1"), ("c136", 6, "n2")):
            arm_entries = []
            for arm_index in range(count):
                homes = []
                for home in range(2):
                    controller = {
                        "system": "softwall", "cells": 4, "period_ms": 180.0,
                        "deadline_ms": 155.0, "nrx_bound_ms": 45.0,
                        "conv_bound_ms": 25.0, "commit_guard_ms": 2.0,
                        "nrx_endpoints": 2, "gc_mode": "off",
                        "global_broker_rpc_timeout_ms": 5.0,
                        "global_broker_transaction_budget_ms": 15.0,
                        "admission_ai_guard_ms": 17.0,
                        "routing_policy": "shortest_queue", "fault_pattern": "mixed",
                        "early_mandatory": "on", "ai_during_nrx": "async_one_if_edf_fit",
                        "trace_sha256": "trace", "ai_bound_ms": {"128": 40.0},
                        "host": node,
                        "deadline_misses": 0, "nrx_bound_violations": 0,
                        "conv_bound_violations": 0, "conv_path_bound_violations": 0,
                        "background_budget_violations": 0, "background_horizon_violations": 0,
                        "records": [{"nrx_response_ms": 20.0,
                                     "conventional_host_path_ms": 6.0}],
                        "background_records": [{"execution_ms": 41.0 if exceed_ai and campaign == "c136" and arm_index == 0 and home == 0 else 28.0}],
                    }
                    if mismatch and campaign == "c136" and arm_index == 0:
                        controller["gc_mode"] = "on"
                    controller_path = result_dir / ("%s_a%d_h%d.json" % (campaign, arm_index, home))
                    controller_path.write_text(json.dumps(controller))
                    homes.append({"controller": str(controller_path)})
                arm_path = result_dir / ("%s_a%d_result.json" % (campaign, arm_index))
                arm_path.write_text(json.dumps({"all_pass": True, "homes": homes}))
                arm_entries.append({"result": str(arm_path)})
            campaign_path = result_dir / ("%s_campaign.json" % campaign)
            campaign_path.write_text(json.dumps({"all_pass": True, "arms": arm_entries}))
            campaigns[campaign] = campaign_path
        return temporary, root, campaigns

    def test_zero_failure_upper(self):
        self.assertAlmostEqual(zero_failure_upper(2), 0.776393202250021, places=12)
        self.assertIsNone(zero_failure_upper(0))

    def test_passes_consistent_two_node_fixture(self):
        temporary, root, campaigns = self.make_fixture()
        self.addCleanup(temporary.cleanup)
        result = analyze(root, campaigns["c135"], campaigns["c136"])
        self.assertEqual(result["status"], "FINITE_SAMPLE_PASS")
        self.assertEqual(result["physical_nodes"], ["n1", "n2"])
        self.assertEqual(result["components"]["ai_execution_ms"]["arm_process_units"], 8)
        self.assertEqual(result["c136_new_node_holdout_components"]["ai_execution_ms"]["arm_process_units"], 6)

    def test_fails_component_exceedance(self):
        temporary, root, campaigns = self.make_fixture(exceed_ai=True)
        self.addCleanup(temporary.cleanup)
        result = analyze(root, campaigns["c135"], campaigns["c136"])
        self.assertEqual(result["status"], "UNQUALIFIED")
        self.assertEqual(result["components"]["ai_execution_ms"]["sample_exceedances"], 1)

    def test_fails_mode_mismatch(self):
        temporary, root, campaigns = self.make_fixture(mismatch=True)
        self.addCleanup(temporary.cleanup)
        result = analyze(root, campaigns["c135"], campaigns["c136"])
        self.assertEqual(result["status"], "UNQUALIFIED")
        self.assertFalse(result["gates"]["mode_signature_consistent"])


if __name__ == "__main__":
    unittest.main()
