#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "compare_v10_v11", HERE / "compare_softwall_envelope_v10_v11.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EnvelopeV10V11RegressionTest(unittest.TestCase):
    def test_frozen_full_control_mode(self):
        old = {
            "results": [{
                "id": MODULE.TARGET,
                "status": "QSU",
                "endpoint_admitted_cells_by_home": [4, 4],
                "endpoint_rejected_cells_by_home": [0, 0],
                "home_max_released_recovery_slack_ms": [75.0, 75.0],
                "home_max_safe_exchange_transaction_ms": [38.0, 38.0],
                "ai_classes": [{
                    "bound_ms": 40.0,
                    "effective_transaction_bound_ms": 55.0,
                    "exchange_only_candidate": False,
                    "exchange_only_candidate_homes": [],
                }],
            }],
        }
        new = {
            "results": [{
                "id": MODULE.TARGET,
                "status": "QSU",
                "home_endpoint_ring_depths": [[1, 1], [1, 1]],
                "endpoint_admitted_cells_by_home": [2, 2],
                "endpoint_rejected_cells_by_home": [2, 2],
                "rejected_recovery_before_exchange": [False, False],
                "home_max_released_recovery_slack_ms": [50.0, 50.0],
                "home_max_safe_exchange_transaction_ms": [58.0, 58.0],
                "ai_classes": [{
                    "bound_ms": 40.0,
                    "effective_transaction_bound_ms": 55.0,
                    "exchange_only_candidate": True,
                    "exchange_only_candidate_homes": [0, 1],
                }],
            }],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            v10 = directory / "v10.json"
            v11 = directory / "v11.json"
            v10.write_text(json.dumps(old))
            v11.write_text(json.dumps(new))
            result = MODULE.compare(v10, v11)
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["v11"]["model_margin_ms"], [3.0, 3.0])
        self.assertTrue(result["v11"]["ai40_exchange_only_candidate"])


if __name__ == "__main__":
    unittest.main()
