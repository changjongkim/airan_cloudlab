#!/usr/bin/env python3

import importlib.util
import json
import pathlib
import tempfile
import unittest


HERE = pathlib.Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location(
    "compare_v11_v12", HERE / "compare_softwall_envelope_v11_v12.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class EnvelopeV11V12RegressionTest(unittest.TestCase):
    def test_completion_guard_correction_preserves_candidate(self):
        shared = {
            "id": MODULE.TARGET,
            "status": "QSU",
            "endpoint_admitted_cells_by_home": [2, 2],
            "home_max_safe_exchange_transaction_ms": [58.0, 58.0],
        }
        old = {
            "counts": {"QSU": 5, "QSN": 0, "MI": 3, "UQ": 10},
            "results": [dict(shared, ai_classes=[{
                "bound_ms": 40.0,
                "control_budget_ms": 15.0,
                "effective_transaction_bound_ms": 55.0,
                "exchange_only_candidate": True,
                "exchange_only_candidate_homes": [0, 1],
            }])],
        }
        new = {
            "counts": old["counts"],
            "results": [dict(shared, ai_classes=[{
                "bound_ms": 40.0,
                "control_budget_ms": 15.0,
                "ai_completion_guard_ms": 2.0,
                "effective_transaction_bound_ms": 57.0,
                "exchange_only_candidate": True,
                "exchange_only_candidate_homes": [0, 1],
            }])],
        }
        with tempfile.TemporaryDirectory() as directory:
            directory = pathlib.Path(directory)
            v11 = directory / "v11.json"
            v12 = directory / "v12.json"
            v11.write_text(json.dumps(old))
            v12.write_text(json.dumps(new))
            result = MODULE.compare(v11, v12)
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["v12"]["model_margin_ms"], [1.0, 1.0])


if __name__ == "__main__":
    unittest.main()
