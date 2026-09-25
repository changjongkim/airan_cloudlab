#!/usr/bin/env python3.11

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("evaluate_softwall_production_gates.py")
SPEC = importlib.util.spec_from_file_location("production_gates", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class ProductionGateTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parents[1])
        self.root = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def write(self, name, value):
        path = self.root / name
        path.write_text(json.dumps(value))
        return path

    def valid_p1(self):
        return {
            "schema": "softwall-du-timing-contract-v2",
            "contract_kind": "production-du",
            "clock": {"synchronized": True},
            "expiry_contract": {"synthetic": False, "derived_from_ue_k2_n2": False},
            "records": [{
                "iq_ready_ns": 1, "phy_submit_ns": 2, "crc_visible_ns": 3,
                "fapi_publish_ns": 4, "mac_consume_ns": 5, "expiry_ns": 6,
                "radio_commit_count": 1,
            }],
        }

    def test_missing_inputs_fail_closed(self):
        result = MODULE.evaluate({key: None for key in ("p1", "p2", "p3", "p4")})
        self.assertFalse(result["all_pass"])
        self.assertEqual(result["first_blocker"], "P1_LIVE_DU_CLOCK")

    def test_synthetic_p1_is_rejected(self):
        p1 = self.valid_p1()
        p1["expiry_contract"]["synthetic"] = True
        passed, failures = MODULE.evaluate_p1(p1)
        self.assertFalse(passed)
        self.assertIn("expiry is synthetic or unspecified", failures)

    def test_development_p2_is_rejected_even_if_timely(self):
        result = {
            "iterations": 10,
            "deadline_counts": {"parallel_pair_wall_le_deadline": 10},
            "correctness": {"neural_correct": 10, "conventional_correct": 10},
            "timing_contract_sha256": "abc",
            "campaign_role": "development",
        }
        passed, failures = MODULE.evaluate_p2(result, "abc")
        self.assertFalse(passed)
        self.assertIn("fast path is not an independent holdout", failures)

    def test_p4_must_bind_exact_inputs(self):
        result = {
            "schema": "softwall-integrated-production-holdout-v1",
            "all_pass": True,
            "input_sha256": {
                "p1_timing_contract": "p1", "p2_fast_path": "wrong",
                "p3_channel_holdout": "p3",
            },
            "checks": {key: True for key in (
                "deadline_misses_zero", "single_commit_violations_zero",
                "credit_violations_zero", "correlated_all_fail_exercised",
                "bounded_qwen_exercised", "independent_nodes_pass",
            )},
        }
        passed, failures = MODULE.evaluate_p4(result, "p1", "p2", "p3")
        self.assertFalse(passed)
        self.assertIn("integrated result does not bind p2_fast_path", failures)


if __name__ == "__main__":
    unittest.main()
