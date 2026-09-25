#!/usr/bin/env python3.11

import json
import unittest
from pathlib import Path

from analyze_confirm156_two_node import combine


ROOT = Path(__file__).resolve().parents[2]


class Confirm156TwoNodeTest(unittest.TestCase):
    def test_sealed_two_node_result_passes(self):
        base = ROOT / "results/softwall_multigpu"
        first = "confirm156_timeline_attempt3_job58853926"
        second = "confirm156b_timeline_holdout_job58854401"
        load = lambda name: json.loads((base / name).read_text(encoding="utf-8"))
        value = combine(
            load(f"{first}_result.json"), load(f"{second}_result.json"),
            load(f"{first}_protocol.json"), load(f"{second}_protocol.json"),
        )
        self.assertTrue(value["all_pass"])
        self.assertEqual(value["totals"]["qwen_runtime_kernels"], 2448)
        self.assertEqual(value["totals"]["recovery_runtime_kernels"], 212)
        self.assertEqual(value["totals"]["deadline_misses"], 0)


if __name__ == "__main__":
    unittest.main()
