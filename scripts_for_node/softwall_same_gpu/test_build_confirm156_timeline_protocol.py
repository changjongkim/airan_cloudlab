#!/usr/bin/env python3.11

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from build_confirm156_timeline_protocol import build_protocol, source_hashes


class Confirm156TimelineProtocolTest(unittest.TestCase):
    def test_protocol_is_deterministic_and_complete(self) -> None:
        scripts_root = Path(__file__).resolve().parent
        task1_root = scripts_root.parent / "task1"
        kwargs = {
            "label": "confirm156_test",
            "scripts_root": scripts_root,
            "task1_root": task1_root,
            "seed_base": 25_600_000,
            "excluded_nodes": "node-b,node-a,node-b",
            "snr_db": 20.0,
            "warmup": 10,
            "release_lead_ms": 3000.0,
        }
        first = build_protocol(**kwargs)
        second = build_protocol(**kwargs)
        self.assertEqual(first, second)
        self.assertEqual(first["branch_order"], ["conditional_open"])
        self.assertEqual(first["excluded_nodes"], ["node-a", "node-b"])
        self.assertEqual(
            first["source_sha256"], source_hashes(scripts_root, task1_root)
        )
        self.assertEqual(first["mode"]["expiry_ms"], 155.0)
        self.assertEqual(first["profiler"]["export"], "sqlite")
        self.assertEqual(len(first["arms"]), 1)
        self.assertEqual(len(first["arms"][0]["seeds"]["receiver"]), 5)


if __name__ == "__main__":
    unittest.main()
