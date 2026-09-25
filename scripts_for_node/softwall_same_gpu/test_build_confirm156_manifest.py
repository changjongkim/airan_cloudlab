#!/usr/bin/env python3.11

import unittest

from build_confirm156_manifest import build_manifest


class Confirm156ManifestTest(unittest.TestCase):
    def test_manifest_is_complete_and_deterministic(self):
        first = build_manifest()
        second = build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["physical_runs"], 4)
        self.assertEqual(first["failed_attempts_preserved"], 2)
        self.assertEqual(first["passing_job"], "58853926")
        self.assertEqual(first["passing_node"], "nid001085")
        self.assertEqual(first["passing_nodes"], ["nid001085", "nid001064"])
        self.assertGreater(first["file_count"], 60)


if __name__ == "__main__":
    unittest.main()
