#!/usr/bin/env python3.11

import unittest

from build_confirm151_152_manifest import build_manifest


class Confirm151152ManifestTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_complete(self):
        first = build_manifest()
        second = build_manifest()
        self.assertEqual(first, second)
        self.assertTrue(first["combined_all_pass"])
        self.assertEqual(
            first["status"],
            "TWO_NODE_PHYSICAL_PATH_PASS_V17_REMAINS_UQ",
        )
        self.assertEqual(first["file_count"], 24)
        self.assertTrue(all(
            len(record["sha256"]) == 64 for record in first["files"]
        ))


if __name__ == "__main__":
    unittest.main()
