#!/usr/bin/env python3

import unittest

from build_confirm153_manifest import build_manifest, collect_files


class Confirm153ManifestTest(unittest.TestCase):
    def test_manifest_is_complete_and_deterministic(self):
        first = build_manifest()
        second = build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["files"], second["files"])
        self.assertEqual(first["file_count"], len(collect_files()))
        self.assertEqual(first["development_attempts"], 3)
        self.assertEqual(first["failed_attempts_preserved"], 2)
        self.assertEqual(first["passing_job"], "58852924")
        self.assertEqual(first["status"], "DEVELOPMENT_CANARY_PASS_V17_REMAINS_UQ")
        self.assertEqual(
            {row["path"] for row in first["files"]}, set(collect_files())
        )


if __name__ == "__main__":
    unittest.main()

