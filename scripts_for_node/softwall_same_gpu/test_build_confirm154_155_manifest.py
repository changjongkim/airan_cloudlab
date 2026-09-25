#!/usr/bin/env python3

import unittest

from build_confirm154_155_manifest import build_manifest, collect_files


class Confirm154155ManifestTest(unittest.TestCase):
    def test_manifest_is_complete_and_deterministic(self):
        first = build_manifest()
        second = build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["file_count"], len(collect_files()))
        self.assertEqual(first["nodes"], ["nid001032", "nid001025"])
        self.assertEqual(first["job_ids"], ["58853453", "58853542"])
        self.assertTrue(first["combined_all_pass"])
        self.assertEqual(
            {row["path"] for row in first["files"]}, set(collect_files())
        )


if __name__ == "__main__":
    unittest.main()

