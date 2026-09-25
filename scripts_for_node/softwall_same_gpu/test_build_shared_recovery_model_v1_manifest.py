#!/usr/bin/env python3.11

import unittest

from build_shared_recovery_model_v1_manifest import build_manifest


class SharedRecoveryManifestTest(unittest.TestCase):
    def test_manifest_is_deterministic_and_complete(self):
        first = build_manifest()
        second = build_manifest()
        self.assertEqual(first, second)
        self.assertEqual(first["status"], "MODEL_PASS_PHYSICAL_UQ")
        self.assertTrue(first["result_all_pass"])
        self.assertEqual(len(first["files"]), 5)
        self.assertTrue(all(len(record["sha256"]) == 64
                            for record in first["files"]))


if __name__ == "__main__":
    unittest.main()
