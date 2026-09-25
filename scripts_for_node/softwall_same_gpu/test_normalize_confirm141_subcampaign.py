#!/usr/bin/env python3

import json
import tempfile
import unittest
from pathlib import Path

from normalize_confirm141_subcampaign import normalize


class NormalizeConfirm141Test(unittest.TestCase):
    def test_ai35_removes_old_same_node_claim(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "input.json"
            source.write_text(json.dumps({"schema": "old", "claim_boundary": "Same A100 node as C139", "all_pass": True}))
            result = normalize(json.loads(source.read_text()), source, "ai35")
            self.assertEqual(result["schema"], "softwall-confirm141-ai35-requalification-v1")
            self.assertNotIn("Same A100 node as C139", result["claim_boundary"])
            self.assertTrue(result["all_pass"])


if __name__ == "__main__":
    unittest.main()
