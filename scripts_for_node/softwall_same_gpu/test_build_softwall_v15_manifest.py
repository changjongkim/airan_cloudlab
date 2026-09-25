#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_softwall_v15_manifest import protocol_sources, relative_hashes


class V15ManifestTest(unittest.TestCase):
    def test_protocol_source_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("x = 1\n")
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"source_sha256": {
                "source.py": hashlib.sha256(source.read_bytes()).hexdigest(),
            }}))
            paths, mismatches = protocol_sources(root, protocol)
            self.assertEqual(paths, [source])
            self.assertEqual(mismatches, [])
            source.write_text("x = 2\n")
            self.assertEqual(len(protocol_sources(root, protocol)[1]), 1)

    def test_relative_hashes_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact = root / "a.json"
            artifact.write_text("{}\n")
            self.assertEqual(list(relative_hashes(root, [artifact, artifact])),
                             ["a.json"])


if __name__ == "__main__":
    unittest.main()
