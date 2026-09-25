#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_softwall_v13_artifact_manifests import matching, protocol_sources, relative_hashes


class V13ManifestTest(unittest.TestCase):
    def test_protocol_source_hash_is_enforced(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source.py"
            source.write_text("x = 1\n", encoding="utf-8")
            expected = hashlib.sha256(source.read_bytes()).hexdigest()
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"source_sha256": {"source.py": expected}}), encoding="utf-8")
            paths, mismatches = protocol_sources(root, protocol)
            self.assertEqual(paths, [source])
            self.assertEqual(mismatches, [])
            source.write_text("x = 2\n", encoding="utf-8")
            _, mismatches = protocol_sources(root, protocol)
            self.assertEqual(len(mismatches), 1)

    def test_relative_hashes_deduplicates(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact = root / "a.json"
            artifact.write_text("{}\n", encoding="utf-8")
            result = relative_hashes(root, [artifact, artifact])
            self.assertEqual(list(result), ["a.json"])

    def test_campaign_collection_excludes_its_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result_dir = root / "results/softwall_multigpu"
            (result_dir / "raw").mkdir(parents=True)
            result = result_dir / "confirm139_result.json"
            manifest = result_dir / "confirm139_artifact_manifest.json"
            result.write_text("{}\n", encoding="utf-8")
            manifest.write_text("{}\n", encoding="utf-8")
            self.assertEqual(matching(root, "confirm139"), [result])


if __name__ == "__main__":
    unittest.main()
