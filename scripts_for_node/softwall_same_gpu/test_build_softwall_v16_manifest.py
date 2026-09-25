#!/usr/bin/env python3

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from build_softwall_v16_manifest import protocol_sources, verify_base_manifest


class V16ManifestTest(unittest.TestCase):
    def test_base_manifest_integrity_is_transitively_checked(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            artifact = root / "artifact.json"
            artifact.write_text("{}\n")
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps({
                "files": {
                    "artifact.json": hashlib.sha256(
                        artifact.read_bytes()
                    ).hexdigest(),
                },
            }))
            _, mismatches = verify_base_manifest(root, manifest)
            self.assertEqual(mismatches, [])
            artifact.write_text("changed\n")
            self.assertEqual(len(verify_base_manifest(root, manifest)[1]), 1)

    def test_container_mount_source_key_is_resolved(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            source = root / "scripts_for_node/softwall_same_gpu/source.py"
            source.parent.mkdir(parents=True)
            source.write_text("x = 1\n")
            protocol = root / "protocol.json"
            protocol.write_text(json.dumps({"source_sha256": {
                "/softwall/source.py": hashlib.sha256(
                    source.read_bytes()
                ).hexdigest(),
            }}))
            paths, mismatches = protocol_sources(root, protocol)
            self.assertEqual(paths, [source])
            self.assertEqual(mismatches, [])


if __name__ == "__main__":
    unittest.main()
