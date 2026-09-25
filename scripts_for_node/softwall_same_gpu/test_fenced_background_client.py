#!/usr/bin/env python3.11
"""Fault-path checks for GPU-completion-marker based AI lease retirement."""

import json
import tempfile
import time
import unittest
from pathlib import Path

from fenced_background_client import FencedBackgroundClient


class _TimedOutChannel:
    def __init__(self, marker: Path | None) -> None:
        self.marker = marker

    def write(self, payload: bytes) -> None:
        if self.marker is None:
            return
        request = json.loads(payload)
        self.marker.write_text(json.dumps({
            "lease_id": request["lease_id"],
            "completed_ns": time.perf_counter_ns(),
            "gpu_ms": 1.0,
        }), encoding="utf-8")

    def readline(self) -> bytes:
        time.sleep(0.001)
        raise TimeoutError("injected delayed RPC response")


class FencedBackgroundClientTest(unittest.TestCase):
    def _client(self, marker: Path | None) -> FencedBackgroundClient:
        client = FencedBackgroundClient.__new__(FencedBackgroundClient)
        client.channel = _TimedOutChannel(marker)
        client.enabled = True
        client.records = []
        client.faults = []
        return client

    def test_matching_gpu_fence_allows_one_retirement_and_disables_ai(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "marker.json"
            client = self._client(marker)
            now = time.perf_counter_ns()
            result = client.run_joint(3, now + 15_000_000, 15,
                                      "lease3", marker,
                                      now + 20_000_000, 2_000_000)
            self.assertTrue(result)
            self.assertFalse(client.enabled)
            self.assertEqual(len(client.records), 1)
            self.assertTrue(client.records[0]["fault_fence_confirmed"])
            self.assertEqual(client.faults[0]["type"], "RPCResponseTimeoutFenced")

    def test_missing_fence_retains_lease(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            marker = Path(tmp) / "absent.json"
            client = self._client(None)
            now = time.perf_counter_ns()
            result = client.run_joint(3, now + 3_000_000, 3,
                                      "lease3", marker,
                                      now + 10_000_000, 1_000_000)
            self.assertFalse(result)
            self.assertFalse(client.enabled)
            self.assertEqual(client.records, [])
            self.assertEqual(client.faults[0]["type"], "UnfencedJointAI")


if __name__ == "__main__":
    unittest.main()
