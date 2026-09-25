#!/usr/bin/env python3

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from deadline_fault_contained_global_trace_client import (
    DeadlineFaultContainedGlobalTraceClient,
)


class DeadlineGlobalTraceClientTest(unittest.TestCase):
    def test_unanswered_commit_is_bounded_and_quarantined(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "broker.sock")
            ready = threading.Event()

            def stalled_broker():
                server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server.bind(path)
                server.listen(1)
                ready.set()
                connection, _ = server.accept()
                channel = connection.makefile("rwb", buffering=0)
                json.loads(channel.readline())
                channel.write(b'{"ok":true,"result":{}}\n')
                channel.flush()
                json.loads(channel.readline())
                time.sleep(0.1)
                channel.close()
                connection.close()
                server.close()

            thread = threading.Thread(target=stalled_broker)
            thread.start()
            ready.wait(1)
            client = DeadlineFaultContainedGlobalTraceClient(
                path, 0, 1, "sealed", rpc_timeout_ms=5
            )
            request = {"request_id": "r0", "broker_token": "t0"}
            begin = time.monotonic()
            self.assertFalse(client.commit(request))
            elapsed_ms = (time.monotonic() - begin) * 1000
            self.assertLess(elapsed_ms, 50)
            self.assertFalse(client.enabled)
            self.assertEqual(client.faults[0]["operation"], "commit")
            self.assertEqual(client.faults[0]["rpc_timeout_ms"], 5)
            client.close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
