#!/usr/bin/env python3

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from instrumented_deadline_global_trace_client import (
    InstrumentedDeadlineGlobalTraceClient,
)


class InstrumentedDeadlineGlobalTraceClientTest(unittest.TestCase):
    def test_timeout_is_recorded_before_quarantine(self):
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
            client = InstrumentedDeadlineGlobalTraceClient(
                path, 0, 1, "sealed", rpc_timeout_ms=5
            )
            request = {"request_id": "r0", "broker_token": "t0"}
            self.assertFalse(client.commit(request))
            telemetry = client.rpc_telemetry()
            self.assertEqual(len(telemetry["records"]), 1)
            self.assertEqual(telemetry["records"][0]["operation"], "commit")
            self.assertTrue(telemetry["records"][0]["faulted"])
            self.assertEqual(telemetry["by_operation"]["commit"]["count"], 1)
            client.close()
            thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
