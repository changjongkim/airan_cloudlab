#!/usr/bin/env python3

import json
import socket
import tempfile
import threading
import time
import unittest
from pathlib import Path

from instrumented_deadline_global_trace_client_v2 import (
    InstrumentedDeadlineGlobalTraceClientV2,
)


class InstrumentedDeadlineGlobalTraceClientV2Test(unittest.TestCase):
    def test_records_faulting_call_but_not_post_quarantine_noops(self):
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "broker.sock")
            ready = threading.Event()
            def stalled_broker():
                server=socket.socket(socket.AF_UNIX,socket.SOCK_STREAM);server.bind(path);server.listen(1);ready.set()
                connection,_=server.accept();channel=connection.makefile("rwb",buffering=0)
                json.loads(channel.readline());channel.write(b'{"ok":true,"result":{}}\n');channel.flush()
                json.loads(channel.readline());time.sleep(0.1);channel.close();connection.close();server.close()
            thread=threading.Thread(target=stalled_broker);thread.start();ready.wait(1)
            client=InstrumentedDeadlineGlobalTraceClientV2(path,0,1,"sealed",rpc_timeout_ms=5)
            request={"request_id":"r0","broker_token":"t0"}
            self.assertFalse(client.commit(request))
            self.assertFalse(client.commit(request))
            self.assertIsNone(client.peek_edf_fitting(0,1,0,{}))
            self.assertEqual(len(client.rpc_telemetry()["records"]),1)
            self.assertTrue(client.rpc_telemetry()["records"][0]["faulted"])
            client.close();thread.join(timeout=1)


if __name__ == "__main__":
    unittest.main()
