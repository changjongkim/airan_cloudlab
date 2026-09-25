#!/usr/bin/env python3

import tempfile
import threading
import time
import unittest
from pathlib import Path

from fault_contained_global_trace_client import FaultContainedGlobalTraceClient
from fault_inject_global_trace_lease_broker import (
    BrokerServer,
    GlobalTraceLeaseClient,
    GlobalTraceLeaseState,
)


def trace_with(count=3):
    return {
        "requests": [
            {
                "request_id": index,
                "source_order": index,
                "context_length": 16,
                "value_tokens": 16,
                "arrival_ms": 0,
                "deadline_ms": 1000,
            }
            for index in range(count)
        ]
    }


class GlobalBrokerFaultContainmentTest(unittest.TestCase):
    def test_applied_commit_with_lost_reply_is_quarantined_without_retry(self):
        epoch_ns = time.perf_counter_ns()
        state = GlobalTraceLeaseState(trace_with(), "sealed", 2, "global")
        with tempfile.TemporaryDirectory() as directory:
            socket_path = str(Path(directory) / "broker.sock")
            server = BrokerServer(
                socket_path, state,
                drop_commit_response_home=0,
                drop_commit_response_number=1,
            )
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            home0 = FaultContainedGlobalTraceClient(
                socket_path, 0, epoch_ns, "sealed"
            )
            home1 = GlobalTraceLeaseClient(socket_path, 1, epoch_ns, "sealed")
            try:
                request0 = home0.peek_edf_fitting(
                    epoch_ns + 1, epoch_ns + 500_000_000, 1000,
                    {16: 10_000_000},
                )
                self.assertFalse(home0.commit(request0))
                self.assertFalse(home0.enabled)
                self.assertEqual(len(home0.faults), 1)
                self.assertEqual(home0.faults[0]["operation"], "commit")
                self.assertEqual(state.snapshot()["states"]["inflight"], 1)

                # An ambiguous commit is never replayed locally.
                self.assertFalse(home0.commit(request0))
                self.assertEqual(server.commit_counts[0], 1)
                self.assertIsNone(home0.peek_edf_fitting(
                    epoch_ns + 2, epoch_ns + 500_000_000, 1000,
                    {16: 10_000_000},
                ))

                # The independent home and broker continue processing other
                # requests while the ambiguous token remains isolated.
                request1 = home1.peek_edf_fitting(
                    epoch_ns + 3, epoch_ns + 500_000_000, 1000,
                    {16: 10_000_000},
                )
                home1.commit(request1)
                home1.complete(request1, epoch_ns + 20_000_000)
                summary = state.snapshot()
                self.assertEqual(summary["timely_requests"], 1)
                self.assertEqual(summary["outstanding_tokens"], 1)
                self.assertEqual(summary["duplicate_commit_count"], 0)
                self.assertEqual(len(server.faults), 1)
            finally:
                home0.close()
                home1.close()
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)


if __name__ == "__main__":
    unittest.main()
