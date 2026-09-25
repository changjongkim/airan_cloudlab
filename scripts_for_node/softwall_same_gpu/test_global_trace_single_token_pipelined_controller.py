#!/usr/bin/env python3

import os
import unittest

os.environ.setdefault("SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS", "7")

import global_trace_single_token_pipelined_controller as wrapper


class SingleTokenPipelinedControllerTest(unittest.TestCase):
    def test_charges_only_launch_commit(self):
        guard, control = wrapper.pipelined_composed_ai_guard_ns(
            2_000_000, 5.0, True
        )
        self.assertEqual(control, round(wrapper.RPC_BOUND_MS * 1e6))
        self.assertEqual(guard, 2_000_000 + control)

    def test_no_broker_has_no_control_charge(self):
        guard, control = wrapper.pipelined_composed_ai_guard_ns(
            2_000_000, 5.0, False
        )
        self.assertEqual((guard, control), (2_000_000, 0))

    def test_queue_status_exposes_ownership_contract(self):
        class Queue:
            enabled = True
            faults = []
            rpc_timeout_ms = 5
            ownership_contract = "one-token"

            def rpc_telemetry(self):
                return {"ownership_evidence": {"maximum_unlaunched_tokens": 1}}

        status = wrapper.pipelined_global_queue_status(Queue())
        self.assertEqual(status["ownership_contract"], "one-token")
        self.assertEqual(
            status["control_plane"],
            "single-token-pipelined-prepare-and-complete",
        )


if __name__ == "__main__":
    unittest.main()
