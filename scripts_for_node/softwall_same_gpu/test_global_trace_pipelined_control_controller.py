#!/usr/bin/env python3

import os
import unittest

os.environ.setdefault("SOFTWALL_BROKER_RPC_ADMISSION_BOUND_MS", "7")

import global_trace_pipelined_control_controller as wrapper


class PipelinedControlControllerTest(unittest.TestCase):
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

    def test_deferred_complete_is_not_mislabeled_as_acknowledged(self):
        class DeferredQueue:
            completion_ack_deferred = True

            def complete(self, request, returned_ns):
                return True

        self.assertFalse(wrapper.controller.complete_selected(
            DeferredQueue(), {"request_id": "r0"}, 10
        ))


if __name__ == "__main__":
    unittest.main()
