#!/usr/bin/env python3

import unittest

import global_trace_qualified_control_bound_controller as wrapper


class QualifiedControlBoundControllerTest(unittest.TestCase):
    def test_charges_bound_instead_of_socket_timeout(self):
        guard, control = wrapper.qualified_composed_ai_guard_ns(
            2_000_000, 5.0, True
        )
        self.assertEqual(control, 3 * round(wrapper.RPC_BOUND_MS * 1e6))
        self.assertEqual(guard, 2_000_000 + control)

    def test_no_broker_has_no_control_charge(self):
        guard, control = wrapper.qualified_composed_ai_guard_ns(
            2_000_000, 5.0, False
        )
        self.assertEqual((guard, control), (2_000_000, 0))


if __name__ == "__main__":
    unittest.main()
