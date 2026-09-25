#!/usr/bin/env python3

import unittest

from build_softwall_context64_mechanism_trace import build


class Context64MechanismTraceTest(unittest.TestCase):
    def test_deterministic_shape(self):
        value = build(count=3)
        self.assertEqual(value["summary"]["offered_value_tokens"], 192)
        self.assertEqual([row["arrival_ms"] for row in value["requests"]],
                         [0.0, 45.0, 90.0])
        self.assertTrue(all(row["context_length"] == 64
                            for row in value["requests"]))


if __name__ == "__main__":
    unittest.main()
