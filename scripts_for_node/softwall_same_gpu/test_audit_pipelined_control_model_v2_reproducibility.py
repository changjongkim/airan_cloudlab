#!/usr/bin/env python3

import unittest

from audit_pipelined_control_model_v2_reproducibility import canonicalize


class CanonicalModelTest(unittest.TestCase):
    def test_list_order_does_not_change_canonical_form(self):
        left = {"states": [{"b": 2}, {"a": 1}], "violations": []}
        right = {"violations": [], "states": [{"a": 1}, {"b": 2}]}
        self.assertEqual(canonicalize(left), canonicalize(right))


if __name__ == "__main__":
    unittest.main()
