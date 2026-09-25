#!/usr/bin/env python3

import unittest

import analyze_softwall_envelope_v15_validation as validation


class V15ValidationHelpersTest(unittest.TestCase):
    def test_authoritative_ids_are_distinct(self):
        self.assertNotEqual(validation.V14_ID, validation.V15_ID)
        self.assertIn("single_token", validation.V15_ID)


if __name__ == "__main__":
    unittest.main()
