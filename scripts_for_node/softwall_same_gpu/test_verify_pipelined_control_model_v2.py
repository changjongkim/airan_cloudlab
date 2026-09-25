#!/usr/bin/env python3

import unittest

from verify_pipelined_control_model_v2 import verify


class PipelinedControlModelV2Test(unittest.TestCase):
    def test_fault_and_single_token_models_pass(self):
        result = verify()
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["fault_protocol_model"]["violations"], [])
        self.assertEqual(
            result["single_token_ownership_model"]["violations"], []
        )


if __name__ == "__main__":
    unittest.main()
