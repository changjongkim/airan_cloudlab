#!/usr/bin/env python3.11

import dataclasses
import unittest

from c164_process_replacement_model_v1 import (
    ReplacementEvidence, qualify_replacement, safety_invariants,
)


def complete(stage="launched"):
    return ReplacementEvidence(stage, True, True, True, True, True, True, True)


class ProcessReplacementModelTests(unittest.TestCase):
    def test_complete_launched_and_prepared_evidence_pass(self):
        for stage, terminal, launches in (
            ("launched", "quiescence_fenced", 1),
            ("prepared", "nonlaunch_fenced", 0),
        ):
            evidence = complete(stage)
            decision = qualify_replacement(evidence)
            self.assertTrue(decision.accepted)
            self.assertEqual(decision.terminal_stage, terminal)
            self.assertEqual(decision.physical_launch_count, launches)
            self.assertTrue(all(safety_invariants(evidence, decision).values()))

    def test_each_missing_proof_fails_closed(self):
        fields = (
            "identity_match", "target_present_before",
            "terminate_client_success", "process_exit_confirmed",
            "target_absent_after", "same_mps_server_survived",
            "mandatory_client_preserved",
        )
        for field in fields:
            evidence = dataclasses.replace(complete(), **{field: False})
            decision = qualify_replacement(evidence)
            self.assertFalse(decision.accepted, field)
            self.assertIsNone(decision.terminal_stage)

    def test_process_exit_without_mps_termination_is_not_quiescence(self):
        evidence = dataclasses.replace(
            complete(), terminate_client_success=False,
            process_exit_confirmed=True, target_absent_after=True,
        )
        self.assertFalse(qualify_replacement(evidence).accepted)

    def test_invalid_stage_rejected(self):
        with self.assertRaises(ValueError):
            complete("fenced")


if __name__ == "__main__":
    unittest.main()
