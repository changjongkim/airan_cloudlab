#!/usr/bin/env python3.11

import json
import tempfile
import unittest
from pathlib import Path

from c161_phase2_faults import (
    ARMS,
    audit_nrx_batch,
    audit_recovery_response,
    marker_paths,
    parse_arm_order,
    periodic_fault_due,
    wait_for_matching_marker,
)


class C161Phase2FaultTests(unittest.TestCase):
    def test_arm_order(self):
        self.assertEqual(parse_arm_order(",".join(ARMS)), ARMS)
        with self.assertRaises(ValueError):
            parse_arm_order(",".join(ARMS[:-1]))

    def test_periodic_fault_due(self):
        self.assertTrue(periodic_fault_due(1, 0, interval=6, target=10))
        self.assertFalse(periodic_fault_due(6, 1, interval=6, target=10))
        self.assertTrue(periodic_fault_due(7, 1, interval=6, target=10))
        self.assertFalse(periodic_fault_due(100, 10, interval=6, target=10))

    def test_nrx_duplicate_and_stale_do_not_mutate(self):
        audit = audit_nrx_batch(
            8,
            (("home0", "r0"), ("home1", "r0")),
            (("home0", "r0"),),
            inject=True,
        )
        self.assertEqual(audit["duplicate"]["reason"], "duplicate_transaction")
        self.assertTrue(audit["duplicate"]["state_unchanged"])
        self.assertEqual(audit["stale"]["reason"], "stale_or_future_epoch")
        self.assertTrue(audit["stale"]["state_unchanged"])

    def test_recovery_duplicate_and_stale_do_not_mutate(self):
        audit = audit_recovery_response(8, "home0/r0", inject=True)
        self.assertEqual(audit["duplicate"]["reason"], "duplicate_radio_commit")
        self.assertTrue(audit["duplicate"]["state_unchanged"])
        self.assertEqual(audit["stale"]["reason"], "stale_or_future_epoch")
        self.assertTrue(audit["stale"]["state_unchanged"])

    def test_marker_requires_matching_lease(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "marker.json"
            path.write_text(json.dumps({"lease_id": "wrong"}), encoding="utf-8")
            self.assertIsNone(wait_for_matching_marker(path, "L", 0))
            path.write_text(json.dumps({"lease_id": "L"}), encoding="utf-8")
            self.assertEqual(wait_for_matching_marker(path, "L", 2**63)["lease_id"], "L")
            self.assertEqual(wait_for_matching_marker(path, "L", 0)["lease_id"], "L")

    def test_marker_paths_are_stable_and_separate(self):
        launch, complete = marker_paths(Path("/tmp/markers"), "lease")
        self.assertNotEqual(launch, complete)
        self.assertEqual(launch.parent, complete.parent)


if __name__ == "__main__":
    unittest.main()
