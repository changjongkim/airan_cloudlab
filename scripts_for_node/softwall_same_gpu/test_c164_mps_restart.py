#!/usr/bin/env python3.11

import copy
import unittest

from analyze_c164_restart import evaluate
from build_c164_mps_restart_marker import build


def snapshot(phase, ns, control_pid, server_pid, inode):
    return {
        "schema": "softwall-c164-mps-epoch-snapshot-v1",
        "phase": phase, "host": "nid-a", "slurm_job_id": "7",
        "clock": "time.perf_counter_ns", "snapshot_ns": ns,
        "control_pids": [control_pid], "server_pids": [server_pid],
        "control_socket": {"inode": inode}, "all_pass": True,
    }


class RestartMarkerTests(unittest.TestCase):
    def test_distinct_epoch_passes(self):
        value = build(snapshot("before_restart", 10, 1, 2, 3),
                      snapshot("after_restart", 40, 4, 5, 6),
                      stop_started_ns=20, stop_completed_ns=30,
                      control_absent=True, server_absent=True)
        self.assertTrue(value["all_pass"])

    def test_reused_daemon_identity_fails(self):
        value = build(snapshot("before_restart", 10, 1, 2, 3),
                      snapshot("after_restart", 40, 1, 2, 3),
                      stop_started_ns=20, stop_completed_ns=30,
                      control_absent=True, server_absent=True)
        self.assertFalse(value["all_pass"])

    def test_no_quiescent_gap_fails(self):
        value = build(snapshot("before_restart", 10, 1, 2, 3),
                      snapshot("after_restart", 40, 4, 5, 6),
                      stop_started_ns=20, stop_completed_ns=30,
                      control_absent=False, server_absent=True)
        self.assertFalse(value["all_pass"])


class PhysicalAnalyzerTests(unittest.TestCase):
    def fixtures(self):
        hashes = {"scripts/x": "a" * 64}
        protocol = {"c164_restart": {
            "schema": "softwall-c164-mps-restart-protocol-v1",
            "lifecycle": "mps_restart_first",
            "availability_scope": "optional closed; uninterrupted service is not claimed",
            "source_sha256": hashes,
        }}
        marker = build(snapshot("before_restart", 10, 1, 2, 3),
                       snapshot("after_restart", 40, 4, 5, 6),
                       stop_started_ns=20, stop_completed_ns=30,
                       control_absent=True, server_absent=True)
        marker["created_ns"] = 50
        base = {"all_pass": True, "campaign": "development", "label": "x",
                "counts": {"deadline_misses": 0}, "maxima_ms": {},
                "gates": {"single_commit_d155_semantics": True}}
        coordinator = {"host": "nid-a", "slurm_job_id": "7", "rounds": [{
            "release_wall_ns": 100, "physical_timely_successes": [1, 2, 3, 4],
            "missing_outcomes_at_cutoff": [], "certificate_order": [1],
            "boundary_case": "E1",
        }]}
        endpoint = {"host": "nid-a", "slurm_job_id": "7"}
        return protocol, base, marker, coordinator, endpoint, hashes

    def test_valid_physical_result_passes(self):
        protocol, base, marker, coordinator, endpoint, hashes = self.fixtures()
        value = evaluate(protocol, base, marker, coordinator, endpoint, endpoint, hashes)
        self.assertTrue(value["all_pass"])

    def test_restart_after_first_release_fails(self):
        protocol, base, marker, coordinator, endpoint, hashes = self.fixtures()
        marker = copy.deepcopy(marker)
        marker["after"]["snapshot_ns"] = 101
        value = evaluate(protocol, base, marker, coordinator, endpoint, endpoint, hashes)
        self.assertFalse(value["all_pass"])

    def test_source_change_fails(self):
        protocol, base, marker, coordinator, endpoint, hashes = self.fixtures()
        value = evaluate(protocol, base, marker, coordinator, endpoint, endpoint,
                         {"scripts/x": "b" * 64})
        self.assertFalse(value["all_pass"])


if __name__ == "__main__":
    unittest.main()
