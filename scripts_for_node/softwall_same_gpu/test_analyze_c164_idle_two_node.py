#!/usr/bin/env python3.11

import copy
import unittest

from analyze_c164_idle_two_node import combine


def fixtures():
    gates = {"g": True}
    base = {
        "all_pass": True, "campaign": "development", "host": "node-a",
        "slurm_job_id": "1", "observed_idle_s": 30.1, "gates": gates,
        "counts": {"actual_nrx": 4, "injected_recoveries": 2,
                   "qwen_units": 1, "radio_commits": 4,
                   "deadline_misses": 0},
        "maxima_ms": {"nrx_release_to_complete": 10,
                      "qwen_execution": 20, "radio_commit": 100,
                      "recovery_path": 5},
    }
    hold = copy.deepcopy(base)
    hold.update({"campaign": "holdout", "host": "node-b",
                 "slurm_job_id": "2", "observed_idle_s": 30.2})
    sources = {"a": "x"}
    protocol = {
        "iterations": 6, "seed_base": 1, "reverse_cases": False,
        "excluded_nodes": [], "source_sha256": sources,
        "c164": {"lifecycle": "idle_30s_first", "required_idle_s": 30.0,
                 "source_sha256": sources},
    }
    hold_protocol = copy.deepcopy(protocol)
    hold_protocol.update({"iterations": 12, "seed_base": 2,
                          "reverse_cases": True,
                          "excluded_nodes": ["node-a"]})
    return base, hold, protocol, hold_protocol


class CombineC164IdleTest(unittest.TestCase):
    def test_valid_independent_pair_passes(self):
        result = combine(*fixtures())
        self.assertTrue(result["all_pass"])
        self.assertEqual(result["counts"]["actual_nrx"], 8)

    def test_same_node_fails(self):
        values = list(fixtures())
        values[1]["host"] = "node-a"
        result = combine(*values)
        self.assertFalse(result["gates"]["independent_job_and_node"])

    def test_source_change_fails(self):
        values = list(fixtures())
        values[3]["c164"]["source_sha256"] = {"a": "changed"}
        result = combine(*values)
        self.assertFalse(result["gates"]["frozen_c164_source_equality"])

    def test_holdout_must_exclude_development_node(self):
        values = list(fixtures())
        values[3]["excluded_nodes"] = []
        result = combine(*values)
        self.assertFalse(result["gates"]["development_node_excluded_from_holdout"])


if __name__ == "__main__":
    unittest.main()
