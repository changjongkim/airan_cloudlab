#!/usr/bin/env python3.11

import copy
import unittest

from analyze_c164_restart_two_node import combine


def result(role, node, job):
    return {
        "all_pass": True, "campaign": role, "host": node,
        "slurm_job_id": job,
        "counts": {"actual_nrx": 4, "injected_recoveries": 2,
                   "qwen_units": 1, "radio_commits": 4,
                   "deadline_misses": 0},
        "maxima_ms": {"nrx_release_to_complete": 10,
                      "recovery_path": 5, "qwen_execution": 20,
                      "radio_commit": 100},
        "gates": {"physical_restart_marker_pass": True, "other": True},
        "restart_identity": {},
    }


def protocol(seed, reverse, excluded):
    return {
        "iterations": 6, "seed_base": seed, "reverse_cases": reverse,
        "excluded_nodes": excluded, "source_sha256": {"base": "a"},
        "c164_restart": {"lifecycle": "mps_restart_first",
                         "restart_scope": "q",
                         "source_sha256": {"restart": "b"}},
    }


class CombineRestartTests(unittest.TestCase):
    def fixtures(self):
        return (result("development", "nid-a", "1"),
                result("holdout", "nid-b", "2"),
                protocol(1, False, []), protocol(2, True, ["nid-a"]))

    def test_valid_pair_passes(self):
        self.assertTrue(combine(*self.fixtures())["all_pass"])

    def test_same_node_fails(self):
        dev, hold, dp, hp = self.fixtures()
        hold["host"] = dev["host"]
        self.assertFalse(combine(dev, hold, dp, hp)["all_pass"])

    def test_source_change_fails(self):
        dev, hold, dp, hp = self.fixtures()
        hp = copy.deepcopy(hp)
        hp["c164_restart"]["source_sha256"] = {"restart": "changed"}
        self.assertFalse(combine(dev, hold, dp, hp)["all_pass"])

    def test_holdout_must_exclude_development_node(self):
        dev, hold, dp, hp = self.fixtures()
        hp["excluded_nodes"] = []
        self.assertFalse(combine(dev, hold, dp, hp)["all_pass"])


if __name__ == "__main__":
    unittest.main()
