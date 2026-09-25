#!/usr/bin/env python3.11

import copy
import unittest

from analyze_c164_qwen_reload import evaluate, overlaps


def record(index, release, complete, response=10.0):
    return {
        "index": index, "release_ns": release, "completed_ns": complete,
        "response_ms": response, "deadline_miss": False, "correct": True,
        "cell_results": [{"gpu_ms": 2.0, "component_bound_violation": False,
                          "correct": True} for _ in range(4)],
    }


def episode(index, launch, ready):
    return {
        "episode": index, "host": "nid-a", "slurm_job_id": "1",
        "clock": "time.perf_counter_ns", "launch_ns": launch,
        "ready_ns": ready, "exit_ns": ready + 10,
        "load_and_warmup_ms": (ready - launch) / 1e6,
        "all_pass": True, "qwen": {"completed_units": 0},
    }


class ReloadAnalysisTests(unittest.TestCase):
    def fixtures(self):
        records = [record(0, 100, 120), record(1, 200, 220),
                   record(2, 300, 320)]
        mandatory = {
            "host": "nid-a", "slurm_job_id": "1",
            "clock": "time.perf_counter_ns", "records": records,
            "completed_iterations": 3, "correct_releases": 3,
            "deadline_misses": 0, "component_bound_violations": 0,
            "stop_observed": True, "error": None,
        }
        episodes = [episode(1, 190, 230)]
        protocol = {
            "campaign": "development", "label": "x", "episodes": 1,
            "excluded_nodes": [], "source_sha256": {"x": "a"},
        }
        inventory = [{"name": "NVIDIA A100", "mig.mode.current": "Disabled"}
                     for _ in range(4)]
        return protocol, mandatory, episodes, inventory, {"x": "a"}

    def test_interval_overlap(self):
        self.assertTrue(overlaps(record(1, 200, 220), episode(1, 190, 230)))
        self.assertFalse(overlaps(record(0, 100, 120), episode(1, 190, 230)))

    def test_valid_result_passes(self):
        protocol, mandatory, episodes, inventory, hashes = self.fixtures()
        mandatory["minimum_test_padding"] = True
        # The production gate requires 30 releases; repeat the three-record
        # pattern while retaining both overlap and quiet samples.
        mandatory["records"] = [record(i, 100 * (i + 1), 100 * (i + 1) + 20)
                                for i in range(30)]
        mandatory["completed_iterations"] = 30
        mandatory["correct_releases"] = 30
        episodes[0]["launch_ns"] = 190
        episodes[0]["ready_ns"] = 230
        self.assertTrue(evaluate(protocol, mandatory, episodes,
                                 inventory, hashes)["all_pass"])

    def test_deadline_miss_fails(self):
        protocol, mandatory, episodes, inventory, hashes = self.fixtures()
        mandatory["records"] *= 10
        for i, row in enumerate(mandatory["records"]): row["index"] = i
        mandatory["completed_iterations"] = 30
        mandatory["correct_releases"] = 30
        mandatory["deadline_misses"] = 1
        mandatory["records"][0]["deadline_miss"] = True
        self.assertFalse(evaluate(protocol, mandatory, episodes,
                                  inventory, hashes)["all_pass"])

    def test_optional_inference_fails(self):
        protocol, mandatory, episodes, inventory, hashes = self.fixtures()
        mandatory["records"] *= 10
        for i, row in enumerate(mandatory["records"]): row["index"] = i
        mandatory["completed_iterations"] = 30
        mandatory["correct_releases"] = 30
        episodes = copy.deepcopy(episodes)
        episodes[0]["qwen"]["completed_units"] = 1
        self.assertFalse(evaluate(protocol, mandatory, episodes,
                                  inventory, hashes)["all_pass"])

    def test_missing_overlap_fails(self):
        protocol, mandatory, episodes, inventory, hashes = self.fixtures()
        mandatory["records"] *= 10
        for i, row in enumerate(mandatory["records"]): row["index"] = i
        mandatory["completed_iterations"] = 30
        mandatory["correct_releases"] = 30
        episodes[0]["launch_ns"] = 10_000
        episodes[0]["ready_ns"] = 11_000
        self.assertFalse(evaluate(protocol, mandatory, episodes,
                                  inventory, hashes)["all_pass"])


if __name__ == "__main__":
    unittest.main()
