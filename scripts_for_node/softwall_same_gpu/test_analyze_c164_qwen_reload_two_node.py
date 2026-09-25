#!/usr/bin/env python3.11

import copy
import unittest

from analyze_c164_qwen_reload_two_node import combine


def result(role, node, job):
    counts = {"reload_episodes": 30, "mandatory_releases": 100,
              "overlap_releases": 80, "quiet_releases": 20,
              "cell_decodes": 400, "deadline_misses": 0,
              "component_bound_violations": 0,
              "optional_inference_units": 0}
    summary = {"count": 1, "mean": 1, "p50": 1, "p99": 1, "max": 1}
    return {"all_pass": True, "campaign": role, "host": node,
            "slurm_job_id": job, "counts": counts,
            "reload_ms": summary,
            "response_ms": {"all": summary, "reload_overlap": summary,
                            "quiet": summary},
            "component_gpu_ms": {"all": summary,
                                 "reload_overlap": summary, "quiet": summary},
            "gates": {"a": True}}


def protocol(seed, excluded):
    return {"seed": seed, "excluded_nodes": excluded, "episodes": 30,
            "source_sha256": {"x": "a"}, "mode": {"x": 1},
            "contract": {"y": 2}}


class CombineQwenReloadTests(unittest.TestCase):
    def fixtures(self):
        return (result("development", "nid-a", "1"),
                result("holdout", "nid-b", "2"),
                protocol(1, []), protocol(2, ["nid-a"]))

    def test_valid_pair_passes(self):
        self.assertTrue(combine(*self.fixtures())["all_pass"])

    def test_same_node_fails(self):
        dev, hold, dp, hp = self.fixtures(); hold["host"] = dev["host"]
        self.assertFalse(combine(dev, hold, dp, hp)["all_pass"])

    def test_source_change_fails(self):
        dev, hold, dp, hp = self.fixtures(); hp = copy.deepcopy(hp)
        hp["source_sha256"] = {"x": "changed"}
        self.assertFalse(combine(dev, hold, dp, hp)["all_pass"])

    def test_safety_violation_fails(self):
        dev, hold, dp, hp = self.fixtures()
        hold["counts"]["deadline_misses"] = 1
        self.assertFalse(combine(dev, hold, dp, hp)["all_pass"])


if __name__ == "__main__":
    unittest.main()
