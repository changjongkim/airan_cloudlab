#!/usr/bin/env python3.11

import copy
import unittest

from analyze_c164_reconnect_two_node import combine


def result(role, host, job):
    counts = {key: 0 for key in (
        "tokens", "prepare_loss", "post_fence_loss", "physical_qwen_launches",
        "terminal_fences", "mandatory_releases", "cell_decodes",
        "overlap_releases", "deadline_misses", "component_bound_violations",
    )}
    summary = {"max": 1.0}
    return {"all_pass": True, "campaign": role, "host": host,
            "slurm_job_id": job, "counts": counts,
            "qwen_gpu_ms": summary, "mandatory_response_ms": summary,
            "mandatory_component_gpu_ms": summary}


class CombineReconnectTests(unittest.TestCase):
    def fixtures(self):
        dev = result("development", "node-a", "1")
        holdout = result("holdout", "node-b", "2")
        dp = {"source_sha256": {"x": "h"}, "mode": {"x": 1},
              "tokens": 60, "seed": 1, "excluded_nodes": []}
        hp = {"source_sha256": {"x": "h"}, "mode": {"x": 1},
              "tokens": 60, "seed": 2, "excluded_nodes": ["node-a"]}
        return dev, holdout, dp, hp

    def test_valid_pair_passes(self):
        self.assertTrue(combine(*self.fixtures())["all_pass"])

    def test_same_node_or_source_change_fails(self):
        dev, holdout, dp, hp = self.fixtures()
        holdout["host"] = dev["host"]
        self.assertFalse(combine(dev, holdout, dp, hp)["all_pass"])
        dev, holdout, dp, hp = self.fixtures()
        hp = copy.deepcopy(hp); hp["source_sha256"] = {"x": "changed"}
        self.assertFalse(combine(dev, holdout, dp, hp)["all_pass"])


if __name__ == "__main__":
    unittest.main()
