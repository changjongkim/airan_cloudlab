#!/usr/bin/env python3.11

import copy
import json
import tempfile
import unittest
from pathlib import Path

from analyze_c164_process_replacement import evaluate


class ProcessReplacementAnalyzerTests(unittest.TestCase):
    def fixtures(self, root: Path):
        episodes = []
        for index, (fault, context) in enumerate((
            ("drop_after_prepare", 128), ("drop_after_launch", 128),
            ("drop_after_prepare", 512), ("drop_after_launch", 512),
        )):
            spec = {"episode": index + 1, "fault_mode": fault,
                    "context_length": context, "token": f"t{index}",
                    "old_worker_epoch": f"w{index}",
                    "new_worker_epoch": f"w{index+1}"}
            launched = fault == "drop_after_launch"
            values = {
                "ready": {"host": "node", "slurm_job_id": "job",
                          "worker_epoch": f"w{index}",
                          "predecessor_journal_valid": index > 0},
                "client": {"channel_loss_observed": True},
                "journal": {"identity": {"token": f"t{index}"},
                            "stage": "quiescence_fenced" if launched else "nonlaunch_fenced",
                            "physical_launch_count": int(launched),
                            "quiescence_fence_count": 1,
                            "recovered_by_worker_epoch": f"w{index+1}"},
                "before": {},
                "termination": {"all_pass": True, "result_code": 0,
                                "started_ns": 0, "returned_ns": 1,
                                "target": {"pid": 100 + index, "server_pid": 9}},
                "exit": {"all_pass": True},
                "after": {"target_match": []},
                "certificate": {"all_pass": True,
                    "evidence": {"same_mps_server_survived": True,
                                 "mandatory_client_preserved": True},
                    "journal_before": {"physical_submission_observed": launched}},
            }
            for name, value in values.items():
                path = root / f"e{index}_{name}.json"
                path.write_text(json.dumps(value))
                spec[name] = str(path)
            episodes.append(spec)
        ready_path = root / "final_ready.json"
        output_path = root / "final_output.json"
        ready_path.write_text(json.dumps({"host": "node", "worker_epoch": "w4",
                                          "predecessor_journal_valid": True}))
        output_path.write_text(json.dumps({"worker_epoch": "w4", "clean_stop": True}))
        protocol = {"campaign": "development", "label": "x", "seed": 1,
                    "excluded_nodes": [], "source_sha256": {"x": "h"},
                    "episodes": episodes,
                    "mode": {"expected_per_pair": 1},
                    "final_worker": {"epoch": "w4", "ready": str(ready_path),
                                     "output": str(output_path)}}
        mandatory = {"host": "node", "slurm_job_id": "job",
                     "completed_iterations": 30, "stop_observed": True,
                     "error": None, "deadline_misses": 0,
                     "component_bound_violations": 0,
                     "records": [{"correct": True, "response_ms": 1,
                                  "cell_results": [{"gpu_ms": 1} for _ in range(4)]}
                                 for _ in range(30)]}
        inventory = [{"name": "NVIDIA A100", "mig.mode.current": "Disabled"}
                     for _ in range(4)]
        return protocol, mandatory, inventory, {"x": "h"}

    def test_valid_campaign_passes(self):
        with tempfile.TemporaryDirectory() as directory:
            self.assertTrue(evaluate(*self.fixtures(Path(directory)))["all_pass"])

    def test_missing_termination_or_ran_safety_fails(self):
        with tempfile.TemporaryDirectory() as directory:
            values = self.fixtures(Path(directory))
            termination = Path(values[0]["episodes"][0]["termination"])
            data = json.loads(termination.read_text()); data["result_code"] = 1
            termination.write_text(json.dumps(data))
            self.assertFalse(evaluate(*values)["all_pass"])
        with tempfile.TemporaryDirectory() as directory:
            values = self.fixtures(Path(directory)); values[1]["deadline_misses"] = 1
            self.assertFalse(evaluate(*values)["all_pass"])


if __name__ == "__main__":
    unittest.main()
