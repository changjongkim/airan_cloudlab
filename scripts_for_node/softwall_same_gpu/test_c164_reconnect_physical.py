#!/usr/bin/env python3.11

import unittest

from analyze_c164_reconnect import evaluate


class ReconnectPhysicalAnalyzerTests(unittest.TestCase):
    def fixtures(self):
        protocol = {
            "tokens": 4, "campaign": "development", "label": "x",
            "worker_epoch": "w", "lifecycle_epoch": 7,
            "excluded_nodes": [], "source_sha256": {"x": "h"},
            "mode": {"expected_per_fault": 2, "expected_per_context_fault": 1,
                     "ai_bounds_ms": {"128": 40, "512": 75}},
        }
        token_rows = []
        journals = []
        for index, (fault, context) in enumerate((
            ("drop_after_prepare", 128), ("drop_after_fence", 128),
            ("drop_after_prepare", 512), ("drop_after_fence", 512),
        )):
            ident = {"token": f"t{index}", "lifecycle_epoch": 7,
                     "payload_sha256": "a" * 64, "worker_epoch": "w"}
            launched = fault == "drop_after_fence"
            record = {"identity": ident,
                      "stage": "fenced" if launched else "nonlaunch_fenced",
                      "journal_seq": 3 if launched else 2,
                      "physical_launch_count": int(launched),
                      "physical_fence_count": 1}
            if launched:
                record.update({"gpu_ms": 1.0, "launch_called_ns": 10,
                               "submitted_ns": 12,
                               "completed_ns": 20})
            token_rows.append({"fault_mode": fault, "context_length": context,
                               "identity": ident, "channel_loss_observed": True,
                               "resolution": "retire_terminal", "record": record,
                               "aligned_release_ns": 15 if launched else None,
                               "duplicate_reconcile_stable": True})
            journals.append(record)
        mandatory = {"host": "n", "slurm_job_id": "j",
                     "completed_iterations": 30, "stop_observed": True,
                     "error": None, "deadline_misses": 0,
                     "component_bound_violations": 0,
                     "records": [{"started_ns": 0, "completed_ns": 30,
                                  "response_ms": 1, "correct": True,
                                  "cell_results": [{"gpu_ms": 1} for _ in range(4)]}
                                 for _ in range(30)]}
        worker = {"host": "n", "slurm_job_id": "j", "worker_epoch": "w",
                  "journal_records": journals, "rejected": []}
        client = {"host": "n", "worker_epoch": "w", "lifecycle_epoch": 7,
                  "records": token_rows}
        inventory = [{"name": "NVIDIA A100", "mig.mode.current": "Disabled"}
                     for _ in range(4)]
        return protocol, mandatory, worker, client, inventory, {"x": "h"}

    def test_valid_result_passes(self):
        self.assertTrue(evaluate(*self.fixtures())["all_pass"])

    def test_duplicate_reconcile_or_deadline_failure_fails(self):
        values = self.fixtures()
        values[3]["records"][0]["duplicate_reconcile_stable"] = False
        self.assertFalse(evaluate(*values)["all_pass"])
        values = self.fixtures()
        values[1]["deadline_misses"] = 1
        self.assertFalse(evaluate(*values)["all_pass"])

    def test_physical_launch_in_prepare_branch_fails(self):
        values = self.fixtures()
        values[3]["records"][0]["record"]["physical_launch_count"] = 1
        self.assertFalse(evaluate(*values)["all_pass"])


if __name__ == "__main__":
    unittest.main()
