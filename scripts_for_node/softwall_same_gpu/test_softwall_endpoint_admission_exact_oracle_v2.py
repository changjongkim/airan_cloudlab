#!/usr/bin/env python3

import unittest

from softwall_endpoint_admission_exact_oracle_v2 import (
    build_jobs,
    exact_maximum_admission,
    slot_greedy_count,
)


class EndpointAdmissionExactOracleV2Test(unittest.TestCase):
    def test_oracle_accepts_current_four_cell_mode(self):
        jobs = build_jobs([4], 155, 2, 12)
        result = exact_maximum_admission(jobs, [45, 45])
        self.assertEqual(result["admitted"], 4)

    def test_oracle_rejects_all_when_first_slot_misses_every_deadline(self):
        jobs = build_jobs([3], 50, 2, 10)
        result = exact_maximum_admission(jobs, [100])
        self.assertEqual(result["admitted"], 0)

    def test_oracle_can_reject_early_job_and_accept_later_job(self):
        jobs = build_jobs([4], 105, 2, 20)
        result = exact_maximum_admission(jobs, [60])
        self.assertEqual(result["admitted"], 1)
        self.assertEqual(slot_greedy_count(jobs, [60]), 1)

    def test_multiple_endpoint_speeds_match_exact_cardinality(self):
        jobs = build_jobs([2, 4], 120, 2, 15)
        exact = exact_maximum_admission(jobs, [20, 45, 60])
        self.assertEqual(slot_greedy_count(jobs, [20, 45, 60]),
                         exact["admitted"])

    def test_one_slot_physical_ring_caps_each_endpoint(self):
        jobs = build_jobs([4], 155, 2, 25)
        exact = exact_maximum_admission(jobs, [45, 45], [1, 1])
        self.assertEqual(exact["admitted"], 2)
        self.assertEqual(
            slot_greedy_count(jobs, [45, 45], [1, 1]), 2
        )


if __name__ == "__main__":
    unittest.main()
