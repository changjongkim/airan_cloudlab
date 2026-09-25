#!/usr/bin/env python3.11

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from analyze_confirm156_timeline import (
    analyze_phases,
    overlap_ns,
    profile_activity,
)


class Confirm156TimelineAnalyzerTest(unittest.TestCase):
    def make_profile(self, path: Path, gap_kernel: bool = False) -> int:
        base = 1_000_000
        with sqlite3.connect(path) as database:
            database.execute(
                "CREATE TABLE TARGET_INFO_SESSION_START_TIME(systemClockNs INTEGER)"
            )
            database.execute(
                "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (?)", (base,)
            )
            database.execute("CREATE TABLE StringIds(id INTEGER, value TEXT)")
            database.execute("INSERT INTO StringIds VALUES (1, 'kernel')")
            database.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
                "start INTEGER, end INTEGER, deviceId INTEGER, shortName INTEGER)"
            )
            database.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (?,?,2,1)",
                [(210, 220), (310, 340)] + ([(250, 260)] if gap_kernel else []),
            )
            database.execute(
                "CREATE TABLE CUPTI_ACTIVITY_KIND_MEMCPY("
                "start INTEGER, end INTEGER, deviceId INTEGER, bytes INTEGER, "
                "srcDeviceId INTEGER, dstDeviceId INTEGER)"
            )
            database.executemany(
                "INSERT INTO CUPTI_ACTIVITY_KIND_MEMCPY VALUES (?,?,2,4096,0,2)",
                [(110, 120), (410, 420)],
            )
            database.execute(
                "CREATE TABLE NVTX_EVENTS("
                "start INTEGER, end INTEGER, text TEXT, textId INTEGER)"
            )
            database.executemany(
                "INSERT INTO NVTX_EVENTS VALUES (?,?,?,NULL)",
                [
                    (100, 150, "c156:home0:r0:forward_p2p"),
                    (200, 250, "c156:home0:r0:install"),
                    (300, 350, "c156:home0:r0:conventional"),
                    (400, 450, "c156:home0:r0:backward_p2p"),
                ],
            )
        return base

    @staticmethod
    def recovery(base: int) -> dict:
        return {
            "key": ["home0", "r0"],
            "actual_start_ns": base + 100,
            "actual_completed_ns": base + 450,
            "phase_windows": {
                "forward_p2p": {"start_ns": base + 100, "end_ns": base + 150},
                "install": {"start_ns": base + 200, "end_ns": base + 250},
                "conventional": {"start_ns": base + 300, "end_ns": base + 350},
                "backward_p2p": {"start_ns": base + 400, "end_ns": base + 450},
            },
        }

    def test_profile_activity_and_phase_causality(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            base = self.make_profile(path)
            observed_base, events, nvtx = profile_activity(path)
            self.assertEqual(observed_base, base)
            audit = analyze_phases([self.recovery(base)], events, nvtx)
            self.assertFalse(audit["uncovered"])
            row = audit["rows"][0]
            self.assertTrue(row["host_phase_order"])
            self.assertTrue(row["gpu_phase_order"])
            self.assertTrue(all(row["phase_nonempty"].values()))
            self.assertTrue(row["nvtx_complete"])

    def test_unattributed_activity_and_overlap_are_detected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trace.sqlite"
            base = self.make_profile(path, gap_kernel=True)
            _, events, nvtx = profile_activity(path)
            audit = analyze_phases([self.recovery(base)], events, nvtx)
            self.assertEqual(len(audit["uncovered"]), 1)
        left = [{"start_ns": 10, "end_ns": 30}]
        right = [{"start_ns": 20, "end_ns": 40}]
        self.assertEqual(overlap_ns(left, right), 10)

    def test_utc_epoch_profile_is_mapped_to_perf_counter(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "utc.sqlite"
            with sqlite3.connect(path) as database:
                database.execute(
                    "CREATE TABLE TARGET_INFO_SESSION_START_TIME(utcEpochNs INTEGER)"
                )
                database.execute(
                    "INSERT INTO TARGET_INFO_SESSION_START_TIME VALUES (10000000)"
                )
                database.execute(
                    "CREATE TABLE CUPTI_ACTIVITY_KIND_KERNEL("
                    "start INTEGER, end INTEGER, deviceId INTEGER)"
                )
                database.execute(
                    "INSERT INTO CUPTI_ACTIVITY_KIND_KERNEL VALUES (110,120,2)"
                )
            base, events, _ = profile_activity(path, {
                "perf_counter_ns": 1_000_000,
                "utc_epoch_ns": 9_999_000,
                "uncertainty_ns": 10,
            })
            self.assertEqual(base, 10_000_000)
            self.assertEqual(events[0]["start_ns"], 1_001_110)
            self.assertEqual(events[0]["end_ns"], 1_001_120)


if __name__ == "__main__":
    unittest.main()
