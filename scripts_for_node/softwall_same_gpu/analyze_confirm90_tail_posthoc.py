#!/usr/bin/env python3.11
"""Posthoc timestamp decomposition of failed Confirm90 NRx bound samples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def offset(ns: int | None, release_ns: int) -> float | None:
    return None if ns is None else (ns - release_ns) / 1e6


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    base = args.root.resolve() / "results/softwall_same_gpu"
    protocol = read(base / "confirm90_tight_calendar_gc_off_protocol.json")
    cases = []
    for arm in protocol["arms"]:
        prefix = arm["prefix"]
        ran = read(base / "raw" / f"{prefix}_controller.json")
        workers = [read(base / "raw" / f"{prefix}_worker{i}.json") for i in range(2)]
        for row in ran["records"]:
            response = row["nrx_response_ms"]
            if response is None or response <= protocol["nrx_bound_ms"]:
                continue
            release = row["release_ns"]
            endpoint = int(row["endpoint_id"][-1])
            execution = min(
                workers[endpoint]["execution_windows"],
                key=lambda w: abs(w["observed_ns"] - row["nrx_dispatched_ns"]),
            )
            same_release = ran["records"][4 * row["index"]:4 * row["index"] + 4]
            mandatory = [
                {
                    "cell": other["cell"],
                    "start_after_release_ms": offset(other["fallback_actual_start_ns"], release),
                    "commit_after_release_ms": offset(other["commit_return_ns"], release),
                    "path_ms": other["conventional_host_path_ms"],
                }
                for other in same_release
                if other["cell"] != row["cell"] and not other["admitted"]
                and other["fallback_actual_start_ns"] is not None
            ]
            ai_before = [
                {
                    "admitted_after_release_ms": offset(ai["admitted_ns"], release),
                    "returned_after_release_ms": offset(ai["returned_ns"], release),
                    "execution_ms": ai["execution_ms"],
                }
                for ai in ran["background_records"]
                if ai.get("release_index") == row["index"]
                and ai.get("phase") == "before_nrx_observation"
            ]
            cases.append({
                "arm": arm["name"], "release_index": row["index"],
                "cell": row["cell"], "nrx_response_ms": response,
                "feature_begin_after_release_ms": offset(row["feature_begin_ns"], release),
                "feature_return_after_release_ms": offset(row["feature_return_ns"], release),
                "nrx_dispatch_after_release_ms": offset(row["nrx_dispatched_ns"], release),
                "worker_published_after_release_ms": offset(execution["published_ns"], release),
                "worker_gpu_ms": execution["gpu_ms"],
                "controller_observed_after_release_ms": offset(row["nrx_observed_ns"], release),
                "post_publish_observation_gap_ms":
                    (row["nrx_observed_ns"] - execution["published_ns"]) / 1e6,
                "mandatory_other_cells": mandatory,
                "ai_before_observation": ai_before,
            })
    report = {
        "schema": "softwall-confirm90-tail-posthoc-v1",
        "frozen_gate_result_unchanged": True,
        "cases": cases,
        "interpretation": "Timestamp overlap only. A delayed controller observation can include AI-thread join, serial early-mandatory conventional commits, OS scheduling, and completion copy. This posthoc report does not revise the frozen bound failure or prove a sole cause.",
    }
    output = base / "confirm90_tail_posthoc.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
