#!/usr/bin/env python3.11
"""Frozen audit of the combined channel/queue/early-mandatory baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm62_combined_protocol.json")
    gates: dict[str, bool] = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms = {}
    for mode in ("always", "low_threshold"):
        prefix = f"confirm62_{mode}_job{args.job}"
        ran = read(base / "raw" / f"{prefix}_controller.json")
        workers = [read(base / "raw" / f"{prefix}_worker{cell}.json") for cell in range(2)]
        background = read(base / "raw" / f"{prefix}_background.json")
        requal = read(base / "raw" / f"{prefix}_requalification.json")
        rows = ran["records"]
        arms[mode] = ran
        gates[f"{mode}_contract"] = (
            ran["iterations"] == protocol["iterations"]
            and ran["period_ms"] == protocol["period_ms"]
            and ran["deadline_ms"] == protocol["deadline_ms"]
            and ran["payload_seed"] == protocol["payload_seed"]
            and ran["channel_seed_base"] == protocol["channel_seed_base"]
            and ran["gate_mode"] == mode
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["early_mandatory"] == "on"
            and ran["noise_reference"] == "pre_fading"
            and len(rows) == 2 * protocol["iterations"]
            and requal["iterations"] == 200
            and requal["deadline_misses"] == 0
        )
        gates[f"{mode}_provenance"] = (
            len({item["host"] for item in (ran, *workers, background, requal)}) == 1
            and all(str(item["slurm_job_id"]) == args.job
                    for item in (ran, *workers, background, requal))
            and all(40 <= worker["visible_sm_count"] <= 44 for worker in workers)
            and int(background["mps_active_thread_percentage"]) == 20
        )
        gates[f"{mode}_safety"] = (
            ran["deadline_misses"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_bound_violations"] == 0
            and ran["conv_path_bound_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and not ran["background_faults"]
            and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and all(item["outstanding"] == 0 for item in ran["endpoint_final"].values())
            and ran["background_units"] == background["completed_units"]
            and all(row["commit_kind"] in ("nrx", "conventional") for row in rows)
        )
        gates[f"{mode}_online_feature"] = all(
            row["index"] == index // 2
            and row["cell"] == index % 2
            and row["channel_seed"] == ran["channel_seed_base"] + index
            and row["feature_begin_ns"] >= row["release_ns"]
            and row["feature_return_ns"] >= row["feature_begin_ns"]
            and row["gate_skipped"] == (
                mode == "low_threshold" and
                row["channel_estimate_power"] < protocol["gate_threshold"]
            )
            for index, row in enumerate(rows)
        )
        gates[f"{mode}_timing"] = all(
            row["commit_return_ns"] >= row["completed_ns"]
            and row["commit_response_ms"] <= protocol["deadline_ms"]
            and (row["conventional_host_path_ms"] is None or
                 row["conventional_host_path_ms"] <= protocol["conv_path_bound_ms"])
            for row in rows
        )
    always, gated = arms["always"], arms["low_threshold"]
    gates["paired_trace"] = all(
        a["index"] == b["index"]
        and a["cell"] == b["cell"]
        and a["channel_seed"] == b["channel_seed"]
        for a, b in zip(always["records"], gated["records"])
    )
    cross_routed = sum(
        row["admitted"] and row["endpoint_id"] != f"nrx{row['cell']}"
        for row in gated["records"]
    )
    gates["cross_routing"] = cross_routed >= protocol["min_cross_routed_cells"]
    gates["early_host_overlap"] = (
        gated["host_overlapped_early_mandatory"]
        >= protocol["min_host_overlap_count"]
    )
    gates["radio_noninferiority"] = (
        gated["correct_cells"] - always["correct_cells"]
        >= protocol["min_correct_cell_difference"]
    )
    ai_gain = 100 * (gated["background_units"] - always["background_units"]) / always["background_units"]
    gates["ai_gain"] = ai_gain >= protocol["min_ai_gain_percent"]
    report = {
        "schema": "softwall-confirm62-combined-baseline-v1",
        "job": args.job,
        "gates": gates,
        "all_pass": all(gates.values()),
        "comparison": {
            "always_correct": always["correct_cells"],
            "low_gate_correct": gated["correct_cells"],
            "always_ai_units": always["background_units"],
            "low_gate_ai_units": gated["background_units"],
            "ai_gain_percent": ai_gain,
            "low_gate_skips": gated["gate_skips"],
            "cross_routed_cells": cross_routed,
            "low_gate_host_overlapped_early_mandatory": gated["host_overlapped_early_mandatory"],
        },
        "interpretation": "Combined component integration on synthetic P150/D130 with idle endpoint queues and AI only after radio. This is not the AI-aware greedy-replan strong baseline, the joint policy, GPU-kernel overlap proof, WCET or production MAC expiry.",
    }
    output = base / f"confirm62_combined_job{args.job}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"], "failed": [name for name, passed in gates.items() if not passed], "comparison": report["comparison"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
