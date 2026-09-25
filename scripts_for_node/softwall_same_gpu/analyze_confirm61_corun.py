#!/usr/bin/env python3.11
"""Frozen paired audit of early conventional with another NRx outstanding."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm61_corun_protocol.json")
    gates: dict[str, bool] = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / path).read_bytes()).hexdigest() == digest
        for path, digest in protocol["source_sha256_before_run"].items()
    )
    arms = {}
    for mode in ("off", "on"):
        prefix = f"confirm61_{mode}_job{args.job}"
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
            and ran["gate_mode"] == "threshold"
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["early_mandatory"] == mode
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
        gates[f"{mode}_observed_timing"] = all(
            row["commit_return_ns"] >= row["completed_ns"]
            and row["commit_response_ms"] <= protocol["deadline_ms"]
            and (row["conventional_host_path_ms"] is None or
                 row["conventional_host_path_ms"] <= protocol["conv_path_bound_ms"])
            for row in rows
        )
    off, on = arms["off"], arms["on"]
    gates["paired_trace"] = all(
        a["index"] == b["index"]
        and a["cell"] == b["cell"]
        and a["channel_seed"] == b["channel_seed"]
        and a["gate_skipped"] == b["gate_skipped"]
        for a, b in zip(off["records"], on["records"])
    )
    gates["early_host_overlap_observed"] = (
        off["host_overlapped_early_mandatory"] == 0
        and on["host_overlapped_early_mandatory"] >= protocol["min_host_overlap_count"]
    )
    gates["radio_noninferiority"] = (
        on["correct_cells"] - off["correct_cells"]
        >= protocol["min_correct_cell_difference"]
    )
    report = {
        "schema": "softwall-confirm61-corun-v1",
        "job": args.job,
        "gates": gates,
        "all_pass": all(gates.values()),
        "comparison": {
            "off_correct": off["correct_cells"],
            "on_correct": on["correct_cells"],
            "off_ai_units": off["background_units"],
            "on_ai_units": on["background_units"],
            "ai_difference_percent": 100 * (on["background_units"] - off["background_units"]) / off["background_units"],
            "off_fallbacks": off["fallbacks"],
            "on_fallbacks": on["fallbacks"],
            "on_host_overlapped_early_mandatory": on["host_overlapped_early_mandatory"],
            "off_response_p99_ms": off["commit_response_ms"]["p99"],
            "on_response_p99_ms": on["commit_response_ms"]["p99"],
        },
        "interpretation": "Physical same-GPU MPS full-path sample and host outstanding overlap only. Host overlap is not GPU kernel overlap, and observed maximum below 25 ms is not WCET. Synthetic P150/D130, fixed routing, and AI after radio do not validate the joint scheme.",
    }
    output = base / f"confirm61_corun_job{args.job}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"], "failed": [k for k, v in gates.items() if not v], "comparison": report["comparison"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
