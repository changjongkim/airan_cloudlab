#!/usr/bin/env python3.11
"""Fresh-seed reverse-order audit of unresolved-NeuralRx AI admission."""

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
    protocol = read(base / "confirm64_reverse_protocol.json")
    gates: dict[str, bool] = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms = {}
    for mode in ("off", "one"):
        prefix = f"confirm64_{mode}_job{args.job}"
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
            and ran["gate_mode"] == "low_threshold"
            and ran["gate_threshold"] == protocol["gate_threshold"]
            and ran["early_mandatory"] == "on"
            and ran["ai_during_nrx"] == mode
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
            and ran["pre_radio_ai_guard_violations"] == 0
            and not ran["background_faults"]
            and not ran["endpoint_faults"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and all(item["outstanding"] == 0 for item in ran["endpoint_final"].values())
            and ran["background_units"] == background["completed_units"]
            and all(row["commit_kind"] in ("nrx", "conventional") for row in rows)
        )
        gates[f"{mode}_timing"] = all(
            row["commit_return_ns"] >= row["completed_ns"]
            and row["commit_response_ms"] <= protocol["deadline_ms"]
            and (row["conventional_host_path_ms"] is None or
                 row["conventional_host_path_ms"] <= protocol["conv_path_bound_ms"])
            for row in rows
        )
        pre_records = [row for row in ran["background_records"]
                       if row.get("phase") == "before_nrx_observation"]
        gates[f"{mode}_pre_record_integrity"] = (
            ran["pre_radio_ai_units"] == len(pre_records)
            and all(not row["fallback_guard_violation"]
                    and row["returned_ns"] + protocol["ai_guard_ns"] <= row["earliest_fallback_start_ns"]
                    and any(
                        radio["index"] == row["release_index"]
                        and radio["admitted"]
                        and radio["nrx_dispatched_ns"] <= row["admitted_ns"]
                        and row["returned_ns"] <= radio["nrx_observed_ns"]
                        for radio in rows
                    )
                    for row in pre_records)
        )
    off, one = arms["off"], arms["one"]
    gates["paired_trace"] = all(
        a["index"] == b["index"]
        and a["cell"] == b["cell"]
        and a["channel_seed"] == b["channel_seed"]
        and a["gate_skipped"] == b["gate_skipped"]
        for a, b in zip(off["records"], one["records"])
    )
    gates["pre_radio_exposure"] = (
        off["pre_radio_ai_units"] == 0
        and one["pre_radio_ai_units"] >= protocol["min_pre_radio_ai_units"]
    )
    gates["radio_noninferiority"] = (
        one["correct_cells"] - off["correct_cells"]
        >= protocol["min_correct_cell_difference"]
    )
    report = {
        "schema": "softwall-confirm64-reverse-v1",
        "job": args.job,
        "gates": gates,
        "all_pass": all(gates.values()),
        "comparison": {
            "off_correct": off["correct_cells"],
            "one_correct": one["correct_cells"],
            "off_ai_units": off["background_units"],
            "one_ai_units": one["background_units"],
            "ai_difference_percent": 100 * (one["background_units"] - off["background_units"]) / off["background_units"],
            "one_pre_radio_ai_units": one["pre_radio_ai_units"],
            "off_nrx_response_p99_ms": off["nrx_response_ms"]["p99"],
            "one_nrx_response_p99_ms": one["nrx_response_ms"]["p99"],
            "off_commit_response_p99_ms": off["commit_response_ms"]["p99"],
            "one_commit_response_p99_ms": one["commit_response_ms"]["p99"],
        },
        "interpretation": "One physically completed AI RPC before host NRx observation and first fallback guard in a P150/D130 synthetic sample. Host outstanding overlap does not prove GPU kernel overlap; sample maximum is not WCET. AI timeout/fault continuation is not qualified. This is not an atomic joint AI/recovery policy or production guarantee.",
    }
    output = base / f"confirm64_reverse_job{args.job}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"], "failed": [name for name, passed in gates.items() if not passed], "comparison": report["comparison"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
