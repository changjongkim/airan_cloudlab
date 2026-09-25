#!/usr/bin/env python3.11
"""Frozen two-seed audit of the conditional PHY value-bin baseline."""

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
    args = parser.parse_args()
    root = args.root.resolve()
    base = root / "results/softwall_same_gpu"
    protocol = read(base / "confirm69_value_gate_protocol.json")
    gates: dict[str, bool] = {}
    gates["source_hashes"] = all(
        hashlib.sha256((root / name).read_bytes()).hexdigest() == digest
        for name, digest in protocol["source_sha256_before_run"].items()
    )
    arms: dict[str, dict] = {}
    for entry in protocol["arms"]:
        prefix = entry["prefix"]
        mode = entry["mode"]
        ran = read(base / "raw" / f"{prefix}_controller.json")
        workers = [read(base / "raw" / f"{prefix}_worker{cell}.json") for cell in range(2)]
        background = read(base / "raw" / f"{prefix}_background.json")
        requal = read(base / "raw" / f"{prefix}_requalification.json")
        arms[prefix] = ran
        rows = ran["records"]
        lower = protocol["thresholds"][mode]
        upper = protocol["value_bin_upper"]
        gates[f"{prefix}_contract"] = (
            ran["iterations"] == protocol["iterations"]
            and ran["period_ms"] == protocol["period_ms"]
            and ran["deadline_ms"] == protocol["deadline_ms"]
            and ran["payload_seed"] == entry["payload_seed"]
            and ran["channel_seed_base"] == entry["channel_seed_base"]
            and ran["gate_mode"] == mode
            and ran["gate_threshold"] == lower
            and ran["gate_upper"] == upper
            and ran["early_mandatory"] == "on"
            and ran["noise_reference"] == "pre_fading"
            and len(rows) == 2 * protocol["iterations"]
            and requal["iterations"] == 200
            and requal["deadline_misses"] == 0
        )
        gates[f"{prefix}_provenance"] = (
            len({item["host"] for item in (ran, *workers, background, requal)}) == 1
            and all(str(item["slurm_job_id"]) == protocol["job"]
                    for item in (ran, *workers, background, requal))
            and all(40 <= worker["visible_sm_count"] <= 44 for worker in workers)
            and int(background["mps_active_thread_percentage"]) == 20
        )
        gates[f"{prefix}_safety"] = (
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
        gates[f"{prefix}_online_gate"] = all(
            row["index"] == index // 2
            and row["cell"] == index % 2
            and row["channel_seed"] == ran["channel_seed_base"] + index
            and row["feature_begin_ns"] >= row["release_ns"]
            and row["feature_return_ns"] >= row["feature_begin_ns"]
            and row["gate_skipped"] == (
                row["channel_estimate_power"] < lower
                or (mode == "value_bin" and row["channel_estimate_power"] >= upper)
            )
            for index, row in enumerate(rows)
        )
    pairs = []
    for pair in protocol["pairs"]:
        low = arms[pair["low_prefix"]]
        value = arms[pair["value_prefix"]]
        label = pair["name"]
        gates[f"{label}_paired_trace"] = all(
            left["index"] == right["index"]
            and left["cell"] == right["cell"]
            and left["channel_seed"] == right["channel_seed"]
            for left, right in zip(low["records"], value["records"])
        )
        gates[f"{label}_extra_value_skips"] = (
            value["gate_skips"] - low["gate_skips"]
            >= protocol["min_extra_value_skips"]
        )
        gates[f"{label}_radio_noninferiority"] = (
            value["correct_cells"] - low["correct_cells"]
            >= protocol["min_correct_cell_difference"]
        )
        pairs.append({
            "name": label,
            "low_correct": low["correct_cells"],
            "value_correct": value["correct_cells"],
            "low_ai_units": low["background_units"],
            "value_ai_units": value["background_units"],
            "ai_gain_percent": 100 * (value["background_units"] - low["background_units"])
            / low["background_units"],
            "low_gate_skips": low["gate_skips"],
            "value_gate_skips": value["gate_skips"],
        })
    report = {
        "schema": "softwall-confirm69-value-gate-baseline-v1",
        "job": protocol["job"],
        "gates": gates,
        "all_pass": all(gates.values()),
        "pairs": pairs,
        "interpretation": "Train-derived incremental-PHY-value bin as a stronger component baseline. Synthetic P150/D130; AI only after radio. AI gain sign is not a frozen pass gate. This is neither AI-aware recovery replan nor joint policy, and no production deadline or novelty claim follows.",
    }
    output = base / f"confirm69_value_gate_job{protocol['job']}.json"
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"],
                      "failed": [name for name, passed in gates.items() if not passed],
                      "pairs": pairs}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
