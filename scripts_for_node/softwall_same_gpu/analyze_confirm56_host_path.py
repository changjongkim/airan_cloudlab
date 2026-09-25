#!/usr/bin/env python3.11
"""Audit Confirm56 against its preregistered host-to-commit gates."""

from __future__ import annotations

import argparse
import hashlib
import json
import statistics
from pathlib import Path


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def distribution(values: list[float]) -> dict:
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": ordered[0] if ordered else None,
        "median": statistics.median(ordered) if ordered else None,
        "p99_nearest_rank": ordered[(99 * len(ordered) + 99) // 100 - 1] if ordered else None,
        "max": ordered[-1] if ordered else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    result = root / "results/softwall_same_gpu"
    raw = result / "raw"
    protocol = read_json(result / "confirm56_host_path_qualification_protocol.json")
    prefix = f"confirm56_host_path_job{args.job}"
    ran = read_json(raw / f"{prefix}_controller.json")
    workers = [read_json(raw / f"{prefix}_worker{cell}.json") for cell in range(2)]
    background = read_json(raw / f"{prefix}_background.json")
    requal = read_json(raw / f"confirm56_requalification_job{args.job}.json")
    records = ran["records"]
    radio = protocol["radio"]
    contract = protocol["contract"]
    gates: dict[str, bool] = {}
    source_evidence = {}
    for path, digest in protocol["source_sha256_before_run"].items():
        current = root / path
        if hashlib.sha256(current.read_bytes()).hexdigest() == digest:
            source_evidence[path] = str(current.relative_to(root))
        elif path == "scripts_for_node/task1/isca_v2/dart_runtime.py":
            frozen = result / "frozen_source/confirm56/dart_runtime.py"
            if hashlib.sha256(frozen.read_bytes()).hexdigest() == digest:
                source_evidence[path] = str(frozen.relative_to(root))
    gates["source_hashes"] = len(source_evidence) == len(protocol["source_sha256_before_run"])
    gates["same_host_job"] = len({item["host"] for item in (ran, *workers, background, requal)}) == 1 and all(
        str(item["slurm_job_id"]) == args.job for item in (ran, *workers, background, requal)
    )
    gates["requalification"] = requal["iterations"] == 200 and requal["deadline_misses"] == 0
    gates["input_contract"] = (
        ran["cells"] == radio["cells"]
        and ran["iterations"] == radio["iterations"]
        and ran["period_ms"] == radio["period_ms"]
        and ran["deadline_ms"] == radio["deadline_ms"]
        and ran["payload_seed"] == radio["payload_seed"]
        and ran["snr_db"] == radio["snr_db"]
        and ran["channel_seed_base"] == radio["channel_seed_base"]
        and ran["nrx_bound_ms"] == contract["nrx_bound_ms"]
        and ran["conv_bound_ms"] == contract["conventional_host_to_commit_bound_ms"]
        and ran["commit_guard_ms"] == contract["commit_guard_ms"]
        and all(40 <= item["visible_sm_count"] <= 44 for item in workers)
        and int(background["mps_active_thread_percentage"]) == contract["ai_cap"]
    )
    gates["record_integrity"] = (
        len(records) == 2 * radio["iterations"]
        and all(
            row["index"] == position // 2
            and row["cell"] == position % 2
            and row["slot_id"] == position
            and row["channel_seed"] == radio["channel_seed_base"] + position
            and row["nrx_dispatched_ns"] >= row["nrx_dispatch_begin_ns"] >= row["release_ns"]
            and row["nrx_observed_ns"] >= row["nrx_dispatched_ns"]
            and row["commit_return_ns"] >= row["completed_ns"] >= row["release_ns"]
            and row["deadline_miss"] == (
                row["commit_return_ns"] > row["release_ns"] + round(radio["deadline_ms"] * 1e6)
            )
            and row["commit_kind"] in ("nrx", "conventional")
            for position, row in enumerate(records)
        )
        and ran["correct_cells"] == sum(bool(row["correct"]) for row in records)
        and ran["deadline_misses"] == sum(bool(row["deadline_miss"]) for row in records)
        and ran["conv_path_bound_violations"] == sum(bool(row["conv_path_bound_violation"]) for row in records)
        and ran["background_units"] == len(ran["background_records"]) == background["completed_units"]
        and all(item["completed_units"] == ran["warmup"] + 1 + radio["iterations"] for item in workers)
    )
    fallback_rows = [row for row in records if row["fallback"]]
    gates["path_measurement_integrity"] = (
        len(fallback_rows) == ran["fallbacks"]
        and all(
            row["conventional_host_path_ms"] is not None
            and row["fallback_actual_start_ns"] <= row["completed_ns"] <= row["commit_return_ns"]
            and abs(row["conventional_host_path_ms"] - (
                row["commit_return_ns"] - row["fallback_actual_start_ns"]
            ) / 1e6) < 1e-9
            and row["conv_path_bound_violation"] == (
                row["conventional_host_path_ms"] > contract["conventional_host_to_commit_bound_ms"]
            )
            for row in fallback_rows
        )
    )
    frozen = protocol["frozen_gates"]
    observed = {
        "completed_cell_records": len(records),
        "correct_cells_at_least": ran["correct_cells"],
        "deadline_misses_by_commit_return": ran["deadline_misses"],
        "conventional_host_path_bound_violations": ran["conv_path_bound_violations"],
        "nrx_bound_violations": ran["nrx_bound_violations"],
        "background_budget_violations": ran["background_budget_violations"],
        "background_release_crossings": ran["background_release_crossings"],
        "background_faults": len(ran["background_faults"]),
        "endpoint_faults": len(ran["endpoint_faults"]),
        "fallback_calendar_outstanding": ran["fallback_calendar_final"]["outstanding"],
        "both_endpoint_outstanding": max(item["outstanding"] for item in ran["endpoint_final"].values()),
        "background_units_greater_than_zero": ran["background_units"] > 0,
    }
    for name, expected in frozen.items():
        if name == "correct_cells_at_least":
            gates[name] = observed[name] >= expected
        else:
            gates[name] = observed[name] == expected
    pair_finish_ms = []
    for index in range(radio["iterations"]):
        pair = records[2 * index:2 * index + 2]
        pair_finish_ms.append((max(row["nrx_observed_ns"] for row in pair) - pair[0]["release_ns"]) / 1e6)
    report = {
        "schema": "softwall-confirm56-host-path-audit-v1",
        "job": args.job,
        "host": ran["host"],
        "gates": gates,
        "source_evidence": source_evidence,
        "all_pass": all(gates.values()),
        "observed": observed,
        "fallback_host_to_commit_ms": distribution([row["conventional_host_path_ms"] for row in fallback_rows]),
        "nrx_pair_observed_ms": distribution(pair_finish_ms),
        "radio_commit_response_ms": distribution([row["commit_response_ms"] for row in records]),
        "interpretation": "Sampled host-path evidence only; no WCET, production MAC expiry, joint policy, or AI-before-radio claim.",
    }
    out = result / f"confirm56_host_path_qualification_job{args.job}.json"
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"all_pass": report["all_pass"], "failed": [k for k, v in gates.items() if not v],
                      "fallback_host_to_commit_ms": report["fallback_host_to_commit_ms"],
                      "nrx_pair_observed_ms": report["nrx_pair_observed_ms"]}, indent=2))
    if not report["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
