#!/usr/bin/env python3
"""Audit the preregistered two-endpoint same-GPU bring-up."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol", type=Path)
    args = parser.parse_args()
    result = args.root / "results/softwall_same_gpu"
    raw = result / "raw"
    protocol = load(args.protocol or result / "confirm55_two_endpoint_bringup_protocol.json")
    radio, contract = protocol["radio"], protocol["contract"]
    prefix = f"confirm55_two_endpoint_job{args.job}"
    ran = load(raw / f"{prefix}_controller.json")
    workers = [load(raw / f"{prefix}_worker{cell}.json") for cell in range(2)]
    background = load(raw / f"{prefix}_background.json")
    requal = load(raw / f"confirm55_requalification_job{args.job}.json")
    records = ran["records"]
    gates = {
        "requalification": requal["iterations"] == 200 and requal["deadline_misses"] == 0,
        "same_host_job": len({x["host"] for x in (ran, *workers, background, requal)}) == 1
        and all(x["slurm_job_id"] == args.job for x in (ran, *workers, background, requal)),
        "fixed_input": ran["cells"] == radio["cells"]
        and ran["iterations"] == radio["iterations"]
        and ran["period_ms"] == radio["period_ms"]
        and ran["deadline_ms"] == radio["deadline_ms"]
        and ran["payload_seed"] == radio["payload_seed"]
        and ran["snr_db"] == radio["snr_db"]
        and ran["channel_seed_base"] == radio["channel_seed_base"]
        and len(records) == 2 * radio["iterations"]
        and all(
            row["index"] == position // 2
            and row["cell"] == position % 2
            and row["slot_id"] == position
            and row["channel_seed"] == radio["channel_seed_base"] + position
            and round((row["cutoff_ns"] - row["release_ns"]) / 1e6)
            == radio["fallback_latest_starts_ms_from_release"][position % 2]
            for position, row in enumerate(records)
        ),
        "contract": ran["nrx_bound_ms"] == contract["nrx_end_to_end_bound_ms"]
        and ran["conv_bound_ms"] == contract["conventional_gpu_bound_ms"]
        and ran["commit_guard_ms"] == contract["commit_guard_ms"]
        and ran["endpoint_timeout_ms"] == contract["endpoint_physical_wait_timeout_ms"]
        and ran["ai_budget_ms"] == contract["ai_host_budget_ms"]
        and ran["ai_rpc_timeout_ms"] == contract["ai_rpc_timeout_ms"]
        and ran["ai_guard_ms"] == contract["ai_guard_ms"]
        and all(40 <= x["visible_sm_count"] <= 44 for x in workers)
        and int(background["mps_active_thread_percentage"]) == 20,
        "worker_counts": all(x["completed_units"] == ran["warmup"] + 1 + radio["iterations"] for x in workers),
        "record_integrity": ran["correct_cells"] == sum(bool(x["correct"]) for x in records)
        and ran["deadline_misses"] == sum(bool(x["deadline_miss"]) for x in records)
        and ran["nrx_bound_violations"] == sum(bool(x["nrx_bound_violation"]) for x in records)
        and ran["conv_bound_violations"] == sum(bool(x["conv_bound_violation"]) for x in records)
        and ran["nrx_commits"] + ran["conv_commits"] == len(records)
        and all(x["endpoint_id"] in (None, f"nrx{x['cell']}") for x in records),
        "mandatory_first": ran["runtime_metrics"].get("mandatory_reserved", 0) == len(records)
        and ran["runtime_metrics"].get("nrx_attached_to_mandatory", 0)
        == len(records) - ran["admission_rejections"]
        and ran["runtime_metrics"].get("fallback_credit_released", 0) == len(records),
        "background_integrity": ran["background_units"] == background["completed_units"]
        == len(ran["background_records"]),
    }
    for path, digest in protocol["source_sha256_before_run"].items():
        gates["source_sha256_" + Path(path).name] = hashlib.sha256((args.root / path).read_bytes()).hexdigest() == digest
    for key, expected in protocol["frozen_gates"].items():
        if key == "completed_cell_records":
            actual = len(records)
        elif key == "correct_cells_at_least":
            gates[key] = ran["correct_cells"] >= expected
            continue
        elif key == "background_units_greater_than_zero":
            gates[key] = ran["background_units"] > 0
            continue
        elif key in ("background_faults", "endpoint_faults"):
            actual = len(ran[key])
        elif key == "fallback_calendar_outstanding":
            actual = ran["fallback_calendar_final"]["outstanding"]
        elif key == "both_endpoint_outstanding":
            gates[key] = all(x["outstanding"] == expected for x in ran["endpoint_final"].values())
            continue
        elif key == "exactly_one_commit_per_cell":
            gates[key] = all(x["commit_kind"] in ("nrx", "conventional") for x in records)
            continue
        else:
            actual = ran[key]
        gates[key] = actual == expected
    gates["all_pass"] = all(gates.values())
    failed = [key for key, value in gates.items() if not value]
    summary = {key: ran[key] for key in (
        "correct_cells", "deadline_misses", "nrx_bound_violations", "conv_bound_violations",
        "admission_rejections", "retime_rejections", "fallbacks", "early_fallbacks",
        "background_units", "response_ms", "nrx_response_ms",
    )}
    report = {"schema": "softwall-confirm55-bringup-audit-v1", "job": args.job,
              "host": ran["host"], "summary": summary, "gates": gates}
    fallback_counts = Counter(row["index"] for row in records if row["fallback"])
    double_fallback_indices = [index for index, count in fallback_counts.items() if count == 2]
    report["descriptive_double_fallback_indices"] = double_fallback_indices
    lines = [
        "# Confirm55 two-endpoint same-GPU MPS bring-up", "",
        f"Job `{args.job}` on `{ran['host']}`; two pinned cap40 NeuralRx endpoints, one cap20 AI worker, one conventional lane; 1,000 simultaneous two-cell P150/D130 releases.",
        "",
        f"Correct {ran['correct_cells']}/{len(records)} cells; RAN misses {ran['deadline_misses']}; NRx 50 ms violations {ran['nrx_bound_violations']}; conventional GPU 25 ms violations {ran['conv_bound_violations']}; AI units {ran['background_units']}.",
        f"Fallbacks {ran['fallbacks']}, early {ran['early_fallbacks']}; NRx rejects {ran['admission_rejections']}; retime rejects {ran['retime_rejections']}.",
        f"Radio p99/max {ran['response_ms']['p99']:.3f}/{ran['response_ms']['max']:.3f} ms; NRx p99/max {ran['nrx_response_ms']['p99']:.3f}/{ran['nrx_response_ms']['max']:.3f} ms.",
        f"Natural two-cell simultaneous fallback occurred at release indices {double_fallback_indices}; this is descriptive and was not a frozen pass gate.",
        "",
        f"Frozen gate: {'PASS' if gates['all_pass'] else 'FAIL'}. Failed checks: {failed or 'none'}.",
        "This is physical integration qualification, not a strong-baseline comparison or WCET proof. Worker timestamps do not separately prove concurrent kernel overlap.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    if not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
