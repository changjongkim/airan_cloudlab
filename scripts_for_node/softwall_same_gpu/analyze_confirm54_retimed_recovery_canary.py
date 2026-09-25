#!/usr/bin/env python3
"""Audit the frozen one-cell GPU integration canary for calendar retiming."""

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
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = args.root / "results/softwall_same_gpu"
    raw = result / "raw"
    protocol = read(result / "confirm54_retimed_recovery_canary_protocol.json")
    prefix = raw / f"confirm54_retimed_job{args.job}_cap80_job{args.job}"
    ran = read(Path(str(prefix) + "_controller.json"))
    worker = read(Path(str(prefix) + "_worker.json"))
    background = read(Path(str(prefix) + "_background.json"))
    requal = read(raw / f"confirm54_requalification_job{args.job}.json")
    radio = protocol["radio"]
    contract = protocol["contract"]
    records = ran["records"]
    gates = {
        "requalification": (
            requal["iterations"] == 200
            and requal["deadline_misses"] == 0
            and requal["slurm_job_id"] == args.job
        ),
        "same_node_job": (
            len({x["host"] for x in (requal, ran, worker, background)}) == 1
            and all(x["slurm_job_id"] == args.job for x in (requal, ran, worker, background))
        ),
        "fixed_input": (
            ran["iterations"] == len(records) == radio["iterations"]
            and ran["period_ms"] == radio["period_ms"]
            and ran["deadline_ms"] == radio["deadline_ms"]
            and ran["payload_seed"] == radio["payload_seed"]
            and ran["channel_seed_base"] == radio["channel_seed_base"]
            and ran["snr_db"] == radio["snr_db"]
            and all(
                row["index"] == index
                and row["channel_seed"] == radio["channel_seed_base"] + index
                for index, row in enumerate(records)
            )
        ),
        "contract": (
            ran["fallback_timing_policy"] == contract["fallback_timing"]
            and ran["nrx_bound_ms"] == contract["nrx_bound_ms"]
            and ran["conv_bound_ms"] == contract["conv_bound_ms"]
            and ran["commit_guard_ms"] == contract["commit_guard_ms"]
            and ran["endpoint_timeout_ms"] == contract["endpoint_timeout_ms"]
            and ran["ai_budget_ms"] == contract["ai_budget_ms"]
            and ran["ai_guard_ms"] == contract["ai_guard_ms"]
            and ran["ai_rpc_timeout_ms"] == contract["ai_rpc_timeout_ms"]
            and ran["recovery_gap_ai"] == contract["recovery_gap_ai"]
            and worker["visible_sm_count"] == 86
            and int(background["mps_active_thread_percentage"]) == contract["background_mps_cap"]
        ),
        "counts": (
            ran["correct_releases"] == sum(bool(row["correct"]) for row in records)
            and ran["deadline_misses"] == sum(bool(row["deadline_miss"]) for row in records)
            and ran["admission_rejections"] == sum(bool(row["admission_rejected"]) for row in records)
            and ran["early_fallbacks"] == sum(bool(row["fallback_early"]) for row in records)
            and ran["early_retime_rejections"] == sum(bool(row["early_retime_rejected"]) for row in records)
            and ran["nrx_commits"] + ran["conv_commits"] == len(records)
            and ran["background_units"] == background["completed_units"] == len(ran["background_records"])
        ),
        "early_calendar_path": (
            ran["runtime_metrics"].get("fallback_early_started", 0) == ran["early_fallbacks"]
            and ran["early_fallbacks"] > 0
            and all(
                not row["fallback_early"]
                or (
                    row["neural_correct"] is False
                    and not row["endpoint_timeout"]
                    and row["fallback_started"] is True
                    and row["fallback_actual_start_ns"] is not None
                    and row["fallback_actual_start_ns"] <= row["fallback_start_ns"]
                    and (
                        index == len(records) - 1
                        or row["fallback_actual_start_ns"]
                        + round((contract["conv_bound_ms"] + contract["commit_guard_ms"]) * 1e6)
                        <= row["release_ns"] + round(radio["period_ms"] * 1e6)
                    )
                )
                for index, row in enumerate(records)
            )
        ),
    }
    for path, expected in protocol["source_sha256_before_run"].items():
        actual = hashlib.sha256((args.root / path).read_bytes()).hexdigest()
        gates["source_sha256_" + Path(path).name] = actual == expected
    for key, expected in protocol["pass_gates"].items():
        if key == "early_fallbacks_greater_than_zero":
            gates[key] = ran["early_fallbacks"] > 0
        elif key == "correct_releases_at_least":
            gates[key] = ran["correct_releases"] >= expected
        elif key == "background_units_greater_than_zero":
            gates[key] = ran["background_units"] > 0
        elif key == "fallback_calendar_outstanding":
            gates[key] = ran["fallback_calendar_final"]["outstanding"] == expected
        elif key == "endpoint_outstanding":
            gates[key] = ran["endpoint_final"]["outstanding"] == expected
        else:
            gates[key] = ran[key] == expected
    gates["no_ai_faults"] = not ran["background_faults"]
    gates["all_pass"] = all(gates.values())
    report = {
        "schema": "softwall-confirm54-retimed-recovery-canary-v1",
        "job": args.job,
        "host": ran["host"],
        "summary": {
            key: ran[key] for key in (
                "correct_releases", "deadline_misses", "admission_rejections",
                "fallbacks", "early_fallbacks", "early_retime_rejections",
                "background_units", "nrx_bound_violations", "conv_bound_violations",
            )
        },
        "gates": gates,
    }
    failed = [key for key, passed in gates.items() if not passed]
    lines = [
        "# Confirm54 calendar-retimed early recovery integration canary", "",
        f"Job `{args.job}` on `{ran['host']}`; first allocation failed before the script started because Slurm could not connect I/O. This is the retry on the unchanged frozen protocol.",
        "",
        f"RAN correct {ran['correct_releases']}/{radio['iterations']}, deadline misses {ran['deadline_misses']}, NRx rejects {ran['admission_rejections']}; "
        f"early fallbacks {ran['early_fallbacks']}, retime rejections {ran['early_retime_rejections']}, completed AI units {ran['background_units']}.",
        f"Frozen integration gate: {'PASS' if gates['all_pass'] else 'FAIL'}. Failed checks: {failed or 'none'}.",
        "",
        "This is an integration canary on an already used one-cell trace. It does not test multi-cell physical overlap, prove worst-case service bounds, or establish AI throughput superiority.",
    ]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    if not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
