#!/usr/bin/env python3
"""Audit frozen Confirm42 transaction integration and recovery-gap AI lease."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    campaign = protocol["campaign"]
    cap = protocol["endpoint"]["mps_cap"]
    prefix = f"{campaign}_cap{cap}_job{args.job}"
    requal = json.loads((args.raw / f"{campaign}_requalification_job{args.job}.json").read_text())
    controller = json.loads((args.raw / (prefix + "_controller.json")).read_text())
    worker = json.loads((args.raw / (prefix + "_worker.json")).read_text())
    background = json.loads((args.raw / (prefix + "_background.json")).read_text())
    records = controller["records"]
    duplicate_probe_attempts = sum(
        item["fallback"] and not item["endpoint_timeout"] for item in records
    )
    radio = protocol["radio"]
    checks = {
        "same_node_job": len({
            (item["host"], item["slurm_job_id"])
            for item in (requal, controller, worker, background)
        }) == 1 and controller["slurm_job_id"] == args.job,
        "requalification_deadline_misses_zero": requal["iterations"] == 200 and requal["deadline_misses"] == 0,
        "radio_trace_and_count": (
            controller["iterations"] == len(records) == radio["iterations"]
            and controller["payload_seed"] == radio["payload_seed"]
            and controller["channel_seed_base"] == radio["channel_seed_base"]
            and controller["period_ms"] == radio["period_ms"]
            and controller["deadline_ms"] == radio["deadline_ms"]
            and controller["snr_db"] == radio["snr_db"]
            and [item["index"] for item in records] == list(range(radio["iterations"]))
            and all(item["channel_seed"] == radio["channel_seed_base"] + item["index"] for item in records)
        ),
        "transaction_parameters": (
            controller["policy"] == "s2_transactional"
            and controller["nrx_bound_ms"] == protocol["endpoint"]["nrx_end_to_end_bound_ms"]
            and controller["conv_bound_ms"] == protocol["recovery"]["conventional_gpu_bound_ms"]
            and controller["commit_guard_ms"] == protocol["recovery"]["commit_guard_ms"]
            and controller["recovery_gap_ai"]
        ),
        "caps_and_worker_units": (
            worker["visible_sm_count"] == 86
            and int(background["mps_active_thread_percentage"]) == protocol["background"]["mps_cap"]
            and background["visible_sm_count"] <= 22
            and worker["completed_units"] == radio["iterations"] + controller["warmup"] + controller["noisy_warmup_units"]
            and background["completed_units"] == controller["background_units"]
        ),
        "ran_deadline_misses_zero": controller["deadline_misses"] == 0,
        "endpoint_timeouts_zero": controller["endpoint_timeouts"] == 0,
        "bound_violations_zero": controller["nrx_bound_violations"] == 0 and controller["conv_bound_violations"] == 0,
        "ai_contract_zero": (
            controller["background_budget_violations"] == 0
            and controller["background_release_crossings"] == 0
            and not controller["background_faults"]
        ),
        "single_commit": (
            controller["duplicate_commits"] == 0
            and controller["nrx_commits"] + controller["conv_commits"] == radio["iterations"]
            and all(item["commit_kind"] in ("nrx", "conventional") for item in records)
        ),
        "post_commit_probe_rejected": (
            controller["runtime_metrics"].get("endpoint_epoch_dropped", 0)
            == duplicate_probe_attempts
        ),
        "all_credits_released": (
            controller["fallback_calendar_final"]["outstanding"] == 0
            and controller["endpoint_final"]["outstanding"] == 0
        ),
        "natural_fallbacks_positive": controller["fallbacks"] > 0,
        "recovery_gap_ai_positive": controller["recovery_gap_ai_units"] > 0,
    }
    result = {
        "schema": "softwall-confirm42-transaction-smoke-audit-v1",
        "job": args.job,
        "host": controller["host"],
        "correct_releases": controller["correct_releases"],
        "deadline_misses": controller["deadline_misses"],
        "fallbacks": controller["fallbacks"],
        "admission_rejections": controller["admission_rejections"],
        "background_units": controller["background_units"],
        "recovery_gap_ai_units": controller["recovery_gap_ai_units"],
        "natural_recoveries": sum(item["fallback"] and item["conventional_correct"] for item in records),
        "synthetic_duplicate_probe_attempts": duplicate_probe_attempts,
        "checks": checks,
        "all_frozen_gates_pass": all(checks.values()),
    }
    lines = [
        "# Confirm42 transactional external endpoint smoke", "",
        f"- Job/node: {args.job} / {result['host']}",
        f"- Radio: {result['correct_releases']}/{radio['iterations']} correct; {result['deadline_misses']} deadline misses",
        f"- Natural fallbacks: {result['fallbacks']}; admission rejections: {result['admission_rejections']}",
        f"- Correct conventional recoveries: {result['natural_recoveries']}",
        f"- Completed AI units: {result['background_units']}; before reserved recovery: {result['recovery_gap_ai_units']}",
        f"- Synthetic post-commit NRx probes: {duplicate_probe_attempts}; accepted duplicates: {controller['duplicate_commits']}",
        f"- All frozen gates pass: {result['all_frozen_gates_pass']}",
        "- Scope: " + protocol["interpretation_limit"],
        "", "## Failed checks", "",
    ]
    lines.extend(f"- {name}" for name, passed in checks.items() if not passed)
    if all(checks.values()):
        lines.append("- None")
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    if not result["all_frozen_gates_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
