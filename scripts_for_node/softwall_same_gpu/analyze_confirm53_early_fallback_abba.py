#!/usr/bin/env python3
"""Audit the frozen paired early-versus-latest fallback mechanism test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def checked_arm(raw: Path, job: str, config: dict, protocol: dict) -> tuple[dict, list[str]]:
    prefix = raw / f"confirm53_arm{config['arm']}_{config['timing']}_job{job}_cap80_job{job}"
    ran = read(Path(str(prefix) + "_controller.json"))
    worker = read(Path(str(prefix) + "_worker.json"))
    background = read(Path(str(prefix) + "_background.json"))
    radio = protocol["radio"]
    contract = protocol["transaction"]
    ai = protocol["background_both_arms"]
    rows = ran["records"]
    errors = []
    expected = {
        "schema": "softwall-same-request-transaction-controller-v1",
        "slurm_job_id": job,
        "iterations": radio["iterations_per_arm"],
        "period_ms": radio["period_ms"],
        "deadline_ms": radio["deadline_ms"],
        "snr_db": radio["snr_db"],
        "payload_seed": config["payload_seed"],
        "channel_seed_base": config["channel_seed_base"],
        "fallback_timing_policy": config["timing"],
        "nrx_bound_ms": contract["nrx_bound_ms"],
        "conv_bound_ms": contract["conventional_gpu_bound_ms"],
        "commit_guard_ms": contract["commit_guard_ms"],
        "endpoint_timeout_ms": contract["endpoint_timeout_ms"],
        "recovery_gap_ai": contract["recovery_gap_ai"],
        "ai_budget_ms": ai["host_budget_ms"],
        "ai_rpc_timeout_ms": ai["socket_timeout_ms"],
        "ai_guard_ms": ai["guard_ms"],
    }
    for key, value in expected.items():
        if ran.get(key) != value:
            errors.append(f"{key}: {ran.get(key)!r} != {value!r}")
    if len(rows) != radio["iterations_per_arm"]:
        errors.append("record count")
    if ran["correct_releases"] != sum(bool(row["correct"]) for row in rows):
        errors.append("correct count")
    if ran["deadline_misses"] != sum(bool(row["deadline_miss"]) for row in rows):
        errors.append("deadline count")
    if ran["admission_rejections"] != sum(bool(row["admission_rejected"]) for row in rows):
        errors.append("admission rejection count")
    if ran["fallbacks"] != sum(bool(row["fallback"]) for row in rows):
        errors.append("fallback count")
    if ran["early_fallbacks"] != sum(bool(row["fallback_early"]) for row in rows):
        errors.append("early fallback count")
    if ran["nrx_commits"] + ran["conv_commits"] != len(rows):
        errors.append("exactly one radio commit per release")
    if ran["background_units"] != len(ran["background_records"]):
        errors.append("background count")
    if ran["background_units"] != background["completed_units"]:
        errors.append("background worker count")
    if not (
        ran["host"] == worker["host"] == background["host"]
        and ran["slurm_job_id"] == worker["slurm_job_id"] == background["slurm_job_id"] == job
    ):
        errors.append("host/job mismatch")
    if worker["visible_sm_count"] != 86:
        errors.append("endpoint MPS cap")
    if int(background["mps_active_thread_percentage"]) != ai["mps_cap"]:
        errors.append("background MPS cap")
    period_ns = round(radio["period_ms"] * 1e6)
    for index, row in enumerate(rows):
        if row["index"] != index or row["channel_seed"] != config["channel_seed_base"] + index:
            errors.append(f"trace index {index}")
            break
        if row["fallback_early"]:
            if config["timing"] != "early_if_clear" or row["neural_correct"] is not False or row["endpoint_timeout"]:
                errors.append(f"invalid early trigger at {index}")
            if index + 1 < len(rows) and row["fallback_actual_start_ns"] is not None:
                next_release_ns = row["release_ns"] + period_ns
                reserved_finish_ns = row["fallback_actual_start_ns"] + round(
                    (contract["conventional_gpu_bound_ms"] + contract["commit_guard_ms"]) * 1e6
                )
                if reserved_finish_ns > next_release_ns:
                    errors.append(f"early reservation crossed next release at {index}")
                if row["release_ns"] + round(row["response_ms"] * 1e6) > next_release_ns:
                    errors.append(f"early recovery crossed next release at {index}")
        if row["fallback"] and (row["fallback_actual_start_ns"] is None or row["fallback_started"] is not True):
            errors.append(f"fallback missing start at {index}")
        target = row["recovery_gap_ai_deadline_ns"]
        if target is not None and target > min(row["fallback_start_ns"], row["release_ns"] + period_ns):
            errors.append(f"gap deadline crossed at {index}")
    release_ns = {row["index"]: row["release_ns"] for row in rows}
    if any(unit["returned_ns"] > release_ns[unit["release_index"]] + period_ns for unit in ran["background_records"]):
        errors.append("observed AI release crossing")
    for key in (
        "deadline_misses", "endpoint_timeouts", "nrx_bound_violations", "conv_bound_violations",
        "duplicate_commits", "fallback_not_started", "fallback_start_late_gt_1ms",
        "background_budget_violations", "background_release_crossings",
    ):
        if ran[key]:
            errors.append(f"{key}={ran[key]}")
    if ran["background_faults"]:
        errors.append("background faults")
    if ran["endpoint_final"]["outstanding"] or ran["fallback_calendar_final"]["outstanding"]:
        errors.append("outstanding credit")
    return ran, errors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-arms", type=int, choices=(2, 4), required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    arms = protocol["radio"]["arms"][:args.completed_arms]
    requal = read(args.raw / f"confirm53_requalification_job{args.job}.json")
    errors = []
    if requal["slurm_job_id"] != args.job or requal["iterations"] != 200 or requal["deadline_misses"]:
        errors.append("requalification")
    data = {}
    summaries = []
    hosts = {requal["host"]}
    for config in arms:
        ran, arm_errors = checked_arm(args.raw, args.job, config, protocol)
        data[config["arm"]] = ran
        hosts.add(ran["host"])
        errors.extend(f"arm{config['arm']}: {problem}" for problem in arm_errors)
        summaries.append({
            "arm": config["arm"], "pair": config["pair"], "timing": config["timing"],
            "correct": ran["correct_releases"], "miss": ran["deadline_misses"],
            "ai_units": ran["background_units"], "admission_rejections": ran["admission_rejections"],
            "fallbacks": ran["fallbacks"], "early_fallbacks": ran["early_fallbacks"],
            "ran_p99_ms": ran["response_ms"]["p99"],
            "no_fallback_response_le_3ms": sum(
                not row["fallback"] and row["response_ms"] <= 3.0
                for row in ran["records"][:-1]
            ),
        })
    if len(hosts) != 1:
        errors.append("not all conditions on one host")
    pair_gates = {}
    pair_rows = []
    for pair in range(1, args.completed_arms // 2 + 1):
        two = [(cfg, data[cfg["arm"]]) for cfg in arms if cfg["pair"] == pair]
        early = next(ran for cfg, ran in two if cfg["timing"] == "early_if_clear")
        latest = next(ran for cfg, ran in two if cfg["timing"] == "latest")
        if [(r["index"], r["channel_seed"]) for r in early["records"]] != [
            (r["index"], r["channel_seed"]) for r in latest["records"]
        ]:
            errors.append(f"pair{pair}: trace mismatch")
        gain = early["correct_releases"] - latest["correct_releases"]
        reject_reduction = latest["admission_rejections"] - early["admission_rejections"]
        early_only = sum(a["correct"] and not b["correct"] for a, b in zip(early["records"], latest["records"]))
        latest_only = sum(b["correct"] and not a["correct"] for a, b in zip(early["records"], latest["records"]))
        pair_gates[f"pair{pair}"] = (
            gain >= protocol["pair_gate"]["minimum_early_minus_latest_correct_tb_per_1000"]
            and reject_reduction > 0 and early["early_fallbacks"] > 0
            and early["background_units"] > 0
            and not any(item.startswith(f"arm{cfg['arm']}:") for cfg, _ in two for item in errors)
        )
        pair_rows.append({
            "pair": pair, "correct_gain": gain, "reject_reduction": reject_reduction,
            "early_only_correct": early_only, "latest_only_correct": latest_only,
            "pass": pair_gates[f"pair{pair}"],
        })
    all_pass = args.completed_arms == 4 and not errors and all(pair_gates.values())
    report = {
        "schema": "softwall-confirm53-early-fallback-abba-v1", "job": args.job,
        "completed_arms": args.completed_arms, "planned_arms": 4,
        "summaries": summaries, "pairs": pair_rows, "errors": errors,
        "pair_gates": pair_gates, "all_pass": all_pass,
    }
    lines = [
        "# Confirm53 known-CRC-failure early recovery ablation", "",
        "| arm | pair | timing | correct/1000 | miss | AI units | NRx rejects | fallback/early | RAN p99 ms |",
        "|---:|---:|---|---:|---:|---:|---:|---|---:|",
    ]
    for row in summaries:
        lines.append(
            f"| {row['arm']} | {row['pair']} | {row['timing']} | {row['correct']} | "
            f"{row['miss']} | {row['ai_units']} | {row['admission_rejections']} | "
            f"{row['fallbacks']}/{row['early_fallbacks']} | {row['ran_p99_ms']:.3f} |"
        )
    lines += ["", "| pair | early−latest correct TB | fewer NRx rejects | early-only correct | latest-only correct | frozen pair gate |",
              "|---:|---:|---:|---:|---:|---|"]
    for row in pair_rows:
        lines.append(f"| {row['pair']} | {row['correct_gain']} | {row['reject_reduction']} | {row['early_only_correct']} | {row['latest_only_correct']} | {'PASS' if row['pass'] else 'FAIL'} |")
    lines += ["", f"Completed arms: {args.completed_arms}/4. Overall frozen gate: {'PASS' if all_pass else 'FAIL or incomplete'}.",
              "This only tests a one-cell known-failure timing mechanism. It is not a comparison to the strong combined baseline or a WCET proof."]
    if args.completed_arms == 4:
        lines += [
            "",
            "AI throughput did not improve consistently: early−latest units were "
            f"{summaries[0]['ai_units'] - summaries[1]['ai_units']:+d} and "
            f"{summaries[3]['ai_units'] - summaries[2]['ai_units']:+d} in the two pairs. "
            "At P45, the 40 ms AI budget plus 2 ms guard leaves about 3 ms for the radio response and host decision; "
            "this makes background admission sensitive to small timing changes. "
            "Descriptive no-fallback response ≤3 ms counts by arm: "
            + ", ".join(str(row["no_fallback_response_le_3ms"]) for row in summaries)
            + ". These counts were inspected after the frozen gate and are not a new pass criterion.",
        ]
    if errors:
        lines += ["", "Errors:", *(f"- {item}" for item in errors)]
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
