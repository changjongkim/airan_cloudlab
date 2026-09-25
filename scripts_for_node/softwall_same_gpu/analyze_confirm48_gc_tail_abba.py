#!/usr/bin/env python3
"""Analyze the prospectively frozen GC-on/off same-request tail diagnostic."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval(data: dict) -> tuple[int, int]:
    rows = data["records"]
    return (
        rows[0]["release_ns"],
        max(row["release_ns"] + round(row["response_ms"] * 1e6) for row in rows),
    )


def overlaps_gc(row: dict, event: dict) -> bool:
    start_ns = row["release_ns"]
    finish_ns = start_ns + round(row["response_ms"] * 1e6)
    return event["started_ns"] <= finish_ns and event["finished_ns"] >= start_ns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--completed-rounds", type=int)
    args = parser.parse_args()
    protocol = read(args.protocol)
    radio = protocol["radio"]
    contract = protocol["controller"]
    planned_rounds = len(radio["order"])
    completed_rounds = args.completed_rounds or planned_rounds
    if not 1 <= completed_rounds <= planned_rounds:
        parser.error("completed rounds must be within frozen order")
    requal = read(args.raw / f"confirm48_requalification_job{args.job}.json")
    gates: dict[str, bool] = {
        "requalification": (
            requal["iterations"] == 200
            and requal["deadline_misses"] == 0
            and requal["slurm_job_id"] == args.job
        ),
        "frozen_order": radio["order"] == ["gc_on", "gc_off", "gc_off", "gc_on"],
    }
    hosts = {requal["host"]}
    jobs = {requal["slurm_job_id"]}
    previous_end = None
    rows = []
    traces = []
    for round_id, condition in enumerate(radio["order"][:completed_rounds], 1):
        prefix = args.raw / (
            f"confirm48_gc_r{round_id}_{condition.removeprefix('gc_')}"
            f"_cap80_job{args.job}"
        )
        ran = read(Path(str(prefix) + "_controller.json"))
        endpoint = read(Path(str(prefix) + "_worker.json"))
        background = read(Path(str(prefix) + "_background.json"))
        hosts.update((ran["host"], endpoint["host"], background["host"]))
        jobs.update((ran["slurm_job_id"], endpoint["slurm_job_id"], background["slurm_job_id"]))
        records = ran["records"]
        traces.append([(r["index"], r["channel_seed"]) for r in records])
        label = f"r{round_id}_{condition}"
        gates[label + "_trace"] = (
            ran["iterations"] == len(records) == radio["iterations_per_condition"]
            and ran["period_ms"] == radio["period_ms"]
            and ran["deadline_ms"] == radio["deadline_ms"]
            and ran["payload_seed"] == radio["payload_seed"]
            and ran["channel_seed_base"] == radio["channel_seed_base"]
            and ran["snr_db"] == radio["snr_db"]
            and [r["index"] for r in records] == list(range(radio["iterations_per_condition"]))
            and all(r["channel_seed"] == radio["channel_seed_base"] + r["index"] for r in records)
        )
        gates[label + "_contract"] = (
            ran["gc_mode"] == condition.removeprefix("gc_")
            and ran["schema"] == "softwall-same-request-transaction-controller-v1"
            and ran["policy"] == "s2_transactional"
            and ran["recovery_gap_ai"] == contract["recovery_gap_ai"]
            and ran["nrx_bound_ms"] == contract["nrx_end_to_end_bound_ms"]
            and ran["conv_bound_ms"] == contract["conventional_gpu_bound_ms"]
            and ran["commit_guard_ms"] == contract["commit_guard_ms"]
            and ran["ai_budget_ms"] == contract["ai_budget_ms"]
            and ran["ai_guard_ms"] == contract["ai_guard_ms"]
            and endpoint["visible_sm_count"] == 86
            and int(background["mps_active_thread_percentage"]) == contract["background_mps_cap"]
            and background["completed_units"] == ran["background_units"]
            and ran["nrx_commits"] + ran["conv_commits"] == len(records)
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["endpoint_final"]["outstanding"] == 0
            and ran["duplicate_commits"] == 0
        )
        gates[label + "_safety"] = (
            ran["deadline_misses"] == 0
            and ran["endpoint_timeouts"] == 0
            and ran["conv_bound_violations"] == 0
            and ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and not ran["background_faults"]
        )
        start_ns, end_ns = interval(ran)
        gates[label + "_serial"] = previous_end is None or previous_end < start_ns
        previous_end = end_ns
        violations = [r for r in records if r["nrx_bound_violation"]]
        overlaps = [
            {"release_index": r["index"], "response_ms": r["response_ms"],
             "generation": e["generation"], "gc_duration_ms": e["duration_ms"]}
            for r in violations for e in ran["gc_long_events"] if overlaps_gc(r, e)
        ]
        collections = sum(ran["gc_counts_by_generation"].values())
        if condition == "gc_on":
            gates[label + "_hypothesis"] = bool(overlaps)
        else:
            gates[label + "_hypothesis"] = collections == 0 and not violations
        rows.append({
            "round": round_id,
            "condition": condition,
            "correct": ran["correct_releases"],
            "deadline_misses": ran["deadline_misses"],
            "nrx_bound_violation_indices": [r["index"] for r in violations],
            "gc_collections": collections,
            "gc_long_events": ran["gc_long_events"],
            "bound_violation_gc_overlaps": overlaps,
            "background_units": ran["background_units"],
        })
    gates["same_host_job"] = len(hosts) == 1 and jobs == {args.job}
    gates["identical_trace"] = all(trace == traces[0] for trace in traces[1:])
    gates["full_frozen_order_completed"] = completed_rounds == planned_rounds
    gates["all_pass"] = all(gates.values())
    report = {
        "schema": "softwall-confirm48-gc-tail-abba-v1",
        "job": args.job,
        "host": next(iter(hosts)) if len(hosts) == 1 else sorted(hosts),
        "rows": rows,
        "gates": gates,
        "completed_rounds": completed_rounds,
        "planned_rounds": planned_rounds,
    }
    lines = [
        "# Confirm48 Python GC tail mechanism diagnostic", "",
        "| round | GC | correct | RAN misses | NRx >30 ms indices | collections | overlapping long GC events | AI units |",
        "|---:|---|---:|---:|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['condition']} | {row['correct']} | "
            f"{row['deadline_misses']} | {row['nrx_bound_violation_indices']} | "
            f"{row['gc_collections']} | {len(row['bound_violation_gc_overlaps'])} | "
            f"{row['background_units']} |"
        )
    lines.extend([
        "", f"Completed/planned rounds: {completed_rounds}/{planned_rounds}.",
        f"Prospective hypothesis gate passes: {gates['all_pass']}.",
        "The gate tests GC as a contributor to the observed tail; it does not establish a hard deadline bound.",
    ])
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        lines.extend(["", "Failed checks: " + ", ".join(failed) + "."])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    if not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
