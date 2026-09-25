#!/usr/bin/env python3
"""Evaluate the frozen same-node ABBA recovery-gap lease ablation."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(on: dict, off: dict) -> dict:
    diffs = [
        int(a["correct"]) - int(b["correct"])
        for a, b in zip(on["records"], off["records"])
    ]
    mean = statistics.mean(diffs)
    stderr = math.sqrt(statistics.variance(diffs) / len(diffs))
    return {
        "difference_pp": mean * 100,
        "confidence_interval_95_pp": [
            (mean - 1.96 * stderr) * 100,
            (mean + 1.96 * stderr) * 100,
        ],
        "on_only_correct": diffs.count(1),
        "off_only_correct": diffs.count(-1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    radio = protocol["radio"]
    requal = read(args.raw / f"confirm43_requalification_job{args.job}.json")
    conditions: list[tuple[str, dict, dict, dict]] = []
    for index, mode in enumerate(("off", "on", "on", "off"), start=1):
        prefix = args.raw / f"confirm43_gap_r{index}_{mode}_cap80_job{args.job}"
        controller = read(Path(str(prefix) + "_controller.json"))
        endpoint = read(Path(str(prefix) + "_worker.json"))
        background = read(Path(str(prefix) + "_background.json"))
        conditions.append((mode, controller, endpoint, background))

    gates: dict[str, bool] = {}
    gates["requalification"] = (
        requal["iterations"] == 200
        and requal["deadline_misses"] == 0
        and requal["slurm_job_id"] == args.job
    )
    hosts = {requal["host"]}
    jobs = {requal["slurm_job_id"]}
    prior_end_ns = None
    for index, (mode, data, endpoint, background) in enumerate(conditions, start=1):
        label = f"r{index}_{mode}"
        hosts.update((data["host"], endpoint["host"], background["host"]))
        jobs.update((data["slurm_job_id"], endpoint["slurm_job_id"], background["slurm_job_id"]))
        records = data["records"]
        gates[f"{label}_trace"] = (
            data["iterations"] == len(records) == radio["iterations_per_condition"]
            and data["payload_seed"] == radio["payload_seed"]
            and data["channel_seed_base"] == radio["channel_seed_base"]
            and data["period_ms"] == radio["period_ms"]
            and data["deadline_ms"] == radio["deadline_ms"]
            and data["snr_db"] == radio["snr_db"]
            and [row["index"] for row in records] == list(range(radio["iterations_per_condition"]))
            and all(row["channel_seed"] == radio["channel_seed_base"] + row["index"] for row in records)
        )
        gates[f"{label}_transaction"] = (
            data["schema"] == "softwall-same-request-transaction-controller-v1"
            and data["policy"] == "s2_transactional"
            and data["recovery_gap_ai"] == (mode == "on")
            and data["nrx_bound_ms"] == protocol["endpoint"]["nrx_end_to_end_bound_ms"]
            and data["conv_bound_ms"] == protocol["recovery"]["conventional_gpu_bound_ms"]
            and data["commit_guard_ms"] == protocol["recovery"]["commit_guard_ms"]
            and data["ai_budget_ms"] == protocol["background"]["rpc_budget_ms"]
            and data["ai_guard_ms"] == protocol["background"]["guard_ms"]
        )
        gates[f"{label}_worker_accounting"] = (
            endpoint["completed_units"] == data["endpoint_requests"] + data["warmup"] + data["noisy_warmup_units"]
            and endpoint["visible_sm_count"] == 86
            and background["completed_units"] == data["background_units"]
            and int(background["mps_active_thread_percentage"]) == protocol["background"]["mps_cap"]
            and background["visible_sm_count"] <= 22
        )
        gates[f"{label}_radio_and_bounds"] = (
            data["deadline_misses"] == 0
            and data["endpoint_timeouts"] == 0
            and data["nrx_bound_violations"] == 0
            and data["conv_bound_violations"] == 0
        )
        gates[f"{label}_ai_contract"] = (
            data["background_budget_violations"] == 0
            and data["background_release_crossings"] == 0
            and not data["background_faults"]
        )
        gates[f"{label}_commit_and_credits"] = (
            data["duplicate_commits"] == 0
            and data["nrx_commits"] + data["conv_commits"] == radio["iterations_per_condition"]
            and data["fallback_calendar_final"]["outstanding"] == 0
            and data["endpoint_final"]["outstanding"] == 0
        )
        start_ns = records[0]["release_ns"]
        end_ns = max(row["release_ns"] + round(row["response_ms"] * 1e6) for row in records)
        gates[f"{label}_serial"] = prior_end_ns is None or prior_end_ns < start_ns
        prior_end_ns = end_ns

    gates["same_host_job"] = len(hosts) == 1 and jobs == {args.job}
    gates["same_release_trace"] = all(
        len({data["records"][index]["channel_seed"] for _, data, _, _ in conditions}) == 1
        for index in range(radio["iterations_per_condition"])
    )
    off_runs = [data for mode, data, _, _ in conditions if mode == "off"]
    on_runs = [data for mode, data, _, _ in conditions if mode == "on"]
    gates["gap_unit_switch"] = (
        all(data["recovery_gap_ai_units"] == 0 for data in off_runs)
        and all(data["recovery_gap_ai_units"] > 0 for data in on_runs)
    )
    # In ABBA order, match each ON run to its closest OFF run.
    pairs = [paired(conditions[1][1], conditions[0][1]), paired(conditions[2][1], conditions[3][1])]
    margin = protocol["primary_gate"]["each_gap_on_radio_utility_noninferior_to_matched_off_margin_pp"]
    gates["both_radio_pairs_noninferior"] = all(
        item["confidence_interval_95_pp"][0] > margin for item in pairs
    )
    off_units = sum(data["background_units"] for data in off_runs)
    on_units = sum(data["background_units"] for data in on_runs)
    gates["gap_on_ai_gain"] = on_units > off_units
    gates["all_pass"] = all(gates.values())
    result = {
        "schema": "softwall-confirm43-recovery-gap-abba-v1",
        "job": args.job,
        "host": next(iter(hosts)) if len(hosts) == 1 else sorted(hosts),
        "rows": [
            {
                "round": index,
                "mode": mode,
                "correct": data["correct_releases"],
                "ran_misses": data["deadline_misses"],
                "natural_fallbacks": data["fallbacks"],
                "background_units": data["background_units"],
                "recovery_gap_ai_units": data["recovery_gap_ai_units"],
            }
            for index, (mode, data, _, _) in enumerate(conditions, start=1)
        ],
        "on_minus_off_completed_ai_units": on_units - off_units,
        "on_minus_off_completed_ai_percent": (on_units / off_units - 1) * 100,
        "matched_radio_pairs": pairs,
        "gates": gates,
    }
    lines = [
        "# Confirm43 transactional recovery-gap AI ABBA ablation", "",
        "| round | gap lease | correct | RAN misses | natural fallbacks | completed AI | in recovery gap |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in result["rows"]:
        lines.append(
            f"| {row['round']} | {row['mode']} | {row['correct']} | {row['ran_misses']} | "
            f"{row['natural_fallbacks']} | {row['background_units']} | {row['recovery_gap_ai_units']} |"
        )
    lines.extend([
        "",
        f"Gap ON minus OFF: {on_units - off_units} completed AI units ({result['on_minus_off_completed_ai_percent']:+.3f}%).",
        f"Matched ON-minus-OFF radio paired 95% CI: "
        + "; ".join(
            f"[{item['confidence_interval_95_pp'][0]:+.4f}, {item['confidence_interval_95_pp'][1]:+.4f}] pp"
            for item in pairs
        ) + ".",
        f"All frozen gates pass: {gates['all_pass']}.",
    ])
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        lines.extend(["", "Failed checks: " + ", ".join(failed) + "."])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    if not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
