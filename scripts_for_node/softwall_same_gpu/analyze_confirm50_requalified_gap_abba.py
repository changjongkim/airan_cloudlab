#!/usr/bin/env python3
"""Audit the frozen 50 ms-contract recovery-gap ABBA comparison."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from analyze_recovery_gap_baab import paired, read


def interval(data: dict) -> tuple[int, int]:
    records = data["records"]
    return (
        records[0]["release_ns"],
        max(row["release_ns"] + round(row["response_ms"] * 1e6)
            for row in records),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = read(args.protocol)
    radio = protocol["radio"]
    endpoint_contract = protocol["endpoint"]
    recovery = protocol["recovery"]
    ai_contract = protocol["background_both_conditions"]
    requal = read(args.raw / f"confirm50_requalification_job{args.job}.json")

    expected = [
        ("off", radio["pairs"][0]),
        ("on", radio["pairs"][0]),
        ("on", radio["pairs"][1]),
        ("off", radio["pairs"][1]),
    ]
    gates: dict[str, bool] = {
        "frozen_order": (
            [pair["order"] for pair in radio["pairs"]]
            == [["gap_off", "gap_on"], ["gap_on", "gap_off"]]
        ),
        "requalification": (
            requal["iterations"] == 200
            and requal["deadline_misses"] == 0
            and requal["slurm_job_id"] == args.job
        ),
    }
    hosts = {requal["host"]}
    jobs = {requal["slurm_job_id"]}
    rows = []
    runs: list[dict] = []
    previous_end_ns = None
    for index, (mode, pair) in enumerate(expected, 1):
        prefix = args.raw / f"confirm50_r{index}_{mode}_cap80_job{args.job}"
        ran = read(Path(str(prefix) + "_controller.json"))
        endpoint = read(Path(str(prefix) + "_worker.json"))
        ai = read(Path(str(prefix) + "_background.json"))
        hosts.update((ran["host"], endpoint["host"], ai["host"]))
        jobs.update((ran["slurm_job_id"], endpoint["slurm_job_id"], ai["slurm_job_id"]))
        records = ran["records"]
        label = f"r{index}_{mode}"
        gates[label + "_trace"] = (
            ran["schema"] == "softwall-same-request-transaction-controller-v1"
            and ran["iterations"] == len(records) == radio["iterations_per_condition"]
            and ran["payload_seed"] == pair["payload_seed"]
            and ran["channel_seed_base"] == pair["channel_seed_base"]
            and ran["period_ms"] == radio["period_ms"]
            and ran["deadline_ms"] == radio["deadline_ms"]
            and ran["snr_db"] == radio["snr_db"]
            and [row["index"] for row in records]
                == list(range(radio["iterations_per_condition"]))
            and all(row["channel_seed"] == pair["channel_seed_base"] + row["index"]
                    for row in records)
        )
        gates[label + "_contract"] = (
            ran["policy"] == "s2_transactional"
            and ran["recovery_gap_ai"] == (mode == "on")
            and ran["endpoint_timeout_ms"] == endpoint_contract["rpc_timeout_ms"]
            and ran["nrx_bound_ms"] == endpoint_contract["nrx_end_to_end_bound_ms"]
            and ran["conv_bound_ms"] == recovery["conventional_gpu_bound_ms"]
            and ran["commit_guard_ms"] == recovery["commit_guard_ms"]
            and ran["ai_budget_ms"] == ai_contract["rpc_budget_ms"]
            and ran["ai_guard_ms"] == ai_contract["guard_ms"]
            and ran["ai_rpc_timeout_ms"] == ai_contract["rpc_timeout_ms"]
            and endpoint["visible_sm_count"] == 86
            and int(ai["mps_active_thread_percentage"]) == ai_contract["mps_cap"]
            and ai["visible_sm_count"] <= 22
        )
        gates[label + "_accounting"] = (
            endpoint["completed_units"]
                == ran["endpoint_requests"] + ran["warmup"] + ran["noisy_warmup_units"]
            and ai["completed_units"] == ran["background_units"]
        )
        gates[label + "_radio_and_bounds"] = (
            ran["deadline_misses"] == 0
            and ran["endpoint_timeouts"] == 0
            and ran["nrx_bound_violations"] == 0
            and ran["conv_bound_violations"] == 0
        )
        gates[label + "_ai"] = (
            ran["background_budget_violations"] == 0
            and ran["background_release_crossings"] == 0
            and not ran["background_faults"]
        )
        gates[label + "_transaction"] = (
            ran["duplicate_commits"] == 0
            and ran["nrx_commits"] + ran["conv_commits"] == radio["iterations_per_condition"]
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["endpoint_final"]["outstanding"] == 0
        )
        start_ns, end_ns = interval(ran)
        gates[label + "_serial"] = previous_end_ns is None or previous_end_ns < start_ns
        previous_end_ns = end_ns
        runs.append(ran)
        rows.append({
            "round": index,
            "pair": pair["pair"],
            "mode": mode,
            "correct": ran["correct_releases"],
            "ran_misses": ran["deadline_misses"],
            "nrx_bound_violations": ran["nrx_bound_violations"],
            "completed_ai": ran["background_units"],
            "gap_ai": ran["recovery_gap_ai_units"],
            "admission_rejections": ran["admission_rejections"],
        })

    gates["same_host_job"] = len(hosts) == 1 and jobs == {args.job}
    gates["paired_trace"] = all(
        runs[left]["records"][index]["channel_seed"]
            == runs[right]["records"][index]["channel_seed"]
        for left, right in ((0, 1), (2, 3))
        for index in range(radio["iterations_per_condition"])
    )
    gates["gap_switch"] = (
        rows[0]["gap_ai"] == rows[3]["gap_ai"] == 0
        and rows[1]["gap_ai"] > 0
        and rows[2]["gap_ai"] > 0
    )
    pair_results = []
    for pair_id, (on_index, off_index) in enumerate(((1, 0), (2, 3)), 1):
        on = runs[on_index]
        off = runs[off_index]
        radio_result = paired(on, off)
        gain = on["background_units"] - off["background_units"]
        pair_results.append({
            "pair": pair_id,
            "on_minus_off_ai_units": gain,
            "on_minus_off_ai_percent": 100 * gain / off["background_units"],
            "on_gap_ai_units": on["recovery_gap_ai_units"],
            "on_minus_off_outside_gap_ai_units": (
                on["background_units"] - on["recovery_gap_ai_units"]
                - off["background_units"]
            ),
            "on_minus_off_admission_rejections": (
                on["admission_rejections"] - off["admission_rejections"]
            ),
            "on_minus_off_radio": radio_result,
        })
        gates[f"pair{pair_id}_radio_noninferior"] = (
            radio_result["confidence_interval_95_pp"][0]
            > protocol["primary_gate"][
                "each_gap_on_radio_utility_noninferior_to_paired_off_margin_pp"]
        )
        gates[f"pair{pair_id}_ai_gain"] = gain > 0

    gates["all_pass"] = all(gates.values())
    report = {
        "schema": "softwall-confirm50-requalified-gap-abba-v1",
        "job": args.job,
        "host": next(iter(hosts)) if len(hosts) == 1 else sorted(hosts),
        "rows": rows,
        "pairs": pair_results,
        "gates": gates,
    }
    lines = [
        "# Confirm50 requalified 50 ms recovery-gap ABBA ablation", "",
        "| round | pair | gap | correct | RAN misses | NRx bound violations | AI units | gap AI | admission rejects |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['pair']} | {row['mode']} | "
            f"{row['correct']} | {row['ran_misses']} | {row['nrx_bound_violations']} | "
            f"{row['completed_ai']} | {row['gap_ai']} | {row['admission_rejections']} |"
        )
    lines.append("")
    for item in pair_results:
        lo, hi = item["on_minus_off_radio"]["confidence_interval_95_pp"]
        lines.append(
            f"Pair {item['pair']}: ON−OFF AI {item['on_minus_off_ai_units']:+d} "
            f"({item['on_minus_off_ai_percent']:+.3f}%; gap "
            f"{item['on_gap_ai_units']}, outside-gap change "
            f"{item['on_minus_off_outside_gap_ai_units']:+d}), "
            f"admission rejection change "
            f"{item['on_minus_off_admission_rejections']:+d}, radio paired 95% CI "
            f"[{lo:+.4f}, {hi:+.4f}] pp."
        )
    lines.extend(["", f"All frozen gates pass: {gates['all_pass']}."])
    failed = [name for name, passed in gates.items() if not passed]
    if failed:
        lines.extend(["", "Failed checks: " + ", ".join(failed) + "."])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")
    if not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
