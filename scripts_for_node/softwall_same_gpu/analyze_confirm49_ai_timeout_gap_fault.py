#!/usr/bin/env python3
"""Audit injected AI RPC timeout against a reserved fallback start."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval(data: dict) -> tuple[int, int]:
    records = data["records"]
    return (
        records[0]["release_ns"],
        max(r["release_ns"] + round(r["response_ms"] * 1e6) for r in records),
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
    fault = protocol["fault"]
    contract = protocol["controller"]
    requal = read(args.raw / f"confirm49_requalification_job{args.job}.json")
    gates: dict[str, bool] = {
        "requalification": (
            requal["iterations"] == 200
            and requal["deadline_misses"] == 0
            and requal["slurm_job_id"] == args.job
        ),
        "frozen_order": radio["order"] == ["timeout45", "timeout25", "timeout25", "timeout45"],
    }
    hosts = {requal["host"]}
    jobs = {requal["slurm_job_id"]}
    previous_end = None
    rows = []
    for round_id, name in enumerate(radio["order"], 1):
        prefix = args.raw / f"confirm49_r{round_id}_{name}_cap80_job{args.job}"
        ran = read(Path(str(prefix) + "_controller.json"))
        endpoint = read(Path(str(prefix) + "_worker.json"))
        ai = read(Path(str(prefix) + "_background.json"))
        hosts.update((ran["host"], endpoint["host"], ai["host"]))
        jobs.update((ran["slurm_job_id"], endpoint["slurm_job_id"], ai["slurm_job_id"]))
        records = ran["records"]
        first = records[0]
        label = f"r{round_id}_{name}"
        expected_timeout = contract["socket_timeouts_ms"][name]
        gates[label + "_trace"] = (
            ran["iterations"] == len(records) == radio["iterations_per_condition"]
            and ran["period_ms"] == radio["period_ms"]
            and ran["deadline_ms"] == radio["deadline_ms"]
            and ran["payload_seed"] == radio["payload_seed"]
            and ran["snr_db"] is None
            and [r["index"] for r in records] == list(range(radio["iterations_per_condition"]))
            and ran["inject_neural_failure_index"] == fault["forced_neural_failure_release_index"]
        )
        gates[label + "_provenance"] = (
            ran["policy"] == "s2_transactional"
            and ran["recovery_gap_ai"]
            and ran["nrx_bound_ms"] == contract["nrx_bound_ms"]
            and ran["conv_bound_ms"] == contract["conventional_bound_ms"]
            and ran["commit_guard_ms"] == contract["commit_guard_ms"]
            and ran["ai_budget_ms"] == contract["ai_budget_ms"]
            and ran["ai_guard_ms"] == contract["ai_guard_ms"]
            and ran["ai_rpc_timeout_ms"] == expected_timeout
            and endpoint["visible_sm_count"] == 86
            and int(ai["mps_active_thread_percentage"]) == contract["background_mps_cap"]
            and ai["response_delay_unit"] == fault["ai_response_delay_unit"]
            and ai["response_delay_ms"] == fault["ai_response_delay_ms"]
            and expected_timeout == (45 if name == "timeout45" else 25)
        )
        gates[label + "_fault_reached"] = (
            first["fallback"]
            and first["neural_correct"] is False
            and first["commit_kind"] == "conventional"
            and first["recovery_gap_ai_units"] == 0
            and first["fallback_start_ns"] - first["release_ns"]
                == round(contract["fallback_latest_start_ms_after_release"] * 1e6)
            and any(f["release_index"] == 0 for f in ran["background_faults"])
        )
        lateness = first["fallback_start_lateness_ms"]
        gates[label + "_timing_hypothesis"] = (
            isinstance(lateness, (int, float))
            and (lateness > 3 if name == "timeout45" else lateness <= 1)
        )
        gates[label + "_safety"] = (
            ran["deadline_misses"] == 0
            and ran["endpoint_timeouts"] == 0
            and ran["duplicate_commits"] == 0
            and ran["nrx_commits"] + ran["conv_commits"] == len(records)
            and ran["fallback_calendar_final"]["outstanding"] == 0
            and ran["endpoint_final"]["outstanding"] == 0
        )
        start_ns, end_ns = interval(ran)
        gates[label + "_serial"] = previous_end is None or previous_end < start_ns
        previous_end = end_ns
        rows.append({
            "round": round_id,
            "name": name,
            "socket_timeout_ms": expected_timeout,
            "fallback_start_lateness_ms": lateness,
            "first_response_ms": first["response_ms"],
            "first_correct": first["correct"],
            "deadline_misses": ran["deadline_misses"],
            "background_faults": ran["background_faults"],
        })
    gates["same_host_job"] = len(hosts) == 1 and jobs == {args.job}
    gates["all_pass"] = all(gates.values())
    report = {
        "schema": "softwall-confirm49-ai-timeout-gap-fault-v1",
        "job": args.job,
        "host": next(iter(hosts)) if len(hosts) == 1 else sorted(hosts),
        "rows": rows,
        "gates": gates,
    }
    lines = [
        "# Confirm49 stalled AI RPC versus recovery fallback reservation", "",
        "| round | socket timeout | fallback start late | first RAN response | deadline misses | AI fault detected |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['socket_timeout_ms']} ms | "
            f"{row['fallback_start_lateness_ms'] if row['fallback_start_lateness_ms'] is None else format(row['fallback_start_lateness_ms'], '.3f')} ms | "
            f"{row['first_response_ms']:.3f} ms | {row['deadline_misses']} | "
            f"{bool(row['background_faults'])} |"
        )
    lines.extend([
        "", f"Prospective fault contrast passes: {gates['all_pass']}.",
        "A late fallback start violates its reservation even when the 80 ms RAN deadline is still met.",
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
