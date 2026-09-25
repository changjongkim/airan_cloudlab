#!/usr/bin/env python3
"""Analyze two seed/order pairs of transactional external S2 against eager dual."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired(external: dict, eager: dict) -> dict:
    differences = [
        int(a["correct"]) - int(b["correct"])
        for a, b in zip(external["records"], eager["records"])
    ]
    mean = statistics.mean(differences)
    stderr = math.sqrt(statistics.variance(differences) / len(differences))
    return {
        "difference_pp": mean * 100,
        "confidence_interval_95_pp": [
            (mean - 1.96 * stderr) * 100,
            (mean + 1.96 * stderr) * 100,
        ],
        "external_only_correct": differences.count(1),
        "eager_only_correct": differences.count(-1),
    }


def interval(data: dict) -> tuple[int, int]:
    records = data["records"]
    return (
        records[0]["release_ns"],
        max(row["release_ns"] + round(row["response_ms"] * 1e6) for row in records),
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
    requal = read(args.raw / f"confirm47_requalification_job{args.job}.json")
    gates: dict[str, bool] = {
        "requalification": (
            requal["iterations"] == 200
            and requal["deadline_misses"] == 0
            and requal["slurm_job_id"] == args.job
        )
    }
    hosts = {requal["host"]}
    jobs = {requal["slurm_job_id"]}
    rows = []
    prior_end_ns = None
    for round_spec in radio["rounds"]:
        round_id = round_spec["round"]
        eager_prefix = args.raw / (
            f"confirm47_r{round_id}_eager_job{args.job}_r1_eager_dual_job{args.job}"
        )
        external_prefix = args.raw / f"confirm47_r{round_id}_external_cap80_job{args.job}"
        eager = read(Path(str(eager_prefix) + "_ran.json"))
        eager_ai = read(Path(str(eager_prefix) + "_ai.json"))
        external = read(Path(str(external_prefix) + "_controller.json"))
        endpoint = read(Path(str(external_prefix) + "_worker.json"))
        external_ai = read(Path(str(external_prefix) + "_background.json"))
        datasets = {"eager_dual": eager, "external_transaction": external}
        hosts.update(item["host"] for item in (eager, eager_ai, external, endpoint, external_ai))
        jobs.update(item["slurm_job_id"] for item in (eager, eager_ai, external, endpoint, external_ai))

        for name, data in datasets.items():
            label = f"r{round_id}_{name}"
            records = data["records"]
            gates[label + "_trace"] = (
                data["iterations"] == len(records) == radio["iterations_per_condition"]
                and data["payload_seed"] == round_spec["payload_seed"]
                and data["channel_seed_base"] == round_spec["channel_seed_base"]
                and data["period_ms"] == radio["period_ms"]
                and data["deadline_ms"] == radio["deadline_ms"]
                and data["snr_db"] == radio["snr_db"]
                and [item["index"] for item in records] == list(range(radio["iterations_per_condition"]))
                and all(item["channel_seed"] == round_spec["channel_seed_base"] + item["index"] for item in records)
            )
            gates[label + "_radio_and_ai"] = (
                data["deadline_misses"] == 0
                and data["background_budget_violations"] == 0
                and data["background_release_crossings"] == 0
                and not data["background_faults"]
            )
        gates[f"r{round_id}_same_release_trace"] = all(
            a["index"] == b["index"]
            and a["channel_seed"] == b["channel_seed"]
            for a, b in zip(external["records"], eager["records"])
        )
        gates[f"r{round_id}_background_accounting"] = (
            eager_ai["completed_units"] == eager["background_units"]
            and external_ai["completed_units"] == external["background_units"]
            and eager_ai["kind"] == external_ai["kind"] == "nrx"
            and eager_ai["mode"] == external_ai["mode"] == "rpc"
            and eager_ai["repeats"] == external_ai["repeats"] == 1
            and int(eager_ai["mps_active_thread_percentage"]) == protocol["background_both_conditions"]["mps_cap"]
            and int(external_ai["mps_active_thread_percentage"]) == protocol["background_both_conditions"]["mps_cap"]
            and eager_ai["visible_sm_count"] <= 22
            and external_ai["visible_sm_count"] <= 22
            and endpoint["visible_sm_count"] == 86
            and endpoint["completed_units"] == (
                external["endpoint_requests"] + external["warmup"]
                + external["noisy_warmup_units"]
            )
        )
        gates[f"r{round_id}_external_transaction"] = (
            external["schema"] == "softwall-same-request-transaction-controller-v1"
            and external["policy"] == "s2_transactional"
            and external["recovery_gap_ai"]
            and external["nrx_bound_ms"] == protocol["external_transaction"]["nrx_end_to_end_bound_ms"]
            and external["conv_bound_ms"] == protocol["external_transaction"]["conventional_gpu_bound_ms"]
            and external["commit_guard_ms"] == protocol["external_transaction"]["commit_guard_ms"]
            and external["endpoint_timeouts"] == 0
            and external["nrx_bound_violations"] == 0
            and external["conv_bound_violations"] == 0
            and external["duplicate_commits"] == 0
            and external["fallback_calendar_final"]["outstanding"] == 0
            and external["endpoint_final"]["outstanding"] == 0
            and external["nrx_commits"] + external["conv_commits"] == radio["iterations_per_condition"]
            and external["recovery_gap_ai_units"] > 0
        )
        comparison = paired(external, eager)
        margin = protocol["primary_gate"]["external_radio_utility_noninferior_to_eager_each_round_margin_pp"]
        gates[f"r{round_id}_utility_noninferior"] = comparison["confidence_interval_95_pp"][0] > margin
        gates[f"r{round_id}_ai_exceeds_eager"] = external["background_units"] > eager["background_units"]

        order = round_spec["order"]
        gates[f"r{round_id}_order"] = order in (
            ["eager_dual", "external_transaction"],
            ["external_transaction", "eager_dual"],
        )
        for name in order:
            start_ns, end_ns = interval(datasets[name])
            gates[f"r{round_id}_{name}_serial"] = prior_end_ns is None or prior_end_ns < start_ns
            prior_end_ns = end_ns
        rows.append({
            "round": round_id,
            "order": order,
            "eager_correct": eager["correct_releases"],
            "external_correct": external["correct_releases"],
            "eager_background_units": eager["background_units"],
            "external_background_units": external["background_units"],
            "external_recovery_gap_ai_units": external["recovery_gap_ai_units"],
            "external_admission_rejections": external["admission_rejections"],
            "eager_nrx_bound_violations_diagnostic": eager["nrx_bound_violations"],
            "external_vs_eager": comparison,
        })

    gates["same_host_job"] = len(hosts) == 1 and jobs == {args.job}
    gates["round_orders_reversed"] = (
        radio["rounds"][0]["order"] == ["eager_dual", "external_transaction"]
        and radio["rounds"][1]["order"] == ["external_transaction", "eager_dual"]
    )
    gates["all_pass"] = all(gates.values())
    report = {
        "schema": "softwall-confirm47-transaction-vs-eager-v1",
        "job": args.job,
        "host": next(iter(hosts)) if len(hosts) == 1 else sorted(hosts),
        "rows": rows,
        "gates": gates,
    }
    lines = [
        "# Confirm47 fully transactional external S2 vs eager-dual", "",
        "| round | order | eager correct | external correct | eager AI | external AI | recovery-gap AI | admission rejections | external−eager AI |",
        "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        gain = row["external_background_units"] - row["eager_background_units"]
        lines.append(
            f"| {row['round']} | {' → '.join(row['order'])} | {row['eager_correct']} | "
            f"{row['external_correct']} | {row['eager_background_units']} | "
            f"{row['external_background_units']} | {row['external_recovery_gap_ai_units']} | "
            f"{row['external_admission_rejections']} | {gain:+d} |"
        )
        ci = row["external_vs_eager"]["confidence_interval_95_pp"]
        lines.append(
            f"Round {row['round']} external−eager radio: {row['external_vs_eager']['difference_pp']:+.4f} pp, "
            f"paired 95% CI [{ci[0]:+.4f}, {ci[1]:+.4f}] pp; "
            f"eager NRx 20 ms branch-bound diagnostic {row['eager_nrx_bound_violations_diagnostic']}."
        )
    lines.extend(["", f"All frozen gates pass: {gates['all_pass']}."])
    if not gates["all_pass"]:
        lines.extend(["", "Failed checks: " + ", ".join(name for name, passed in gates.items() if not passed) + "."])
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(args.output.read_text(encoding="utf-8"), end="")
    if not gates["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
