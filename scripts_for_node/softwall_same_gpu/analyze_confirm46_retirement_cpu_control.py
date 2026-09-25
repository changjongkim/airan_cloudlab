#!/usr/bin/env python3
"""Audit the frozen same-seed CPU-sham versus MPS-client retirement control."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def interval(data: dict) -> tuple[int, int]:
    rows = data["records"]
    return (
        rows[0]["release_ns"],
        max(r["release_ns"] + round(r["response_ms"] * 1e6) for r in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True, type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    protocol = read(args.protocol)
    radio = protocol["radio"]
    requal = read(args.raw / f"confirm46_requalification_job{args.job}.json")
    gates = {
        "requalification": (
            requal["iterations"] == 200
            and requal["deadline_misses"] == 0
            and requal["slurm_job_id"] == args.job
        )
    }
    hosts = {requal["host"]}
    jobs = {requal["slurm_job_id"]}
    rows = []
    mps_only = 0
    cpu_only = 0
    positive_rounds = 0
    negative_rounds = 0
    prior_end = None
    for spec in protocol["execution"]["rounds"]:
        rnd = spec["round"]
        data = {}
        for condition in ("cpu_sham", "mps_sham"):
            prefix = args.raw / f"confirm46_r{rnd}_{condition}_job{args.job}"
            ran = read(Path(str(prefix) + "_ran.json"))
            worker = read(Path(str(prefix) + "_worker.json"))
            data[condition] = ran
            hosts.update((ran["host"], worker["host"]))
            jobs.update((ran["slurm_job_id"], worker["slurm_job_id"]))
            records = ran["records"]
            gates[f"r{rnd}_{condition}_contract"] = (
                ran["iterations"] == len(records) == radio["iterations_per_condition"]
                and ran["seed"] == spec["seed"]
                and ran["period_ms"] == radio["period_ms"]
                and ran["deadline_ms"] == radio["deadline_ms"]
                and ran["retire_before_first_release"]
                and ran["endpoint_retired_at_release"] == -1
                and ran["endpoint_retired"]
                and ran["injected_overruns"] == 0
                and ran["timeouts_detected"] == 0
                and ran["worker_active_releases"] == 0
                and ran["correct_releases"] == radio["iterations_per_condition"]
                and [r["index"] for r in records] == list(range(radio["iterations_per_condition"]))
                and ran["deadline_misses"] == sum(r["deadline_miss"] for r in records)
                and worker["completed_units"] == 0
            )
            gates[f"r{rnd}_{condition}_worker"] = (
                worker["schema"] == (
                    "softwall-cpu-sham-worker-v1"
                    if condition == "cpu_sham" else "softwall-ai-unit-v1"
                )
                and (
                    worker.get("cuda_context_created") is False
                    if condition == "cpu_sham" else
                    int(worker["mps_active_thread_percentage"]) == protocol["worker"]["mps_cap"]
                    and worker["visible_sm_count"] <= 22
                )
            )
        gates[f"r{rnd}_same_input"] = (
            data["cpu_sham"]["seed"] == data["mps_sham"]["seed"]
            and len(data["cpu_sham"]["records"]) == len(data["mps_sham"]["records"])
        )
        for condition in spec["order"]:
            start, end = interval(data[condition])
            gates[f"r{rnd}_{condition}_serial"] = prior_end is None or prior_end < start
            prior_end = end
        cpu = data["cpu_sham"]
        mps = data["mps_sham"]
        for a, b in zip(cpu["records"], mps["records"]):
            mps_only += int(b["deadline_miss"] and not a["deadline_miss"])
            cpu_only += int(a["deadline_miss"] and not b["deadline_miss"])
        positive_rounds += int(mps["deadline_misses"] > cpu["deadline_misses"])
        negative_rounds += int(mps["deadline_misses"] < cpu["deadline_misses"])
        rows.append({
            "round": rnd,
            "seed": spec["seed"],
            "order": spec["order"],
            "cpu_misses": cpu["deadline_misses"],
            "mps_misses": mps["deadline_misses"],
            "cpu_miss_indices": [r["index"] for r in cpu["records"] if r["deadline_miss"]],
            "mps_miss_indices": [r["index"] for r in mps["records"] if r["deadline_miss"]],
        })
    discordant_rounds = positive_rounds + negative_rounds
    one_sided_p = sum(
        math.comb(discordant_rounds, k)
        for k in range(positive_rounds, discordant_rounds + 1)
    ) / (2 ** discordant_rounds)
    gates["same_host_job"] = len(hosts) == 1 and jobs == {args.job}
    gates["position_balanced_order"] = [r["order"] for r in rows] == [
        ["cpu_sham", "mps_sham"] if round_id % 2 else ["mps_sham", "cpu_sham"]
        for round_id in range(1, 11)
    ]
    gates["mps_excess_exact"] = one_sided_p < protocol["primary_gate"]["mps_retirement_excess_misses_one_sided_round_sign_p_below"]
    gates["all_pass"] = all(gates.values())
    report = {
        "schema": "softwall-confirm46-retirement-cpu-control-v1",
        "job": args.job,
        "host": next(iter(hosts)) if len(hosts) == 1 else sorted(hosts),
        "rows": rows,
        "mps_only_misses": mps_only,
        "cpu_only_misses": cpu_only,
        "positive_mps_excess_rounds": positive_rounds,
        "negative_mps_excess_rounds": negative_rounds,
        "one_sided_exact_p": one_sided_p,
        "gates": gates,
    }
    lines = [
        "# Confirm46 CPU-sham versus MPS-client retirement control", "",
        "| round | seed | order | CPU-sham misses | MPS-sham misses | CPU indices | MPS indices |",
        "|---:|---:|---|---:|---:|---|---|",
    ]
    for row in rows:
        lines.append(
            f"| {row['round']} | {row['seed']} | {' → '.join(row['order'])} | "
            f"{row['cpu_misses']} | {row['mps_misses']} | "
            f"{row['cpu_miss_indices']} | {row['mps_miss_indices']} |"
        )
    lines.extend([
        "", f"Paired discordant misses: MPS-only {mps_only}, CPU-only {cpu_only}. "
        f"Round-level sign test: MPS excess {positive_rounds}, CPU excess {negative_rounds}, "
        f"one-sided exact p = {one_sided_p:.6g}.",
        f"All frozen gates pass: {gates['all_pass']}.",
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
