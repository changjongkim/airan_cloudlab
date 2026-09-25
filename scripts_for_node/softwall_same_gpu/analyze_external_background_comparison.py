#!/usr/bin/env python3
"""Trace-matched external S2 versus local S2/eager with bounded background AI."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def paired_difference(left: dict, right: dict) -> dict:
    differences = [
        int(a["correct"]) - int(b["correct"])
        for a, b in zip(left["records"], right["records"])
    ]
    count = len(differences)
    mean = statistics.mean(differences)
    stderr = math.sqrt(statistics.variance(differences) / count)
    return {
        "difference_pp": mean * 100,
        "confidence_interval_95_pp": [
            (mean - 1.96 * stderr) * 100,
            (mean + 1.96 * stderr) * 100,
        ],
        "left_only_correct": differences.count(1),
        "right_only_correct": differences.count(-1),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--external-campaign", default="confirm38_external_background_main"
    )
    parser.add_argument(
        "--local-campaign", default="confirm38_local_background_baselines"
    )
    args = parser.parse_args()
    protocol = read(args.protocol)
    external_prefix = args.raw / (
        args.external_campaign + "_cap80_job" + args.job
    )
    local_prefix = args.local_campaign + "_r1_"
    external = read(Path(str(external_prefix) + "_controller.json"))
    endpoint = read(Path(str(external_prefix) + "_worker.json"))
    external_ai = read(Path(str(external_prefix) + "_background.json"))
    local = {}
    local_ai = {}
    for policy in ("s2_reserved", "eager_dual"):
        prefix = args.raw / (local_prefix + policy + "_job" + args.job)
        local[policy] = read(Path(str(prefix) + "_ran.json"))
        local_ai[policy] = read(Path(str(prefix) + "_ai.json"))

    conditions = {"external_s2": external, **local}
    radio = protocol["radio"]
    expected = {
        "iterations": radio["iterations"],
        "period_ms": radio["period_ms"],
        "deadline_ms": radio["deadline_ms"],
        "snr_db": radio["snr_db"],
        "payload_seed": radio["payload_seed"],
        "channel_seed_base": radio["channel_seed_base"],
    }
    gates = {}
    for name, data in conditions.items():
        gates[name + "_radio_matches"] = all(
            data.get(key) == value for key, value in expected.items()
        )
        gates[name + "_records_complete"] = (
            len(data["records"]) == radio["iterations"]
        )
        gates[name + "_deadline_misses_zero"] = data["deadline_misses"] == 0
        gates[name + "_background_faults_zero"] = not data["background_faults"]
        gates[name + "_background_budget_violations_zero"] = (
            data["background_budget_violations"] == 0
        )
        gates[name + "_background_crossings_zero"] = (
            data["background_release_crossings"] == 0
        )
    hosts = {data["host"] for data in conditions.values()}
    jobs = {data["slurm_job_id"] for data in conditions.values()}
    gates["same_host_and_job"] = (
        len(hosts) == len(jobs) == 1 and jobs == {args.job}
    )
    gates["trace_matches_release_by_release"] = all(
        len({
            data["records"][index]["channel_seed"]
            for data in conditions.values()
        }) == 1
        and len({
            data["records"][index]["index"]
            for data in conditions.values()
        }) == 1
        for index in range(radio["iterations"])
    )
    gates["external_endpoint_no_timeout"] = external["endpoint_timeouts"] == 0
    gates["external_endpoint_units_match"] = endpoint["completed_units"] == (
        external["warmup"] + external["noisy_warmup_units"]
        + external["iterations"]
    )
    workers = {"external_s2": external_ai, **local_ai}
    for name, worker in workers.items():
        gates[name + "_background_worker_units_match"] = (
            worker["completed_units"] == conditions[name]["background_units"]
        )
        gates[name + "_background_cap_effective"] = (
            worker["visible_sm_count"] <= 22
        )
        gates[name + "_worker_provenance_matches"] = (
            worker["host"] == conditions[name]["host"]
            and worker["slurm_job_id"] == conditions[name]["slurm_job_id"]
        )
    gates["external_background_positive"] = external["background_units"] > 0
    gates["external_natural_recoveries_positive"] = any(
        item.get("neural_correct") is False
        and item.get("conventional_correct") is True
        for item in external["records"]
    )

    external_vs_eager = paired_difference(external, local["eager_dual"])
    external_vs_local_s2 = paired_difference(external, local["s2_reserved"])
    gates["external_utility_noninferior_to_eager"] = (
        external_vs_eager["confidence_interval_95_pp"][0]
        > protocol["primary_gate"]["utility_margin_pp"]
    )
    gates["external_background_exceeds_eager"] = (
        external["background_units"] > local["eager_dual"]["background_units"]
    )
    gates["all_pass"] = all(gates.values())
    duration_s = radio["iterations"] * radio["period_ms"] / 1000
    result = {
        "schema": "softwall-external-background-comparison-v1",
        "campaign": protocol["campaign"],
        "host": external["host"],
        "job": args.job,
        "releases": radio["iterations"],
        "correct": {name: data["correct_releases"] for name, data in conditions.items()},
        "deadline_misses": {name: data["deadline_misses"] for name, data in conditions.items()},
        "background_units": {name: data["background_units"] for name, data in conditions.items()},
        "background_units_per_s": {
            name: data["background_units"] / duration_s
            for name, data in conditions.items()
        },
        "external_vs_eager": external_vs_eager,
        "external_vs_local_s2": external_vs_local_s2,
        "gates": gates,
        "protocol": str(args.protocol),
    }
    lines = [
        "# Same-trace external S2 with bounded background AI",
        "",
        "| condition | correct | RAN misses | background units | background units/s |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, data in conditions.items():
        lines.append(
            f"| {name} | {data['correct_releases']} | "
            f"{data['deadline_misses']} | {data['background_units']} | "
            f"{data['background_units'] / duration_s:.3f} |"
        )
    for name, comparison in (
        ("eager", external_vs_eager),
        ("local S2", external_vs_local_s2),
    ):
        ci = comparison["confidence_interval_95_pp"]
        lines.append(
            f"External minus {name} correct: {comparison['difference_pp']:.4f} pp; "
            f"95% paired CI [{ci[0]:.4f}, {ci[1]:.4f}] pp."
        )
    lines.extend(["", f"All frozen gates pass: {gates['all_pass']}."])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
