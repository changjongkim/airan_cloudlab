#!/usr/bin/env python3
"""Evaluate the preregistered one-cell N1 crossover without retuning gates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


POLICIES = ("conventional_only", "eager_dual", "external_transaction")


def read(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def path_for(raw: Path, job: str, round_no: int, period: int, policy: str) -> Path:
    tag = "external" if policy == "external_transaction" else policy
    prefix = f"confirm51_r{round_no}_p{period}_{tag}_job{job}"
    if policy == "external_transaction":
        return raw / f"{prefix}_cap80_job{job}_controller.json"
    return raw / f"{prefix}_r1_{policy}_job{job}_ran.json"


def count_record_misses(data: dict) -> int:
    return sum(bool(item["deadline_miss"]) for item in data["records"])


def count_record_correct(data: dict) -> int:
    return sum(bool(item["correct"]) for item in data["records"])


def check_input(data: dict, round_cfg: dict, period: int, iterations: int) -> list[str]:
    errors = []
    expected = {
        "period_ms": period,
        "deadline_ms": 80,
        "iterations": iterations,
        "payload_seed": round_cfg["payload_seed"],
        "channel_seed_base": round_cfg["channel_seed_base"],
        "snr_db": -8.5,
    }
    for field, value in expected.items():
        if data.get(field) != value:
            errors.append(f"{field}={data.get(field)!r} expected {value!r}")
    if len(data["records"]) != iterations:
        errors.append("record count differs from frozen iterations")
    if data["deadline_misses"] != count_record_misses(data):
        errors.append("deadline_misses does not match records")
    if data["correct_releases"] != count_record_correct(data):
        errors.append("correct_releases does not match records")
    for index, item in enumerate(data["records"]):
        if item["index"] != index:
            errors.append(f"record index mismatch at {index}")
            break
        if item["channel_seed"] != round_cfg["channel_seed_base"] + index:
            errors.append(f"channel seed mismatch at {index}")
            break
    return errors


def ai_crossings(data: dict) -> int:
    release_by_index = {item["index"]: item for item in data["records"]}
    period_ns = round(data["period_ms"] * 1e6)
    return sum(
        item["returned_ns"] > release_by_index[item["release_index"]]["release_ns"] + period_ns
        for item in data["background_records"]
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--early-stop-after-first-round", action="store_true")
    args = parser.parse_args()
    protocol = read(args.protocol)
    radio = protocol["radio"]
    periods = radio["periods_ms"]
    rounds = radio["rounds"]
    iterations = radio["iterations_per_condition"]
    if args.early_stop_after_first_round:
        first = rounds[0]["round"]
        misses = [
            read(path_for(args.raw, args.job, first, period, "eager_dual"))["deadline_misses"]
            for period in periods
        ]
        if any(misses):
            raise ValueError("early stop requires zero eager misses at every first-round period")
        rounds = rounds[:1]
    rows = []
    all_errors = []
    results = {}
    hosts = set()
    for round_cfg in rounds:
        round_no = round_cfg["round"]
        for period in periods:
            key = (round_no, period)
            conditions = {}
            for policy in POLICIES:
                path = path_for(args.raw, args.job, round_no, period, policy)
                data = read(path)
                errors = check_input(data, round_cfg, period, iterations)
                if data.get("slurm_job_id") != args.job:
                    errors.append("job id mismatch")
                hosts.add(data.get("host"))
                if data.get("background_units") != len(data["background_records"]):
                    errors.append("background_units does not match records")
                ai_errors = sum((
                    data["background_budget_violations"],
                    data["background_release_crossings"],
                    len(data["background_faults"]),
                    ai_crossings(data),
                ))
                if ai_errors:
                    errors.append(f"AI contract failures={ai_errors}")
                if policy == "external_transaction":
                    for field in (
                        "endpoint_timeouts", "nrx_bound_violations",
                        "conv_bound_violations", "duplicate_commits",
                        "fallback_not_started", "fallback_start_late_gt_1ms",
                    ):
                        if data[field]:
                            errors.append(f"{field}={data[field]}")
                    for item in data["records"]:
                        if item["fallback"] and item["fallback_actual_start_ns"] is None:
                            errors.append("fallback actual start missing")
                            break
                        target = item["recovery_gap_ai_deadline_ns"]
                        if target is not None and target > min(
                            item["fallback_start_ns"],
                            item["release_ns"] + round(period * 1e6),
                        ):
                            errors.append("gap lease target exceeds a required boundary")
                            break
                conditions[policy] = data
                all_errors.extend(f"r{round_no} P{period} {policy}: {error}" for error in errors)
            if len({conditions[policy]["host"] for policy in POLICIES}) != 1:
                all_errors.append(f"r{round_no} P{period}: host mismatch")
            eager = conditions["eager_dual"]
            external = conditions["external_transaction"]
            radio_diff_pp = 100 * (
                external["correct_releases"] - eager["correct_releases"]
            ) / iterations
            diagnostic_pass = (
                eager["deadline_misses"] > 0
                and external["deadline_misses"] == 0
                and external["background_units"] > 0
                and radio_diff_pp >= -0.25
                and not any(
                    error.startswith(f"r{round_no} P{period} ")
                    for error in all_errors
                )
            )
            results[key] = diagnostic_pass
            rows.append(
                f"| {round_no} | {period} | "
                f"{conditions['conventional_only']['deadline_misses']} / {eager['deadline_misses']} / {external['deadline_misses']} | "
                f"{conditions['conventional_only']['correct_releases']} / {eager['correct_releases']} / {external['correct_releases']} | "
                f"{conditions['conventional_only']['background_units']} / {eager['background_units']} / {external['background_units']} | "
                f"{conditions['conventional_only']['response_ms']['p99']:.3f} / {eager['response_ms']['p99']:.3f} / {external['response_ms']['p99']:.3f} | "
                f"{80 - eager['response_ms']['p99']:.3f} | {radio_diff_pp:+.3f} | "
                f"{'PASS' if diagnostic_pass else 'FAIL'} |"
            )

    if len(hosts) != 1:
        all_errors.append(f"all conditions span multiple hosts: {sorted(hosts)}")
    replicated = [
        period for period in periods
        if all(results[(round_cfg["round"], period)] for round_cfg in rounds)
    ]
    if args.early_stop_after_first_round:
        # A replicated pass requires both frozen rounds. Zero eager misses at
        # every first-round point makes that pass mathematically impossible.
        replicated = []
    lines = [
        "# Confirm51 N1 one-cell load diagnostic",
        "",
        "Policies in each triple: conventional-only / eager-dual / external transaction.",
        "This is an operating-point diagnostic; the full combined baseline and multicell N1 grid remain untested.",
        "Evidence grade: C (one complete round; opposite-order round stopped after the diagnostic pass became impossible)."
        if args.early_stop_after_first_round else "Evidence grade: B only if a crossover repeats in both opposite-order rounds; otherwise C.",
        ("Round 2 was stopped after round 1 because no eager deadline miss at any frozen period made the two-round pass impossible."
         if args.early_stop_after_first_round else "Both frozen rounds completed."),
        "",
        "| round | period ms | misses | correct TB | completed AI units | RAN p99 ms | eager p99 slack ms | external−eager correct pp | diagnostic |",
        "|---:|---:|---|---|---|---|---:|---:|---|",
        *rows,
        "",
        f"Replicated diagnostic crossover periods: {replicated or 'none'}.",
        ("Eager failure point: not observed in the frozen one-cell grid; its distance below P12 remains unknown."
         if args.early_stop_after_first_round else "Eager failure distance must be read from the full period grid above."),
        f"Input/contract errors: {len(all_errors)}.",
    ]
    if all_errors:
        lines.extend(["", "Errors:", *(f"- {error}" for error in all_errors)])
    lines.extend([
        "",
        "Decision: " + (
            "Prioritize N2 and full combined-baseline implementation at the replicated period; no novelty claim yet."
            if replicated and not all_errors else
            "Do not add seeds or relax bounds. Complete the comparable multicell N1 path before deciding the system claim."
        ),
        "",
        "The zero-miss observations are empirical, not a worst-case guarantee.",
        "",
    ])
    args.output.write_text("\n".join(lines), encoding="utf-8")
    print(f"confirm51 replicated periods={replicated}, contract_errors={len(all_errors)}", flush=True)


if __name__ == "__main__":
    main()
