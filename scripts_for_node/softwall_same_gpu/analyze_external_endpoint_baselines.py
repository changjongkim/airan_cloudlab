#!/usr/bin/env python3
"""Trace-matched utility comparison for external S2 and local baselines."""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", type=Path, required=True)
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--external-controller", type=Path, required=True)
    parser.add_argument("--external-worker", type=Path, required=True)
    parser.add_argument("--external-manifest", type=Path)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--margin-pp", type=float, default=0.25)
    args = parser.parse_args()
    sources = {}
    for policy in ("conventional_only", "eager_dual"):
        matches = sorted(args.raw.glob(f"{args.campaign}_{policy}_job*.json"))
        if len(matches) != 1:
            raise SystemExit(f"expected one {policy} result, found {len(matches)}")
        sources[policy] = json.loads(matches[0].read_text(encoding="utf-8"))
    external = json.loads(args.external_controller.read_text(encoding="utf-8"))
    worker = json.loads(args.external_worker.read_text(encoding="utf-8"))
    manifest = (
        json.loads(args.external_manifest.read_text(encoding="utf-8"))
        if args.external_manifest else None
    )
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    conventional = sources["conventional_only"]
    eager = sources["eager_dual"]
    if protocol.get("campaign") != args.campaign:
        raise SystemExit("comparison protocol campaign does not match")
    if manifest and protocol.get("source_endpoint") != manifest.get("campaign"):
        raise SystemExit("external manifest campaign does not match protocol")
    radio = protocol["radio"]
    fields = ("iterations", "period_ms", "deadline_ms", "snr_db")
    for field in fields:
        values = [conventional.get(field), eager.get(field), external.get(field)]
        if len(set(values)) != 1:
            raise SystemExit(f"unmatched {field}: {values}")
    if len({conventional.get("host"), eager.get("host"), external.get("host")}) != 1:
        raise SystemExit("conditions were not run on the same host")
    if len({
        conventional.get("slurm_job_id"),
        eager.get("slurm_job_id"),
        external.get("slurm_job_id"),
    }) != 1:
        raise SystemExit("conditions were not run in the same Slurm allocation")
    if worker.get("host") != external.get("host"):
        raise SystemExit("external worker and controller hosts do not match")
    if worker.get("slurm_job_id") != external.get("slurm_job_id"):
        raise SystemExit("external worker and controller allocations do not match")
    if conventional.get("policy") != "conventional_only":
        raise SystemExit("conventional input has the wrong policy")
    if eager.get("policy") != "eager_dual":
        raise SystemExit("eager input has the wrong policy")
    if conventional.get("cells") != 1 or eager.get("cells") != 1:
        raise SystemExit("local baselines must use exactly one cell")
    if any(item.get("payload_seed") != radio["payload_seed"] for item in (conventional, eager)):
        raise SystemExit("local payload seed does not match frozen protocol")
    if external.get("payload_seed") not in (None, radio["payload_seed"]):
        raise SystemExit("external payload seed does not match frozen protocol")
    if any(
        item.get("channel_seed_base") != radio["channel_seed_base"]
        for item in (conventional, eager, external)
    ):
        raise SystemExit("channel seed base does not match frozen protocol")
    if manifest:
        controller_cmd = manifest["processes"]["controller"]["cmdline"]
        def command_value(flag: str) -> str:
            try:
                return controller_cmd[controller_cmd.index(flag) + 1]
            except (ValueError, IndexError) as error:
                raise SystemExit(
                    f"missing {flag} in captured external command"
                ) from error
        if int(command_value("--seed")) != radio["payload_seed"]:
            raise SystemExit("captured external payload seed does not match protocol")
        if int(command_value("--channel-seed-base")) != radio["channel_seed_base"]:
            raise SystemExit("captured external channel seed base does not match protocol")
        if Path(command_value("--output")).resolve() != args.external_controller.resolve():
            raise SystemExit("captured external output does not match analyzed file")
    elif external.get("payload_seed") != radio["payload_seed"]:
        raise SystemExit(
            "external raw result lacks matching payload seed and no manifest was supplied"
        )
    count = external["iterations"]
    if any(len(item["records"]) != count for item in (conventional, eager, external)):
        raise SystemExit("record count mismatch")
    differences = []
    seed_mismatches = 0
    external_wins = 0
    eager_wins = 0
    for conv_record, eager_record, external_record in zip(
        conventional["records"], eager["records"], external["records"]
    ):
        indices = {
            conv_record.get("index"),
            eager_record.get("index"),
            external_record.get("index"),
        }
        if len(indices) != 1:
            raise SystemExit(f"release index mismatch: {sorted(indices)}")
        seeds = {
            conv_record.get("channel_seed"),
            eager_record.get("channel_seed"),
            external_record.get("channel_seed"),
        }
        seed_mismatches += int(len(seeds) != 1)
        difference = int(external_record["correct"]) - int(eager_record["correct"])
        differences.append(difference)
        external_wins += int(difference > 0)
        eager_wins += int(difference < 0)
    mean = statistics.mean(differences)
    variance = statistics.variance(differences) if count > 1 else 0.0
    standard_error = math.sqrt(variance / count)
    difference_pp = mean * 100.0
    ci = [
        (mean - 1.96 * standard_error) * 100.0,
        (mean + 1.96 * standard_error) * 100.0,
    ]
    recoveries = sum(
        item.get("neural_correct") is False
        and item.get("conventional_correct") is True
        for item in external["records"]
    )
    expected_worker_units = (
        external["warmup"]
        + external.get("noisy_warmup_units", 0)
        + external["iterations"]
    )
    gates = {
        "channel_seeds_match": seed_mismatches == 0,
        "deadline_misses_zero": (
            conventional["deadline_misses"]
            + eager["deadline_misses"]
            + external["deadline_misses"]
        ) == 0,
        "external_natural_recoveries_positive": recoveries > 0,
        "external_endpoint_timeouts_zero": external["endpoint_timeouts"] == 0,
        "external_endpoint_requests_all": (
            external.get("endpoint_requests") == external["iterations"]
        ),
        "external_conventional_accounting": (
            external.get("conventional_runs") == external["fallbacks"]
        ),
        "external_worker_units_match": (
            worker["completed_units"] == expected_worker_units
        ),
        "external_worker_visible_sms_match": (
            worker["visible_sm_count"]
            == protocol["endpoint"]["expected_visible_sms"]
        ),
        "external_utility_noninferior": ci[0] > -args.margin_pp,
        "external_uses_fewer_conventional_executions": (
            external["fallbacks"] < eager["iterations"]
        ),
    }
    gates["all_pass"] = all(gates.values())
    result = {
        "schema": "softwall-external-endpoint-baseline-v1",
        "campaign": args.campaign,
        "releases": count,
        "correct": {
            "conventional": conventional["correct_releases"],
            "eager": eager["correct_releases"],
            "external": external["correct_releases"],
        },
        "deadline_misses": {
            "conventional": conventional["deadline_misses"],
            "eager": eager["deadline_misses"],
            "external": external["deadline_misses"],
        },
        "external_fallbacks": external["fallbacks"],
        "external_recoveries": recoveries,
        "external_endpoint_requests": external.get("endpoint_requests"),
        "external_endpoint_timeouts": external["endpoint_timeouts"],
        "external_worker": {
            "completed_units": worker["completed_units"],
            "expected_units": expected_worker_units,
            "visible_sm_count": worker["visible_sm_count"],
            "gpu_ms": worker["gpu_ms"],
        },
        "external_minus_eager_percentage_points": difference_pp,
        "confidence_interval_95_percentage_points": ci,
        "external_wins": external_wins,
        "eager_wins": eager_wins,
        "channel_seed_mismatches": seed_mismatches,
        "trace_provenance": {
            "frozen_protocol": str(args.protocol),
            "external_execution_manifest": (
                str(args.external_manifest) if args.external_manifest else None
            ),
            "payload_seed": radio["payload_seed"],
            "channel_seed_base": radio["channel_seed_base"],
            "host": external.get("host"),
            "slurm_job_id": external.get("slurm_job_id"),
        },
        "gates": gates,
    }
    lines = [
        "# External same-request endpoint vs local baselines",
        "",
        "| condition | releases | correct | deadline miss | conventional executions |",
        "|---|---:|---:|---:|---:|",
        f"| conventional | {count} | {conventional['correct_releases']} | "
        f"{conventional['deadline_misses']} | {count} |",
        f"| eager dual | {count} | {eager['correct_releases']} | "
        f"{eager['deadline_misses']} | {count} |",
        f"| external S2 | {count} | {external['correct_releases']} | "
        f"{external['deadline_misses']} | {external['fallbacks']} |",
        "",
        f"External−eager: {difference_pp:.4f} pp; 95% CI "
        f"[{ci[0]:.4f}, {ci[1]:.4f}]; margin −{args.margin_pp:.3f} pp.",
        f"Natural recoveries: {recoveries}; external/eager wins: "
        f"{external_wins}/{eager_wins}.",
        f"All frozen gates pass: {gates['all_pass']}.",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines) + "\n", encoding="utf-8")
    args.output.with_suffix(".json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(args.output.read_text(encoding="utf-8"), end="")


if __name__ == "__main__":
    main()
