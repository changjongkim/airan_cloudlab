#!/usr/bin/env python3
"""Aggregate C137 on-path broker RPC telemetry qualification arms."""

import argparse
import json
from collections import Counter
from pathlib import Path

from analyze_confirm136_v12_requalification import analyze as analyze_base


REQUIRED_OPERATIONS = ("prepare", "commit", "complete")


def aggregate_rpc_evidence(arm_results, timeout_ms):
    operation_counts = Counter()
    operation_maxima = {}
    records = 0
    faulted = 0
    wall_timeout_exceeded = 0
    every_arm_covered = True
    for result in arm_results:
        evidence = result.get("control_evidence", {})
        counts = evidence.get("by_operation", {})
        every_arm_covered = every_arm_covered and all(
            int(counts.get(operation, 0)) > 0 for operation in REQUIRED_OPERATIONS
        )
        records += int(evidence.get("records", 0))
        faulted += int(evidence.get("faulted", 0))
        wall_timeout_exceeded += int(evidence.get("wall_timeout_exceeded", 0))
        for home in result.get("homes", []):
            telemetry = home.get("rpc_telemetry") or {}
            for record in telemetry.get("records", []):
                operation = record["operation"]
                elapsed = float(record["elapsed_ms"])
                operation_counts[operation] += 1
                operation_maxima[operation] = max(
                    elapsed, operation_maxima.get(operation, elapsed)
                )
    maximum = max(operation_maxima.values()) if operation_maxima else None
    gates = {
        "every_arm_prepare_commit_complete_coverage": every_arm_covered,
        "rpc_record_counts_reconcile": records == sum(operation_counts.values()),
        "no_rpc_fault": faulted == 0,
        "no_wall_timeout_exceedance": wall_timeout_exceeded == 0,
        "maximum_within_declared_timeout": maximum is not None and maximum <= timeout_ms,
    }
    return {
        "declared_per_rpc_bound_ms": timeout_ms,
        "records": records,
        "by_operation": dict(sorted(operation_counts.items())),
        "max_elapsed_ms_by_operation": dict(sorted(operation_maxima.items())),
        "maximum_elapsed_ms": maximum,
        "faulted": faulted,
        "wall_timeout_exceeded": wall_timeout_exceeded,
        "gates": gates,
        "all_pass": all(gates.values()),
    }


def analyze(campaign_path, arm_paths):
    result = analyze_base(campaign_path, arm_paths)
    arm_results = [json.loads(Path(path).resolve().read_text()) for path in arm_paths]
    campaign = json.loads(Path(campaign_path).resolve().read_text())
    control = aggregate_rpc_evidence(
        arm_results, float(campaign["global_broker_rpc_timeout_ms"])
    )
    result["schema"] = "softwall-confirm137-service-bound-telemetry-v1"
    result["control_evidence"] = control
    result["gates"]["control_rpc_telemetry"] = control["all_pass"]
    result["all_pass"] = all(result["gates"].values())
    result["claim_boundary"] = (
        "On-path wall-clock telemetry for prepare/commit/complete under the frozen "
        "5 ms per-RPC budget in the synthetic two-home A100/MPS mode. This is "
        "finite-sample control-path qualification, not WCET, production d_MAC, "
        "cross-family, or scheduler-latency isolation evidence."
    )
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = analyze(args.campaign, args.arms)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2) + "\n")
    temporary.replace(output)
    if not result["all_pass"]:
        raise SystemExit("C137 service-bound telemetry gate failed")


if __name__ == "__main__":
    main()
