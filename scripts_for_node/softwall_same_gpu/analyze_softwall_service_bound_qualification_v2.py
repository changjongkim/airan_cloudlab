#!/usr/bin/env python3
"""Combine original-mode component evidence with C137 RPC telemetry mode."""

import argparse
import json
from pathlib import Path

from analyze_softwall_clustered_bound_qualification import (
    campaign_arm_results,
    collect_arm,
    numeric,
    sha256,
    summarize_component,
    zero_failure_upper,
)


def sum_totals(original, telemetry):
    keys = (
        "arms", "radio_records", "home_releases", "atomic_exchanges",
        "candidate_branches", "ai40_candidate_exchanges",
    )
    totals = {key: int(original[key]) + int(telemetry[key]) for key in keys}
    totals["nodes"] = int(original["nodes"]) + len(set(telemetry.get("nodes", [])))
    totals["declared_safety_violations"] = int(
        original.get("declared_safety_violations", 0)
    )
    return totals


def analyze(root, original_path, combined_path, c137_path):
    original = json.loads(original_path.read_text())
    combined = json.loads(combined_path.read_text())
    c137, c137_arm_paths = campaign_arm_results(root, c137_path)
    telemetry_arms = [collect_arm(root, "C137_RPC_TELEMETRY", path)
                      for path in c137_arm_paths]
    signature = original["mode_signature"]
    logical_mode_match = all(arm["mode_signature"] == signature for arm in telemetry_arms)
    bounds = {
        "nrx_response_ms": signature["nrx_bound_ms"],
        "conv_host_path_ms": signature["conv_bound_ms"],
        "ai_execution_ms": signature["ai128_bound_ms"],
    }
    components = {
        key: summarize_component(telemetry_arms, key, float(bound))
        for key, bound in bounds.items()
    } if logical_mode_match and all(numeric(value) for value in bounds.values()) else {}
    component_pass = bool(components) and all(
        item["sample_exceedances"] == 0 for item in components.values()
    )
    telemetry_nodes = sorted(set(arm["node"] for arm in telemetry_arms))
    original_nodes = set(original["physical_nodes"])
    control = c137.get("control_evidence", {})
    gates = {
        "original_mode_finite_sample_pass": original.get("finite_sample_pass") is True,
        "telemetry_campaign_pass": c137.get("all_pass") is True,
        "logical_contract_fields_match": logical_mode_match,
        "telemetry_components_within_declared_bounds": component_pass,
        "control_rpc_telemetry_pass": control.get("all_pass") is True,
        "six_telemetry_arms": len(telemetry_arms) == 6,
        "one_new_telemetry_node": len(telemetry_nodes) == 1
            and set(telemetry_nodes).isdisjoint(original_nodes),
    }
    totals = sum_totals(combined["totals"], dict(c137["totals"], nodes=c137["nodes"]))
    return {
        "schema": "softwall-service-bound-qualification-v2",
        "status": "FINITE_SAMPLE_TELEMETRY_PASS" if all(gates.values()) else "UNQUALIFIED",
        "inputs": {
            "original_clustered_qualification": str(original_path.relative_to(root)),
            "original_sha256": sha256(original_path),
            "c135_c136_combined": str(combined_path.relative_to(root)),
            "combined_sha256": sha256(combined_path),
            "c137": str(c137_path.relative_to(root)),
            "c137_sha256": sha256(c137_path),
        },
        "mode_partition": {
            "original_uninstrumented": {
                "campaigns": ["C135", "C136"],
                "nodes": original["physical_nodes"],
                "arms": len(original["arms"]),
                "status": original["status"],
            },
            "rpc_instrumented": {
                "campaign": "C137",
                "nodes": telemetry_nodes,
                "arms": len(telemetry_arms),
                "status": "FINITE_SAMPLE_PASS" if c137.get("all_pass") else "UNQUALIFIED",
            },
            "rule": (
                "Instrumentation is a mode axis. Component and mechanism evidence may be "
                "compared across the two partitions, but C137 does not retroactively turn "
                "the uninstrumented C135/C136 control path into measured RPC telemetry."
            ),
        },
        "logical_mode_signature": signature if logical_mode_match else None,
        "c137_component_evidence": components,
        "c137_control_rpc_evidence": control,
        "mechanism_cross_mode_reproduction": {
            "totals": totals,
            "node_names": sorted(original_nodes | set(telemetry_nodes)),
            "zero_failure_95_upper_if_iid": {
                "radio_tb": zero_failure_upper(totals["radio_records"]),
                "home_release": zero_failure_upper(totals["home_releases"]),
                "arm_process_lifecycle": zero_failure_upper(totals["arms"]),
                "physical_node": zero_failure_upper(totals["nodes"]),
            },
            "interpretation": (
                "Mechanism reproduction spans three A100 nodes, but one node uses added "
                "RPC telemetry. Treat the node statistic as cross-mode sensitivity, not "
                "an exact-mode hardware-population guarantee."
            ),
        },
        "gates": gates,
        "all_pass": all(gates.values()),
        "hard_real_time_status": "UNPROVEN",
        "claim_boundary": (
            "C137 directly measured 5,993 on-path broker calls on a third A100 node and "
            "kept every prepare/commit/complete call within the frozen 5 ms budget while "
            "the PHY, AI40 conditional exchange, and credit gates passed. This removes a "
            "telemetry gap but remains finite-sample evidence, not WCET or production timing."
        ),
    }


def main():
    root_default = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=root_default)
    parser.add_argument("--original", type=Path)
    parser.add_argument("--combined", type=Path)
    parser.add_argument("--c137", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    original = (args.original or root / "results/softwall_multigpu/softwall_v12_clustered_bound_qualification_v1.json").resolve()
    combined = (args.combined or root / "results/softwall_multigpu/confirm135_136_combined_v12_qualification.json").resolve()
    c137 = (args.c137 or root / "results/softwall_multigpu/confirm137_service_bound_telemetry_result.json").resolve()
    output = (args.output or root / "results/softwall_multigpu/softwall_service_bound_qualification_v2.json").resolve()
    result = analyze(root, original, combined, c137)
    output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "status": result["status"]}, indent=2))
    if not result["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
