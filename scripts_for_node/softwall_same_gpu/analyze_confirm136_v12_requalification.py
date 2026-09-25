#!/usr/bin/env python3
"""Aggregate cross-node V12 AI40 requalification arms."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_confirm135_static_counterfactual import analyze_arm


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def zero_failure_upper(trials, confidence=0.95):
    """Exact one-sided binomial upper bound after zero observed failures."""
    if trials <= 0:
        raise ValueError("trials must be positive")
    if not 0 < confidence < 1:
        raise ValueError("confidence must lie strictly between zero and one")
    return 1.0 - (1.0 - confidence) ** (1.0 / trials)


def analyze(campaign_path, arm_paths):
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    project_root = campaign_path.parents[2]
    expected = campaign["arms"]
    arms = []
    seed_keys = []
    nodes = set()
    artifact_mismatches = []
    counterfactual_events = []
    for arm_path, arm_expected in zip(arm_paths, expected):
        arm_path = Path(arm_path).resolve()
        result = json.loads(arm_path.read_text())
        protocol_path = Path(result["protocol"])
        protocol = json.loads(protocol_path.read_text())
        seed_key = (
            tuple(protocol["payload_seeds"]),
            tuple(protocol["channel_seeds"]),
        )
        seed_keys.append(seed_key)
        home_nodes = set()
        for home in result["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            home_nodes.add(controller["host"])
        nodes.update(home_nodes)
        for path, expected_hash in result["artifact_sha256"].items():
            path = Path(path)
            if not path.is_file() or sha256(path) != expected_hash:
                artifact_mismatches.append(str(path))
        counterfactual = analyze_arm(arm_path)
        counterfactual_events.extend(counterfactual["events"])
        arms.append({
            "name": arm_expected["name"],
            "result": str(arm_path),
            "protocol": str(protocol_path),
            "seed_match": (
                list(protocol["payload_seeds"]) == arm_expected["payload_seeds"]
                and list(protocol["channel_seeds"]) == arm_expected["channel_seeds"]
            ),
            "nodes": sorted(home_nodes),
            "all_pass": result["all_pass"],
            "radio_records": sum(
                home["summary"]["records"] for home in result["homes"]
            ),
            "atomic_exchanges": sum(
                home["summary"]["atomic_exchange"] for home in result["homes"]
            ),
            "candidate_branches": result["candidate_evidence"]["branches"],
            "ai40_candidate_exchanges": result["candidate_evidence"]["exchanges"],
            "minimum_reserved_horizon_margin_ms": result[
                "candidate_evidence"
            ]["minimum_reserved_horizon_margin_ms"],
            "counterfactual_events": len(counterfactual["events"]),
        })

    source_mismatches = []
    for relative, expected_hash in campaign["source_sha256"].items():
        path = project_root / relative
        if not path.is_file() or sha256(path) != expected_hash:
            source_mismatches.append(relative)

    total_radio = sum(arm["radio_records"] for arm in arms)
    total_home_releases = sum(
        2 * int(expected_arm["iterations"]) for expected_arm in expected
    )
    event_checks = [
        check
        for event in counterfactual_events
        for check in event["checks"].values()
    ]
    gates = {
        "all_declared_arms_present": len(arms) == len(expected) == 6,
        "all_seed_pairs_match": all(arm["seed_match"] for arm in arms),
        "all_seed_pairs_unique": len(seed_keys) == len(set(seed_keys)),
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "candidate_branch_each_arm": all(
            arm["candidate_branches"] > 0 for arm in arms
        ),
        "ai40_exchange_each_arm": all(
            arm["ai40_candidate_exchanges"] > 0 for arm in arms
        ),
        "nonnegative_physical_margin_each_arm": all(
            arm["minimum_reserved_horizon_margin_ms"] is not None
            and arm["minimum_reserved_horizon_margin_ms"] >= 0
            for arm in arms
        ),
        "static_counterfactual_rejects_every_exchange": bool(counterfactual_events)
            and all(event["terms"]["static_margin_ms"] < 0
                    for event in counterfactual_events),
        "conditional_contract_accepts_every_exchange": bool(counterfactual_events)
            and all(event["terms"]["conditional_margin_ms"] >= 0
                    for event in counterfactual_events),
        "all_event_execution_checks_pass": bool(event_checks) and all(event_checks),
        "single_new_physical_node": len(nodes) == 1
            and nodes.isdisjoint(set(campaign["excluded_nodes"])),
        "artifact_hashes_match": not artifact_mismatches,
        "source_hashes_match": not source_mismatches,
    }
    return {
        "schema": "softwall-confirm136-v12-cross-node-requalification-v1",
        "campaign_protocol": str(campaign_path),
        "campaign_protocol_sha256": sha256(campaign_path),
        "nodes": sorted(nodes),
        "excluded_nodes": campaign["excluded_nodes"],
        "arms": arms,
        "totals": {
            "arms": len(arms),
            "radio_records": total_radio,
            "home_releases": total_home_releases,
            "atomic_exchanges": sum(arm["atomic_exchanges"] for arm in arms),
            "candidate_branches": sum(arm["candidate_branches"] for arm in arms),
            "ai40_candidate_exchanges": sum(
                arm["ai40_candidate_exchanges"] for arm in arms
            ),
            "counterfactual_events": len(counterfactual_events),
            "minimum_physical_guarded_horizon_margin_ms": min(
                event["observed"]["physical_guarded_horizon_margin_ms"]
                for event in counterfactual_events
            ) if counterfactual_events else None,
        },
        "zero_failure_sensitivity": {
            "confidence": 0.95,
            "tb_level_iid_upper": zero_failure_upper(total_radio),
            "home_release_level_iid_upper": zero_failure_upper(
                total_home_releases
            ),
            "arm_level_iid_upper": zero_failure_upper(len(arms)),
            "scope": (
                "Exact binomial zero-failure sensitivity under the stated IID "
                "unit; persistent-process and within-arm dependence mean the "
                "arm-level value is the conservative reported qualification "
                "summary. None of these values is a WCET proof."
            ),
        },
        "artifact_hash_mismatches": artifact_mismatches,
        "source_hash_mismatches": source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Same-family cross-node finite-sample requalification of the V12 "
            "synthetic AI40 mechanism class; not cross-family, production d_MAC, "
            "WCET, BurstGPT throughput, restart, or exactly-once evidence."
        ),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign", required=True)
    parser.add_argument("--arms", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    value = analyze(args.campaign, args.arms)
    output = Path(args.output).resolve()
    temporary = output.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(output)
    if not value["all_pass"]:
        raise SystemExit("C136 V12 requalification gate failed")


if __name__ == "__main__":
    main()
