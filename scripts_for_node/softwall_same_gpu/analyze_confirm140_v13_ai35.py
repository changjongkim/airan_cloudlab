#!/usr/bin/env python3
"""Aggregate the corrected-control AI35 conditional-exchange campaign."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_confirm135_static_counterfactual import analyze_arm


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(campaign_path, arm_paths):
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    root = campaign_path.parents[2]
    expected = campaign["arms"]
    arms = []
    nodes = set()
    artifact_mismatches = []
    source_mismatches = []
    events = []
    seeds = []
    for path, declared in zip(arm_paths, expected):
        path = Path(path).resolve()
        result = json.loads(path.read_text())
        protocol = json.loads(Path(result["protocol"]).read_text())
        seeds.append((tuple(protocol["payload_seeds"]),
                      tuple(protocol["channel_seeds"])))
        home_nodes = set()
        for home in result["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            home_nodes.add(controller["host"])
        nodes.update(home_nodes)
        for name, digest in result["artifact_sha256"].items():
            if not Path(name).is_file() or sha256(name) != digest:
                artifact_mismatches.append(name)
        for container, digest in protocol["source_sha256"].items():
            relative = campaign["container_source_map"].get(container)
            if relative is None or not (root / relative).is_file() \
                    or sha256(root / relative) != digest:
                source_mismatches.append({"arm": str(path), "source": container})
        audit = analyze_arm(path)
        events.extend(audit["events"])
        arms.append({
            "name": declared["name"],
            "result": str(path),
            "all_pass": result["all_pass"],
            "seed_match": (
                protocol["payload_seeds"] == declared["payload_seeds"]
                and protocol["channel_seeds"] == declared["channel_seeds"]
            ),
            "nodes": sorted(home_nodes),
            "radio_records": sum(home["summary"]["records"] for home in result["homes"]),
            "atomic_exchanges": sum(home["summary"]["atomic_exchange"] for home in result["homes"]),
            "candidate_evidence": result["candidate_evidence"],
            "control_evidence": result["control_evidence"],
            "counterfactual_events": len(audit["events"]),
        })
    campaign_source_mismatches = []
    for relative, digest in campaign["source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            campaign_source_mismatches.append(relative)
    event_checks = [value for event in events for value in event["checks"].values()]
    gates = {
        "two_declared_arms": len(arms) == len(expected) == 2,
        "seed_pairs_match": all(arm["seed_match"] for arm in arms),
        "seed_pairs_unique": len(seeds) == len(set(seeds)),
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "candidate_branch_each_arm": all(
            arm["candidate_evidence"]["branches"] > 0 for arm in arms
        ),
        "ai35_exchange_each_arm": all(
            arm["candidate_evidence"]["exchanges"] > 0 for arm in arms
        ),
        "nonnegative_physical_margin_each_arm": all(
            arm["candidate_evidence"]["minimum_reserved_horizon_margin_ms"] is not None
            and arm["candidate_evidence"]["minimum_reserved_horizon_margin_ms"] >= 0
            for arm in arms
        ),
        "static_counterfactual_rejects_every_exchange": bool(events) and all(
            event["terms"]["static_margin_ms"] < 0 for event in events
        ),
        "conditional_contract_accepts_every_exchange": bool(events) and all(
            event["terms"]["conditional_margin_ms"] >= 0 for event in events
        ),
        "effective_transaction_is_58_ms": bool(events) and all(
            event["terms"]["effective_transaction_bound_ms"] == 58.0
            for event in events
        ),
        "all_event_execution_checks_pass": bool(event_checks) and all(event_checks),
        "one_allowed_physical_node": len(nodes) == 1
            and nodes.isdisjoint(set(campaign["excluded_nodes"])),
        "artifact_hashes_match": not artifact_mismatches,
        "arm_sources_match": not source_mismatches,
        "campaign_sources_match": not campaign_source_mismatches,
    }
    return {
        "schema": "softwall-confirm140-v13-ai35-conditional-exchange-v1",
        "campaign_protocol": str(campaign_path),
        "campaign_protocol_sha256": sha256(campaign_path),
        "nodes": sorted(nodes),
        "arms": arms,
        "totals": {
            "radio_records": sum(arm["radio_records"] for arm in arms),
            "atomic_exchanges": sum(arm["atomic_exchanges"] for arm in arms),
            "candidate_branches": sum(
                arm["candidate_evidence"]["branches"] for arm in arms
            ),
            "ai35_candidate_exchanges": len(events),
            "minimum_physical_guarded_horizon_margin_ms": min(
                event["observed"]["physical_guarded_horizon_margin_ms"]
                for event in events
            ) if events else None,
        },
        "model_counterfactual": {
            "static_all_fail_slack_ms": 53.0,
            "effective_transaction_bound_ms": 58.0,
            "conditional_decision_window_ms": 58.0,
            "static_margin_ms": -5.0,
            "conditional_margin_ms": 0.0,
            "events": events,
        },
        "artifact_hash_mismatches": artifact_mismatches,
        "arm_source_hash_mismatches": source_mismatches,
        "campaign_source_hash_mismatches": campaign_source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Two-arm finite-sample physical mechanism validation of a synthetic "
            "context64 AI35+control21+completion2 transaction that static all-fail "
            "reservation rejects but conditional recovery release admits. Same A100 "
            "node as C139; not throughput superiority, WCET, production d_MAC, "
            "cross-family, or independent-node requalification."
        ),
    }


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
        raise SystemExit("C140 V13 AI35 gate failed")


if __name__ == "__main__":
    main()
