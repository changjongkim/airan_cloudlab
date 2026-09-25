#!/usr/bin/env python3
"""Aggregate V14 pipelined-control AI45 conditional-exchange arms."""

import argparse
import hashlib
import json
from pathlib import Path

from analyze_confirm135_static_counterfactual import counterfactual_terms


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(campaign_path, arm_paths):
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    root = campaign_path.parents[2]
    expected = campaign["arms"]
    arms = []
    nodes = set()
    seeds = []
    events = []
    artifact_mismatches = []
    source_mismatches = []
    for path, declared in zip(arm_paths, expected):
        path = Path(path).resolve()
        result = json.loads(path.read_text())
        protocol = json.loads(Path(result["protocol"]).read_text())
        broker = json.loads(Path(result["broker"]).read_text())
        broker_states = {
            row["request_id"]: row["state"] for row in broker["requests"]
        }
        seeds.append((tuple(protocol["payload_seeds"]),
                      tuple(protocol["channel_seeds"])))
        home_nodes = set()
        commit_records = []
        deferred_records = []
        arm_events = []
        for home in result["homes"]:
            controller = json.loads(Path(home["controller"]).read_text())
            home_nodes.add(controller["host"])
            telemetry = home["rpc_telemetry"]
            commit_records.extend(
                row for row in telemetry["records"]
                if row["operation"] == "commit"
            )
            deferred_records.extend(
                row for row in telemetry["records"]
                if row["operation"] in {"prepare", "abort", "complete"}
            )
            by_index = {i: [] for i in range(protocol["iterations"])}
            for row in controller["records"]:
                by_index[row["index"]].append(row)
            background = {
                row["request_id"]: row
                for row in controller["background_records"]
            }
            for decision in controller["recovery_decisions"]:
                rows = by_index[decision["release_index"]]
                admitted = [row for row in rows if row["admitted"]]
                success = [row for row in admitted if row["nrx_commit"]]
                rejected = [row for row in rows if not row["admitted"]]
                record = background.get(decision.get("selected_request_id"))
                target = (
                    len(admitted) == 2 and len(success) == 2
                    and len(rejected) == 2 and decision.get("ai_lease")
                    and record is not None
                    and float(record["bound_ms"])
                    == float(protocol["target_ai_bound_ms"])
                )
                if not target:
                    continue
                terms = counterfactual_terms(
                    protocol, controller["cells"], len(success), len(rejected),
                    record["bound_ms"],
                )
                release_ns = rows[0]["release_ns"]
                physical_margin = (
                    record["horizon_ns"] - record["returned_ns"]
                    - round(protocol["admission_ai_guard_ms"] * 1e6)
                ) / 1e6
                event = {
                    "arm": declared["name"],
                    "home": home["home"],
                    "release_index": decision["release_index"],
                    "request_id": record["request_id"],
                    "terms": terms,
                    "physical_guarded_horizon_margin_ms": physical_margin,
                    "immediate_complete_ack": record["global_complete_confirmed"],
                    "final_broker_state": broker_states.get(record["request_id"]),
                    "checks": {
                        "static_rejects": terms["static_margin_ms"] < 0,
                        "conditional_accepts": terms["conditional_margin_ms"] >= 0,
                        "effective_bound_54_ms": (
                            terms["effective_transaction_bound_ms"] == 54.0
                        ),
                        "physical_horizon": physical_margin >= 0,
                        "deferred_not_mislabeled_ack": not record[
                            "global_complete_confirmed"
                        ],
                        "eventually_completed_at_broker": (
                            broker_states.get(record["request_id"]) == "completed"
                        ),
                    },
                }
                arm_events.append(event)
                events.append(event)
        nodes.update(home_nodes)
        for name, digest in result["artifact_sha256"].items():
            if not Path(name).is_file() or sha256(name) != digest:
                artifact_mismatches.append(name)
        for container, digest in protocol["source_sha256"].items():
            relative = campaign["container_source_map"].get(container)
            if relative is None or not (root / relative).is_file() \
                    or sha256(root / relative) != digest:
                source_mismatches.append({"arm": str(path), "source": container})
        arms.append({
            "name": declared["name"],
            "result": str(path),
            "all_pass": result["all_pass"],
            "seed_match": (
                protocol["payload_seeds"] == declared["payload_seeds"]
                and protocol["channel_seeds"] == declared["channel_seeds"]
            ),
            "nodes": sorted(home_nodes),
            "radio_records": sum(
                home["summary"]["records"] for home in result["homes"]
            ),
            "candidate_branches": result["candidate_evidence"]["branches"],
            "candidate_exchanges": len(arm_events),
            "commit_records": len(commit_records),
            "commit_max_ms": max(
                (row["elapsed_ms"] for row in commit_records), default=None
            ),
            "deferred_records": len(deferred_records),
            "deferred_rpc_max_ms": max(
                (row["elapsed_ms"] for row in deferred_records), default=None
            ),
        })
    campaign_source_mismatches = []
    for relative, digest in campaign["source_sha256"].items():
        path = root / relative
        if not path.is_file() or sha256(path) != digest:
            campaign_source_mismatches.append(relative)
    model_path = root / campaign["finite_model_artifact"]
    model = json.loads(model_path.read_text())
    checks = [value for event in events for value in event["checks"].values()]
    gates = {
        "two_declared_arms": len(arms) == len(expected) == 2,
        "seed_pairs_match": all(arm["seed_match"] for arm in arms),
        "seed_pairs_unique": len(seeds) == len(set(seeds)),
        "all_arm_gates_pass": all(arm["all_pass"] for arm in arms),
        "candidate_branch_each_arm": all(
            arm["candidate_branches"] > 0 for arm in arms
        ),
        "ai45_exchange_each_arm": all(
            arm["candidate_exchanges"] > 0 for arm in arms
        ),
        "all_event_checks_pass": bool(checks) and all(checks),
        "commit_only_bound_each_arm": all(
            arm["commit_records"] > 0
            and arm["commit_max_ms"] <= campaign["commit_bound_ms"]
            for arm in arms
        ),
        "deferred_rpc_observed_each_arm": all(
            arm["deferred_records"] > 0 for arm in arms
        ),
        "finite_model_pass": model["all_pass"] and not model["violations"],
        "one_new_physical_node": len(nodes) == 1
            and nodes.isdisjoint(set(campaign["excluded_nodes"])),
        "artifact_hashes_match": not artifact_mismatches,
        "arm_sources_match": not source_mismatches,
        "campaign_sources_match": not campaign_source_mismatches,
    }
    return {
        "schema": "softwall-confirm142-v14-pipelined-ai45-v1",
        "campaign_protocol": str(campaign_path),
        "campaign_protocol_sha256": sha256(campaign_path),
        "nodes": sorted(nodes),
        "arms": arms,
        "totals": {
            "radio_records": sum(arm["radio_records"] for arm in arms),
            "candidate_branches": sum(
                arm["candidate_branches"] for arm in arms
            ),
            "ai45_candidate_exchanges": len(events),
            "minimum_physical_guarded_horizon_margin_ms": min(
                event["physical_guarded_horizon_margin_ms"] for event in events
            ) if events else None,
            "maximum_commit_rpc_ms": max(
                arm["commit_max_ms"] for arm in arms
                if arm["commit_max_ms"] is not None
            ),
            "maximum_deferred_rpc_ms": max(
                arm["deferred_rpc_max_ms"] for arm in arms
                if arm["deferred_rpc_max_ms"] is not None
            ),
        },
        "model_counterfactual": {
            "static_all_fail_slack_ms": 53.0,
            "raw_ai_bound_ms": 45.0,
            "launch_commit_bound_ms": 7.0,
            "completion_guard_ms": 2.0,
            "effective_transaction_bound_ms": 54.0,
            "conditional_decision_window_ms": 58.0,
            "static_margin_ms": -1.0,
            "conditional_margin_ms": 4.0,
        },
        "finite_model": {
            "path": str(model_path),
            "states": model["states"],
            "edges": model["edges"],
            "violations": len(model["violations"]),
        },
        "events": events,
        "artifact_hash_mismatches": artifact_mismatches,
        "arm_source_hash_mismatches": source_mismatches,
        "campaign_source_hash_mismatches": campaign_source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Two-arm finite-sample validation that asynchronous prepare/complete "
            "leave only the 7 ms launch commit on the RAN admission path and admit "
            "an exchange-only AI45 class (54 ms effective) in a synthetic 58 ms "
            "conditional window. This is not WCET or production d_MAC evidence."
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
        raise SystemExit("C142 V14 pipelined AI45 gate failed")


if __name__ == "__main__":
    main()
