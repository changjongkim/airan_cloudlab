#!/usr/bin/env python3
"""Aggregate the prospective C135 v11 AI40 mechanism campaign."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def analyze(campaign_path, arm_paths):
    campaign_path = Path(campaign_path).resolve()
    campaign = json.loads(campaign_path.read_text())
    arms = []
    artifact_mismatches = []
    source_mismatches = []
    seed_pairs = []
    for arm_path in map(lambda item: Path(item).resolve(), arm_paths):
        result = json.loads(arm_path.read_text())
        protocol_path = Path(result["protocol"])
        protocol = json.loads(protocol_path.read_text())
        seed_pairs.append((
            tuple(protocol["payload_seeds"]),
            tuple(protocol["channel_seeds"]),
        ))
        for path, expected in result["artifact_sha256"].items():
            if not Path(path).is_file() or sha256(path) != expected:
                artifact_mismatches.append(path)
        for container_path, expected in protocol["source_sha256"].items():
            relative = campaign["container_source_map"].get(container_path)
            if relative is None or sha256(
                    campaign_path.parents[2] / relative) != expected:
                source_mismatches.append({
                    "arm": str(arm_path), "source": container_path,
                })
        arms.append({
            "result": str(arm_path),
            "protocol": str(protocol_path),
            "all_pass": result["all_pass"],
            "candidate_evidence": result["candidate_evidence"],
            "radio_records": sum(
                home["summary"]["records"] for home in result["homes"]
            ),
            "atomic_exchanges": sum(
                home["summary"]["atomic_exchange"] for home in result["homes"]
            ),
        })
    gates = {
        "two_independent_arms": len(arms) == 2 and len(set(seed_pairs)) == 2,
        "all_arm_gates_pass": all(item["all_pass"] for item in arms),
        "candidate_branch_each_arm": all(
            item["candidate_evidence"]["branches"] > 0 for item in arms
        ),
        "ai40_exchange_each_arm": all(
            item["candidate_evidence"]["exchanges"] > 0 for item in arms
        ),
        "nonnegative_margin_each_arm": all(
            item["candidate_evidence"]["minimum_reserved_horizon_margin_ms"]
            is not None
            and item["candidate_evidence"]["minimum_reserved_horizon_margin_ms"] >= 0
            for item in arms
        ),
        "artifact_hashes_match": not artifact_mismatches,
        "source_hashes_match": not source_mismatches,
    }
    return {
        "schema": "softwall-confirm135-v11-ai40-campaign-result-v1",
        "campaign_protocol": str(campaign_path),
        "campaign_protocol_sha256": sha256(campaign_path),
        "arms": arms,
        "totals": {
            "radio_records": sum(item["radio_records"] for item in arms),
            "atomic_exchanges": sum(item["atomic_exchanges"] for item in arms),
            "candidate_branches": sum(
                item["candidate_evidence"]["branches"] for item in arms
            ),
            "ai40_candidate_exchanges": sum(
                item["candidate_evidence"]["exchanges"] for item in arms
            ),
        },
        "artifact_hash_mismatches": artifact_mismatches,
        "source_hash_mismatches": source_mismatches,
        "gates": gates,
        "all_pass": all(gates.values()),
        "claim_boundary": (
            "Synthetic context128 mechanism qualification in the declared "
            "two-home A100/MPS mode; not BurstGPT throughput, WCET, production "
            "d_MAC, cross-family, or restart evidence."
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
    temporary.write_text(json.dumps(value, indent=2))
    temporary.replace(output)
    if not value["all_pass"]:
        raise SystemExit("confirm135 campaign gate failed")


if __name__ == "__main__":
    main()
