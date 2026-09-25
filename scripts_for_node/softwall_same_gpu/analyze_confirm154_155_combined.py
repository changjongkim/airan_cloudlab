#!/usr/bin/env python3.11
"""Combine the two prospective controlled-outcome integrated holdouts."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign154", type=Path, required=True)
    parser.add_argument("--protocol154", type=Path, required=True)
    parser.add_argument("--campaign155", type=Path, required=True)
    parser.add_argument("--protocol155", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    campaigns = [
        json.loads(args.campaign154.read_text(encoding="utf-8")),
        json.loads(args.campaign155.read_text(encoding="utf-8")),
    ]
    protocols = [
        json.loads(args.protocol154.read_text(encoding="utf-8")),
        json.loads(args.protocol155.read_text(encoding="utf-8")),
    ]
    nodes = [row["nodes"][0] for row in campaigns]
    jobs = [row["job_ids"][0] for row in campaigns]
    totals = {
        key: sum(row["totals"][key] for row in campaigns)
        for key in campaigns[0]["totals"]
    }
    mode_keys = (
        "period_ms", "expiry_ms", "nrx_bound_ms", "conventional_bound_ms",
        "ai_bound_ms", "guard_ms", "capacity", "qwen_context_length",
        "qwen_model", "snr_db", "warmup", "release_lead_ms",
        "release_semantics",
    )
    gates = {
        "both_campaigns_pass": all(row["all_pass"] for row in campaigns),
        "distinct_nodes_and_jobs": len(set(nodes)) == 2 and len(set(jobs)) == 2,
        "second_protocol_excludes_first_node": nodes[0] in protocols[1]["excluded_nodes"],
        "source_hashes_identical": (
            protocols[0]["source_sha256"] == protocols[1]["source_sha256"]
        ),
        "mode_identical": all(
            protocols[0]["mode"][key] == protocols[1]["mode"][key]
            for key in mode_keys
        ),
        "independent_seeds": (
            protocols[0]["arms"] != protocols[1]["arms"]
            and {
                seed
                for arm in protocols[0]["arms"]
                for group in arm["seeds"].values()
                for seed in group.values()
            }.isdisjoint({
                seed
                for arm in protocols[1]["arms"]
                for group in arm["seeds"].values()
                for seed in group.values()
            })
        ),
        "opposite_branch_order": (
            protocols[1]["branch_order"]
            == list(reversed(protocols[0]["branch_order"]))
        ),
        "combined_totals": totals == {
            "accepted_debts": 32,
            "correct_home_commits": 20,
            "deadline_misses": 0,
            "physical_recoveries": 20,
            "qwen_units": 4,
            "rejected_debts": 4,
            "submitted_debts": 36,
            "success_outcomes": 12,
        },
    }
    value = {
        "schema": "softwall-confirm154-155-controlled-integrated-holdout-v1",
        "status": (
            "TWO_NODE_CONTROLLED_INTEGRATED_HOLDOUT_PASS_ACTUAL_NRX_UQ"
            if all(gates.values()) else "COMBINED_HOLDOUT_FAIL"
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
        "nodes": nodes,
        "job_ids": jobs,
        "totals": totals,
        "campaigns": [str(args.campaign154), str(args.campaign155)],
        "protocols": [str(args.protocol154), str(args.protocol155)],
        "artifact_sha256": {
            str(path): sha256(path)
            for path in (
                args.campaign154, args.protocol154,
                args.campaign155, args.protocol155,
            )
        },
        "claim_scope": (
            "Two A100 nodes, opposite branch order, independent seeds, and "
            "prospectively controlled NRx outcome transitions. This qualifies "
            "the global-certificate/Qwen-fence/shared-cuPHY transaction for "
            "the frozen synthetic mode, but actual NeuralRx-driven transitions, "
            "integrated fault containment, WCET, production d_MAC, and "
            "throughput superiority remain unqualified."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

