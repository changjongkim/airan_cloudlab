#!/usr/bin/env python3
"""Audit physical evidence for the v11 AI40+control15 exchange class."""

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path


def audit_controller(data, source):
    by_release = defaultdict(list)
    for row in data["records"]:
        by_release[int(row["index"])].append(row)
    background = {
        row["request_id"]: row for row in data["background_records"]
    }
    branch_count = 0
    exchange_count = 0
    candidate_count = 0
    offered_bounds = Counter()
    candidate_examples = []
    for decision in data["recovery_decisions"]:
        rows = by_release[int(decision["release_index"])]
        admitted = [row for row in rows if row["admitted"]]
        successes = [row for row in admitted if row["nrx_commit"]]
        rejected = [row for row in rows if not row["admitted"]]
        branch = (
            len(admitted) == 2
            and len(successes) == 2
            and len(rejected) == 2
        )
        branch_count += int(branch)
        request_id = decision.get("selected_request_id")
        record = background.get(request_id)
        if decision.get("ai_lease") and record is not None:
            exchange_count += 1
            offered_bounds[float(record["bound_ms"])] += 1
            exact = branch and float(record["bound_ms"]) == 40.0
            if exact:
                candidate_count += 1
                margin_ms = (
                    int(record["horizon_ns"])
                    - int(record["returned_ns"])
                    - round(float(data["admission_ai_guard_ms"]) * 1e6)
                ) / 1e6
                if len(candidate_examples) < 5:
                    candidate_examples.append({
                        "release_index": decision["release_index"],
                        "request_id": request_id,
                        "pending_cells": decision["pending_cells"],
                        "execution_ms": record["execution_ms"],
                        "reserved_horizon_margin_ms": margin_ms,
                    })
    safety = {
        "deadline_misses": data["deadline_misses"],
        "nrx_bound_violations": data["nrx_bound_violations"],
        "conv_bound_violations": (
            data["conv_bound_violations"] + data["conv_path_bound_violations"]
        ),
        "ai_bound_violations": data["background_budget_violations"],
        "ai_horizon_violations": data["background_horizon_violations"],
    }
    return {
        "source": str(source),
        "release_count": len(data["recovery_decisions"]),
        "two_admitted_two_success_two_rejected_branches": branch_count,
        "after_observation_ai_exchanges": exchange_count,
        "exchange_bound_counts": {
            str(bound): count for bound, count in sorted(offered_bounds.items())
        },
        "ai40_candidate_occurrences": candidate_count,
        "candidate_examples": candidate_examples,
        "safety": safety,
        "all_declared_safety_zero": all(value == 0 for value in safety.values()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--controllers", nargs="+", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    paths = [Path(item).resolve() for item in args.controllers]
    audits = [audit_controller(json.loads(path.read_text()), path) for path in paths]
    total_candidates = sum(item["ai40_candidate_occurrences"] for item in audits)
    output = {
        "schema": "softwall-v11-candidate-physical-evidence-audit-v1",
        "status": (
            "PHYSICAL_OCCURRENCE_FOUND"
            if total_candidates else "NO_OFFERED_AI40_EVIDENCE"
        ),
        "scope": (
            "Post-hoc audit of frozen C132-C134 controller artifacts for the "
            "v11 branch with two admitted/successful NRx jobs, two rejected "
            "live recoveries, and a raw 40 ms AI request. It does not infer "
            "an unlogged end-to-end RPC duration or change frozen gates."
        ),
        "analyzer_sha256": hashlib.sha256(
            Path(__file__).resolve().read_bytes()
        ).hexdigest(),
        "controller_count": len(audits),
        "controller_sha256": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        },
        "totals": {
            "releases": sum(item["release_count"] for item in audits),
            "two_admitted_two_success_two_rejected_branches": sum(
                item["two_admitted_two_success_two_rejected_branches"]
                for item in audits
            ),
            "after_observation_ai_exchanges": sum(
                item["after_observation_ai_exchanges"] for item in audits
            ),
            "ai40_candidate_occurrences": total_candidates,
            "all_declared_safety_zero": all(
                item["all_declared_safety_zero"] for item in audits
            ),
        },
        "controllers": audits,
        "claim_boundary": (
            "Zero occurrences means the sealed BurstGPT replay did not "
            "physically exercise the model-derived AI40 class; it is not "
            "evidence that the class is infeasible."
        ),
    }
    temporary = Path(args.output).resolve().with_suffix(".tmp")
    temporary.write_text(json.dumps(output, indent=2))
    temporary.replace(Path(args.output).resolve())


if __name__ == "__main__":
    main()
