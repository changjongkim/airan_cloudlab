#!/usr/bin/env python3.11
"""Retrospectively validate C162 predictions against frozen C159-Q2 runs.

This is a consistency check over already observed physical decisions.  It is
not the independent C162 holdout and makes no new physical-bound claim.
"""

from __future__ import annotations

import argparse
import collections
import dataclasses
import hashlib
import json
import math
from pathlib import Path

from c162_feasibility_model_v1 import EnvelopePoint, predict


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_campaign(coordinator_path: Path, protocol_path: Path) -> dict:
    coordinator = json.loads(coordinator_path.read_text())
    protocol = json.loads(protocol_path.read_text())
    accepted_debts = len(protocol["accepted_physical_keys"])
    rows = []
    confusion = collections.Counter()
    for physical in coordinator["rounds"]:
        attempts = physical["launch_revalidation_attempts"]
        if not attempts:
            raise ValueError(f"round {physical['sequence']} has no revalidation")
        decision_ns = attempts[-1]["launch_now_ns"]
        if decision_ns != physical["outcome_transition"]["applied_at_ns"]:
            raise ValueError(f"round {physical['sequence']} transition mismatch")
        unresolved = len(physical["outcome_transition"]["unresolved_obligations"])
        # The integer-ms C162 classifier rounds observation time upward.  This
        # makes the retrospective comparison conservative at sub-ms boundaries.
        decision_ms = math.ceil(decision_ns / 1_000_000)
        point = EnvelopePoint(
            accepted_debts=accepted_debts,
            unresolved_debts=unresolved,
            decision_time_ms=decision_ms,
            context_length=physical["context_length"],
        )
        prediction = predict(point)
        predicted_lease = prediction.state == "QSU"
        observed_lease = bool(physical["lease_accepted"])
        confusion[(predicted_lease, observed_lease)] += 1
        rows.append({
            "sequence": physical["sequence"],
            "context_length": physical["context_length"],
            "accepted_debts": accepted_debts,
            "unresolved_debts": unresolved,
            "decision_ns": decision_ns,
            "conservative_decision_ms": decision_ms,
            "predicted_state": prediction.state,
            "predicted_reason": prediction.reason,
            "observed_lease_accepted": observed_lease,
            "observed_lease_reason": physical["lease_reason"],
            "match": predicted_lease == observed_lease,
        })
    matches = sum(row["match"] for row in rows)
    return {
        "label": protocol["label"],
        "host": coordinator["host"],
        "slurm_job_id": coordinator["slurm_job_id"],
        "rounds": len(rows),
        "matches": matches,
        "mismatches": len(rows) - matches,
        "confusion": {
            "predicted_accept_observed_accept": confusion[(True, True)],
            "predicted_accept_observed_reject": confusion[(True, False)],
            "predicted_reject_observed_accept": confusion[(False, True)],
            "predicted_reject_observed_reject": confusion[(False, False)],
        },
        "coordinator_sha256": sha256(coordinator_path),
        "protocol_sha256": sha256(protocol_path),
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--campaign", nargs=2, action="append", metavar=("COORDINATOR", "PROTOCOL"), required=True)
    args = parser.parse_args()

    campaigns = [validate_campaign(Path(c), Path(p)) for c, p in args.campaign]
    total = sum(item["rounds"] for item in campaigns)
    matches = sum(item["matches"] for item in campaigns)
    gates = {
        "two_independent_nodes": len({item["host"] for item in campaigns}) == 2,
        "expected_1200_rounds": total == 1200,
        "prediction_matches_every_observed_lease_decision": matches == total,
        "both_accept_and_reject_observed": all(
            sum(item["confusion"][key] for item in campaigns) > 0
            for key in (
                "predicted_accept_observed_accept",
                "predicted_reject_observed_reject",
            )
        ),
    }
    value = {
        "schema": "softwall-c162-q2-retrospective-validation-v1",
        "status": "C162_Q2_RETROSPECTIVE_PASS" if all(gates.values()) else "C162_Q2_RETROSPECTIVE_FAIL",
        "all_pass": all(gates.values()),
        "scope": "Retrospective consistency check against frozen C159-Q2 physical decisions; not an independent C162 physical holdout.",
        "gates": gates,
        "total_rounds": total,
        "matches": matches,
        "mismatches": total - matches,
        "model_sha256": sha256(args.scripts_root / "c162_feasibility_model_v1.py"),
        "validator_sha256": sha256(args.scripts_root / "c162_validate_q2_predictions.py"),
        "campaigns": campaigns,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "total_rounds": total, "matches": matches,
        "campaigns": [{key: item[key] for key in ("label", "host", "rounds", "matches", "mismatches", "confusion")} for item in campaigns],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
