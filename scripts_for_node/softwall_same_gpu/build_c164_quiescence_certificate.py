#!/usr/bin/env python3.11
"""Compose MPS termination, process exit and client-survival evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c164_process_replacement_model_v1 import (
    ReplacementEvidence, qualify_replacement, safety_invariants,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--termination", type=Path, required=True)
    parser.add_argument("--exit", dest="exit_path", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    journal, before, termination, exited, after = map(load, (
        args.journal, args.before, args.termination, args.exit_path, args.after
    ))
    target = before.get("target_match", [])
    mandatory_before = before.get("mandatory_match", [])
    mandatory_after = after.get("mandatory_match", [])
    target_row = target[0] if len(target) == 1 else None
    same_server = bool(target_row) and target_row["server_pid"] in after.get("server_pids", [])
    mandatory_before_ids = {(row["pid"], row["server_pid"]) for row in mandatory_before}
    mandatory_after_ids = {(row["pid"], row["server_pid"]) for row in mandatory_after}
    evidence = ReplacementEvidence(
        journal_stage=journal.get("stage"),
        identity_match=(
            bool(target_row)
            and target_row.get("pid") == termination.get("target", {}).get("pid")
            == exited.get("client_pid")
        ),
        target_present_before=len(target) == 1,
        terminate_client_success=termination.get("all_pass") is True,
        process_exit_confirmed=exited.get("all_pass") is True,
        target_absent_after=not after.get("target_match"),
        same_mps_server_survived=same_server,
        mandatory_client_preserved=(
            bool(mandatory_before_ids)
            and mandatory_before_ids.issubset(mandatory_after_ids)
        ),
    )
    decision = qualify_replacement(evidence)
    checks = {
        "replacement_model_accepts": decision.accepted,
        "safety_invariants": all(safety_invariants(evidence, decision).values()),
        "same_control_socket": before.get("control_socket") == after.get("control_socket"),
        "monotonic_evidence_time": (
            before.get("observed_ns", 2**63) <= termination.get("started_ns", -1)
            <= termination.get("returned_ns", -1) <= exited.get("observed_ns", -1)
            <= after.get("observed_ns", -1)
        ),
        "journal_worker_pid_matches_target": bool(target_row)
            and journal.get("worker_pid") == target_row.get("pid"),
        "launched_has_physical_submission": (
            journal.get("stage") != "launched"
            or journal.get("physical_submission_observed") is True
        ),
    }
    value = {
        "schema": "softwall-c164-gpu-quiescence-certificate-v1",
        "status": "C164_GPU_QUIESCENCE_CERTIFICATE_PASS" if all(checks.values()) else "C164_GPU_QUIESCENCE_CERTIFICATE_FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "evidence": vars(evidence), "decision": vars(decision),
        "journal_before": journal, "journal_before_sha256": sha256(args.journal),
        "artifact_sha256": {"before": sha256(args.before),
                            "termination": sha256(args.termination),
                            "exit": sha256(args.exit_path), "after": sha256(args.after)},
    }
    args.output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], "checks": checks}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
