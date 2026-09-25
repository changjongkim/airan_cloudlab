#!/usr/bin/env python3.11
"""Exhaustively audit the process-replacement evidence product."""

from __future__ import annotations

import hashlib
import itertools
import json
from pathlib import Path

from c164_process_replacement_model_v1 import (
    ReplacementEvidence, qualify_replacement, safety_invariants,
)


FIELDS = (
    "identity_match", "target_present_before", "terminate_client_success",
    "process_exit_confirmed", "target_absent_after",
    "same_mps_server_survived", "mandatory_client_preserved",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    root = Path(__file__).resolve().parents[2]
    rows = []
    violations = []
    accepted = 0
    for stage in ("prepared", "launched"):
        for values in itertools.product((False, True), repeat=len(FIELDS)):
            evidence = ReplacementEvidence(stage, **dict(zip(FIELDS, values)))
            decision = qualify_replacement(evidence)
            expected = all(values)
            checks = {
                "decision_matches_complete_evidence_rule": decision.accepted == expected,
                "safety_invariants": all(safety_invariants(evidence, decision).values()),
            }
            accepted += int(decision.accepted)
            if not all(checks.values()):
                violations.append({"stage": stage, "values": values,
                                   "reason": decision.reason, "checks": checks})
            rows.append({"stage": stage, **dict(zip(FIELDS, values)),
                         "accepted": decision.accepted,
                         "reason": decision.reason,
                         "terminal_stage": decision.terminal_stage,
                         "checks": checks})
    checks = {
        "complete_state_product": len(rows) == 256,
        "only_two_complete_evidence_states_accept": accepted == 2,
        "all_other_states_fail_closed": len(rows) - accepted == 254,
        "zero_invariant_violations": not violations,
    }
    source = Path(__file__)
    model = source.with_name("c164_process_replacement_model_v1.py")
    test = source.with_name("test_c164_process_replacement_model_v1.py")
    value = {
        "schema": "softwall-c164-process-replacement-model-audit-v1",
        "status": "C164_PROCESS_REPLACEMENT_MODEL_PASS" if all(checks.values()) else "C164_PROCESS_REPLACEMENT_MODEL_FAIL",
        "all_pass": all(checks.values()), "checks": checks,
        "counts": {"state_product": len(rows), "accepted": accepted,
                   "fail_closed": len(rows) - accepted,
                   "invariant_violations": len(violations)},
        "mps_quiescence_primitive": {
            "command": "terminate_client <server PID> <client PID>",
            "success_code": "0",
            "official_reference": "https://docs.nvidia.com/deploy/mps/when-to-use-mps.html#client-early-termination",
        },
        "claim_boundary": (
            "Pure evidence-composition model. Physical MPS termination, "
            "mandatory RAN continuity, process replacement, production d_MAC "
            "and WCET remain unqualified until separate gates pass."
        ),
        "source_sha256": {source.name: sha256(source), model.name: sha256(model),
                          test.name: sha256(test)},
        "violations": violations, "states": rows,
    }
    output = root / "results/softwall_multigpu/c164_process_replacement_model_v1.json"
    output.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"status": value["status"], **value["counts"]}, indent=2))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
