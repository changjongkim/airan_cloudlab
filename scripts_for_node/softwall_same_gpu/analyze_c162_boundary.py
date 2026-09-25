#!/usr/bin/env python3.11
"""Gate one frozen C162 physical boundary campaign."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from build_c162_boundary_protocol import source_hashes
from c162_boundary_cases import CASE_BY_ID, case_for_sequence
from c162_feasibility_model_v1 import EnvelopePoint, predict


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values, fraction):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values):
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, .50), "p99": percentile(values, .99),
        "max": max(values) if values else None,
    }


def inventory_ok(path: Path) -> tuple[bool, list]:
    with path.open(newline="") as handle:
        rows = [{k.strip(): v.strip() for k, v in row.items()}
                for row in csv.DictReader(handle)]
    return (
        len(rows) == 4
        and all("A100" in row.get("name", "") for row in rows)
        and all(row.get("mig.mode.current", "").lower() == "disabled" for row in rows)
    ), rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--nrx-worker", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol, spec = load(args.protocol), load(args.peer_spec)
    coordinator, nrx, qwen = load(args.coordinator), load(args.nrx_worker), load(args.qwen)
    owners = [load(Path(row["owner_output"])) for row in spec["peers"]]
    valid_inventory, inventory = inventory_ok(args.inventory)
    iterations = protocol["iterations"]
    accepted = tuple(tuple(key) for key in protocol["accepted_physical_keys"])
    accepted_set = set(accepted)
    expected_pairs = {(sequence, key) for sequence in range(1, iterations + 1) for key in accepted}
    owner_records = [(tuple(owner["key"]), row) for owner in owners for row in owner["records"]]
    owner_pairs = {(row["sequence"], key) for key, row in owner_records}
    nrx_pairs = {(row["sequence"], tuple(row["key"])) for row in nrx["records"]}
    rounds = coordinator["rounds"]
    round_by_sequence = {row["sequence"]: row for row in rounds}
    same_host = (
        coordinator["host"] == nrx["host"] == qwen["host"]
        and all(owner["host"] == coordinator["host"] for owner in owners)
    )

    case_counts = collections.Counter()
    prediction_matches = []
    observed_decisions = collections.defaultdict(list)
    structure_ok = True
    for sequence, row in enumerate(rounds, start=1):
        case = case_for_sequence(sequence, protocol["reverse_cases"])
        case_counts[case.case_id] += 1
        attempts = row.get("launch_revalidation_attempts", [])
        launch_ns = attempts[-1]["launch_now_ns"] if attempts else -1
        decision_ms = math.ceil(launch_ns / 1_000_000)
        prediction = predict(EnvelopePoint(
            4, 4 - case.success_count, decision_ms, case.context_length
        ))
        observed_decisions[case.case_id].append(decision_ms)
        prediction_matches.append(
            prediction.state == case.expected_state
            and prediction.ai_safe == bool(row["lease_accepted"])
            and bool(row["lease_accepted"]) == case.expected_lease
        )
        successes = set(map(tuple, row["success_keys"]))
        failures = set(map(tuple, row["injected_failure_keys"]))
        recoveries = set(map(tuple, row["physical_recoveries"]))
        physical_successes = set(map(tuple, row["physical_timely_successes"]))
        structure_ok &= (
            row["sequence"] == sequence
            and row["boundary_case"] == case.case_id
            and row["target_decision_ms"] == case.target_decision_ms
            and row["context_length"] == case.context_length
            and row["missing_outcomes_at_cutoff"] == []
            and physical_successes == accepted_set
            and successes == set(accepted[:case.success_count])
            and failures == accepted_set - successes
            and recoveries == failures
            and row["rejected_keys"] == [protocol["expected_global_reject"]]
            and row["outcome_transition"]["kind"]
                == "atomic_prespecified_failure_injection"
            and len(attempts) >= 1
            and attempts[-1]["within_budget"] is True
            and all(attempt["within_budget"] is False for attempt in attempts[:-1])
        )

    owner_semantics = True
    for key, record in owner_records:
        row = round_by_sequence[record["sequence"]]
        success = key in set(map(tuple, row["success_keys"]))
        owner_semantics &= (
            record["timely_success"] is True
            and record["neural_correct"] is True
            and record["commit_count"] == 1
            and record["deadline_miss"] is False
            and ((success and record["commit_source"] == "actual_nrx")
                 or ((not success) and record["commit_source"] == "shared_conventional"
                     and record["recovery_contract_valid"] is True))
        )

    qwen_expected = [row for row in rounds if row["expected_lease"]]
    qwen_rejected = [row for row in rounds if not row["expected_lease"]]
    qwen_execution_ok = all(
        row["qwen"] is not None
        and row["qwen"]["launched"] is True
        and row["qwen"]["fence_confirmed"] is True
        and row["qwen"]["bound_violation"] is False
        and row["qwen"]["lease_interval_violation"] is False
        and row["qwen"]["launch_control_bound_violation"] is False
        and row["lease_retire"]["accepted"] is True
        for row in qwen_expected
    ) and all(
        row["qwen"] is None and row["qwen_attempt"] is None
        and row["lease_accepted"] is False and row["lease_retire"] is None
        for row in qwen_rejected
    )

    nrx_ms = [row["nrx_release_to_complete_ms"] for _, row in owner_records]
    commit_ms = [row["release_to_commit_ms"] for _, row in owner_records]
    recovery_ms = [
        (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
        for row in coordinator["physical_recoveries"]
    ]
    access_values = list(nrx["peer_access"].values()) + list(coordinator["peer_access"].values())
    expected_case_count = protocol["samples_per_case"]
    gates = {
        "source_and_grid_hash_match": (
            source_hashes(args.scripts_root, args.task1_root) == protocol["source_sha256"]
            and sha256(Path(protocol["grid"]["path"])) == protocol["grid"]["sha256"]
        ),
        "node_inventory_and_exclusion": (
            valid_inventory and same_host
            and coordinator["host"] not in protocol["excluded_nodes"]
        ),
        "complete_physical_sample": (
            len(rounds) == iterations
            and owner_pairs == expected_pairs and nrx_pairs == expected_pairs
            and nrx["completed_units"] == len(expected_pairs)
            and all(owner["completed_iterations"] == iterations for owner in owners)
        ),
        "prespecified_case_sample": (
            set(case_counts) == set(CASE_BY_ID)
            and all(case_counts[case] == expected_case_count for case in CASE_BY_ID)
        ),
        "physical_nrx45_stable_success": (
            len(nrx_ms) == len(expected_pairs) and max(nrx_ms) <= 45
            and all(row.get("output_finite") for row in nrx["records"])
        ),
        "atomic_injection_and_fifth_reject": structure_ok,
        "model_predicts_every_lease_decision": all(prediction_matches),
        "e6_one_ms_boundary_observed": (
            max(observed_decisions["E6a_latest_safe_context64"]) <= 88
            and min(observed_decisions["E6b_first_unsafe_context64"]) >= 89
        ),
        "qwen_physical_execution_and_fence": qwen_execution_ok,
        "physical_recovery25": recovery_ms and max(recovery_ms) <= 25,
        "single_commit_d155_semantics": (
            owner_semantics and max(commit_ms) <= 155
        ),
        "p2p_and_lifecycle": (
            all(access_values)
            and all(coordinator["preflight_echo_equal"])
            and all(nrx["preflight_echo_equal"])
            and coordinator["ipc_handles_closed_before_ack"] is True
            and nrx["ipc_handles_closed_before_ack"] is True
            and coordinator["qwen_stop_acknowledged"] is True
            and coordinator["error"] is None and nrx["error"] is None
        ),
    }
    value = {
        "schema": "softwall-c162-boundary-result-v1",
        "status": "C162_BOUNDARY_PASS" if all(gates.values()) else "C162_BOUNDARY_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "campaign": protocol["campaign"], "label": protocol["label"],
        "host": coordinator["host"], "slurm_job_id": coordinator["slurm_job_id"],
        "iterations": iterations, "case_counts": dict(case_counts),
        "observed_conservative_decision_ms": {
            case: summarize(values) for case, values in observed_decisions.items()
        },
        "counts": {
            "actual_nrx": len(nrx_ms), "injected_recoveries": len(recovery_ms),
            "qwen_units": sum(row["qwen"] is not None for row in rounds),
            "radio_commits": len(commit_ms), "deadline_misses": sum(row["deadline_miss"] for _, row in owner_records),
        },
        "maxima_ms": {
            "nrx_release_to_complete": max(nrx_ms),
            "recovery_path": max(recovery_ms),
            "radio_commit": max(commit_ms),
            "qwen_execution": max((row["qwen"]["execution_ms"] for row in qwen_expected), default=None),
        },
        "inventory": inventory,
        "artifact_sha256": {
            "protocol": sha256(args.protocol), "peer_spec": sha256(args.peer_spec),
            "coordinator": sha256(args.coordinator), "nrx_worker": sha256(args.nrx_worker),
            "qwen": sha256(args.qwen), "inventory": sha256(args.inventory),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temp = args.output.with_suffix(args.output.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
