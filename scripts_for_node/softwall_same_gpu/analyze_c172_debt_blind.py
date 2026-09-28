#!/usr/bin/env python3.11
"""Gate one frozen C172 physical necessity campaign."""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
from pathlib import Path

from build_c172_debt_blind_protocol import source_hashes
from c172_debt_blind_cases import (
    SCENARIOS,
    scenario_for_sequence,
    timing_cell_contains,
)


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summarize(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": sum(values) / len(values) if values else None,
        "p50": percentile(values, 0.50),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
        "min": min(values) if values else None,
    }


def rule_of_three_zero_upper(n: int):
    return None if n <= 0 else 3.0 / n


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
    coordinator = load(args.coordinator)
    nrx, qwen = load(args.nrx_worker), load(args.qwen)
    owners = [load(Path(row["owner_output"])) for row in spec["peers"]]
    valid_inventory, inventory = inventory_ok(args.inventory)
    iterations = int(protocol["iterations"])
    accepted = tuple(tuple(key) for key in protocol["accepted_physical_keys"])
    accepted_set = set(accepted)
    expected_pairs = {
        (sequence, key)
        for sequence in range(1, iterations + 1) for key in accepted
    }
    owner_records = [
        (tuple(owner["key"]), row) for owner in owners for row in owner["records"]
    ]
    owner_pairs = {(row["sequence"], key) for key, row in owner_records}
    nrx_pairs = {(row["sequence"], tuple(row["key"])) for row in nrx["records"]}
    rounds = coordinator["rounds"]
    rounds_by_sequence = {row["sequence"]: row for row in rounds}
    recovery_by_sequence: dict[int, list[dict]] = collections.defaultdict(list)
    for recovery in coordinator["physical_recoveries"]:
        recovery_by_sequence[recovery["sequence"]].append(recovery)
    owner_by_sequence: dict[int, list[tuple[tuple[str, str], dict]]] = collections.defaultdict(list)
    for key, row in owner_records:
        owner_by_sequence[row["sequence"]].append((key, row))

    same_host = (
        coordinator["host"] == nrx["host"] == qwen["host"] == protocol["host"]
        and all(owner["host"] == coordinator["host"] for owner in owners)
    )
    attempt_counts = collections.Counter()
    scenario_counts = collections.Counter()
    timing_invalid_counts = collections.Counter()
    unused_timing_valid_counts = collections.Counter()
    observed_decisions: dict[str, list[float]] = collections.defaultdict(list)
    guard_violations = collections.Counter()
    deadline_violations = collections.Counter()
    all_attempt_guard_violations = collections.Counter()
    all_attempt_deadline_violations = collections.Counter()
    bound_valid_rounds = collections.Counter()
    bound_valid_guard_violations = collections.Counter()
    bound_valid_deadline_violations = collections.Counter()
    prediction_matches: list[bool] = []
    qwen_ok: list[bool] = []
    owner_structure_ok = True
    round_structure_ok = True
    recovery_ms: list[float] = []
    recovery_accuracy: list[bool] = []
    recovery_upper_ok: list[bool] = []
    control_ms: list[float] = []
    qwen_ms: list[float] = []
    qwen_accuracy: list[bool] = []
    qwen_upper_ok: list[bool] = []
    guard_ms = float(protocol["service_vector_ms"]["radio_expiry"] -
                     protocol["service_vector_ms"]["completion_guard"])
    expiry_ms = float(protocol["service_vector_ms"]["radio_expiry"])
    qwen_lower_tol = float(
        protocol["service_vector_ms"]["qwen_lower_accuracy_tolerance"]
    )
    recovery_lower_tol = float(
        protocol["service_vector_ms"]["recovery_lower_accuracy_tolerance"]
    )

    target_samples = int(protocol["target_samples_per_scenario"])
    for sequence, row in enumerate(rounds, start=1):
        scenario = scenario_for_sequence(sequence, protocol["reverse_scenarios"])
        name = scenario.scenario_id
        attempt_counts[name] += 1
        observed = float(row["observed_decision_ms"])
        observed_decisions[name].append(observed)
        timing_cell_ok = timing_cell_contains(scenario, observed)
        selected = timing_cell_ok and scenario_counts[name] < target_samples
        if selected:
            scenario_counts[name] += 1
        elif timing_cell_ok:
            unused_timing_valid_counts[name] += 1
        else:
            timing_invalid_counts[name] += 1

        successes = set(map(tuple, row["success_keys"]))
        failures = set(map(tuple, row["injected_failure_keys"]))
        recoveries = set(map(tuple, row["physical_recoveries"]))
        attempts = row.get("launch_revalidation_attempts", [])
        round_structure_ok &= (
            row["sequence"] == sequence
            and row["scenario_id"] == name
            and row["policy"] == scenario.policy
            and row["boundary_case"] == scenario.case_id
            and row["context_length"] == scenario.context_length
            and row["missing_outcomes_at_cutoff"] == []
            and set(map(tuple, row["physical_timely_successes"])) == accepted_set
            and successes == set(accepted[:scenario.success_count])
            and failures == accepted_set - successes
            and recoveries == failures
            and row["rejected_keys"] == [protocol["expected_global_reject"]]
            and len(attempts) >= 1
            and attempts[-1]["within_budget"] is True
        )
        rows_for_round = owner_by_sequence[sequence]
        commit_max = max(item["release_to_commit_ms"] for _, item in rows_for_round)
        guard_crossed = float(row["last_recovery_complete_ms"]) > guard_ms
        deadline_crossed = commit_max > expiry_ms or any(
            item["deadline_miss"] for _, item in rows_for_round
        )
        all_attempt_guard_violations[name] += int(guard_crossed)
        all_attempt_deadline_violations[name] += int(deadline_crossed)

        for key, item in rows_for_round:
            success = key in successes
            owner_structure_ok &= (
                item["timely_success"] is True
                and item["neural_correct"] is True
                and item["commit_count"] == 1
                and ((success and item["commit_source"] == "actual_nrx")
                     or ((not success)
                         and item["commit_source"] == "shared_conventional"))
            )
        if not selected:
            continue

        prediction_matches.append(
            bool(row["lease_accepted"]) == scenario.expected_softwall_lease
        )
        guard_violations[name] += int(guard_crossed)
        deadline_violations[name] += int(deadline_crossed)
        launched_expected = scenario.policy in {"debt_blind", "shadow"} or (
            scenario.policy == "softwall" and scenario.expected_softwall_lease
        )
        qwen_bound_valid = True
        control_bound_valid = True
        if launched_expected:
            q = row["qwen_attempt"]
            qwen_ok.append(
                q is not None and q["launched"] is True
                and q["fence_confirmed"] is True
                and (row["lease_retire"] is None
                     if scenario.policy in {"debt_blind", "shadow"}
                     else row["lease_retire"] is not None)
            )
            if q is not None and q.get("launched"):
                bound = float(row["ai_bound_ms"])
                elapsed = float(q["execution_ms"])
                qwen_ms.append(elapsed)
                control = float(q["accept_after_revalidation_ms"])
                control_ms.append(control)
                lower_ok = elapsed >= bound - qwen_lower_tol
                upper_ok = elapsed <= bound
                qwen_accuracy.append(lower_ok)
                qwen_upper_ok.append(upper_ok)
                qwen_bound_valid = lower_ok and upper_ok
                control_bound_valid = control <= 5.0
        else:
            qwen_ok.append(
                row["qwen"] is None and row["qwen_attempt"] is None
                and row["lease_accepted"] is False
                and row["lease_retire"] is None
            )

        round_recovery_valid = True
        for recovery in recovery_by_sequence[sequence]:
            elapsed = (
                recovery["actual_completed_ns"] - recovery["actual_start_ns"]
            ) / 1e6
            lower_ok = elapsed >= 25.0 - recovery_lower_tol
            upper_ok = elapsed <= 25.0
            recovery_ms.append(elapsed)
            recovery_accuracy.append(lower_ok)
            recovery_upper_ok.append(upper_ok)
            round_recovery_valid &= lower_ok and upper_ok
        if qwen_bound_valid and control_bound_valid and round_recovery_valid:
            bound_valid_rounds[name] += 1
            bound_valid_guard_violations[name] += int(guard_crossed)
            bound_valid_deadline_violations[name] += int(deadline_crossed)

    per_scenario = {}
    for scenario in SCENARIOS:
        n = scenario_counts[scenario.scenario_id]
        g = guard_violations[scenario.scenario_id]
        d = deadline_violations[scenario.scenario_id]
        per_scenario[scenario.scenario_id] = {
            "attempts": attempt_counts[scenario.scenario_id],
            "timing_invalid_attempts": timing_invalid_counts[scenario.scenario_id],
            "unused_timing_valid_attempts": unused_timing_valid_counts[scenario.scenario_id],
            "n": n,
            "guard_violations": g,
            "guard_violation_fraction": g / n if n else None,
            "deadline_violations": d,
            "deadline_violation_fraction": d / n if n else None,
            "jointly_bound_valid_rounds": bound_valid_rounds[scenario.scenario_id],
            "bound_valid_guard_violations": (
                bound_valid_guard_violations[scenario.scenario_id]
            ),
            "bound_valid_deadline_violations": (
                bound_valid_deadline_violations[scenario.scenario_id]
            ),
            "zero_guard_95pct_rule_of_three_upper": (
                rule_of_three_zero_upper(n) if g == 0 else None
            ),
            "zero_deadline_95pct_rule_of_three_upper": (
                rule_of_three_zero_upper(n) if d == 0 else None
            ),
            "decision_ms": summarize(observed_decisions[scenario.scenario_id]),
            "all_attempt_guard_violations": (
                all_attempt_guard_violations[scenario.scenario_id]
            ),
            "all_attempt_deadline_violations": (
                all_attempt_deadline_violations[scenario.scenario_id]
            ),
        }

    samples = protocol["target_samples_per_scenario"]
    attempts_per_scenario = protocol["attempts_per_scenario"]
    forced_threshold = math.ceil(
        samples * protocol["minimum_effect_size"]["forced_guard_violation_fraction"]
    )
    jointly_valid_required = int(
        protocol["minimum_effect_size"]["jointly_bound_valid_rounds_per_scenario"]
    )
    expected_violation_gate = (
        all(
            bound_valid_rounds[row.scenario_id] >= jointly_valid_required
            for row in SCENARIOS
        )
        and bound_valid_guard_violations["E4_debt_blind_launch"]
            >= forced_threshold
        and bound_valid_deadline_violations["E4_debt_blind_launch"]
            >= forced_threshold
        and bound_valid_guard_violations["E6b_shadow_launch"]
            >= forced_threshold
        and deadline_violations["E6b_shadow_launch"] == 0
        and all(
            guard_violations[name] == 0 and deadline_violations[name] == 0
            for name in (
                "E4_softwall_reject", "E6a_softwall_admit",
                "E6b_softwall_reject",
            )
        )
    )
    access_values = list(nrx["peer_access"].values()) + list(
        coordinator["peer_access"].values()
    )
    nrx_ms = [row["nrx_release_to_complete_ms"] for _, row in owner_records]
    accuracy_fraction = float(
        protocol["minimum_effect_size"]["padding_lower_accuracy_fraction"]
    )
    upper_fraction = float(
        protocol["minimum_effect_size"]["padding_upper_compliance_fraction"]
    )
    gates = {
        "frozen_source_and_capacity_hash_match": (
            source_hashes(args.scripts_root, args.task1_root)
                == protocol["source_sha256"]
            and sha256(Path(protocol["capacity_model"]["path"]))
                == protocol["capacity_model"]["sha256"]
        ),
        "node_inventory_and_prespecified_exclusion": (
            valid_inventory and same_host
            and coordinator["host"] not in
                protocol["node_exclusion_rule"]["excluded_before_allocation"]
        ),
        "complete_physical_sample": (
            len(rounds) == iterations
            and owner_pairs == expected_pairs and nrx_pairs == expected_pairs
            and nrx["completed_units"] == len(expected_pairs)
            and all(owner["completed_iterations"] == iterations for owner in owners)
        ),
        "prespecified_scenario_sample": (
            set(attempt_counts) == {row.scenario_id for row in SCENARIOS}
            and all(value == attempts_per_scenario
                    for value in attempt_counts.values())
            and set(scenario_counts) == {row.scenario_id for row in SCENARIOS}
            and all(value == samples for value in scenario_counts.values())
        ),
        "physical_nrx45_stable_success": (
            nrx_ms and max(nrx_ms) <= 45
            and all(row.get("output_finite") for row in nrx["records"])
        ),
        "atomic_injection_and_owner_semantics": (
            round_structure_ok and owner_structure_ok
        ),
        "model_predicts_every_softwall_decision": all(prediction_matches),
        "qwen_launch_and_fence_semantics": all(qwen_ok),
        "launch_control_within_declared_bound": (
            control_ms and max(control_ms) <= 5.0
        ),
        "qwen_bound_realization": (
            qwen_accuracy and sum(qwen_accuracy) / len(qwen_accuracy) >= accuracy_fraction
            and sum(qwen_upper_ok) / len(qwen_upper_ok) >= upper_fraction
        ),
        "recovery_bound_realization": (
            recovery_accuracy
            and sum(recovery_accuracy) / len(recovery_accuracy) >= accuracy_fraction
            and sum(recovery_upper_ok) / len(recovery_upper_ok) >= upper_fraction
        ),
        "predicted_guard_and_deadline_outcomes": expected_violation_gate,
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
    all_qwen_rows = [
        row["qwen_attempt"] for row in rounds
        if row.get("qwen_attempt") and row["qwen_attempt"].get("launched")
    ]
    all_recovery_ms = [
        (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
        for row in coordinator["physical_recoveries"]
    ]
    result = {
        "schema": "softwall-c172-debt-blind-result-v1",
        "status": "C172_PASS" if all(gates.values()) else "C172_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "campaign": protocol["campaign"],
        "label": protocol["label"],
        "host": coordinator["host"],
        "slurm_job_id": coordinator["slurm_job_id"],
        "iterations": iterations,
        "per_scenario": per_scenario,
        "counts": {
            "physical_attempts": len(rounds),
            "selected_boundary_rounds": sum(scenario_counts.values()),
            "timing_invalid_attempts": sum(timing_invalid_counts.values()),
            "unused_timing_valid_attempts": sum(
                unused_timing_valid_counts.values()
            ),
            "actual_nrx": len(nrx_ms),
            "recoveries": len(recovery_ms),
            "qwen_units": len(qwen_ms),
            "radio_commits": len(owner_records),
            "observed_deadline_misses": sum(
                row["deadline_miss"] for _, row in owner_records
            ),
        },
        "service_accuracy": {
            "launch_control_ms": summarize(control_ms),
            "qwen_transaction_ms": summarize(qwen_ms),
            "qwen_lower_accuracy_fraction": (
                sum(qwen_accuracy) / len(qwen_accuracy) if qwen_accuracy else None
            ),
            "qwen_upper_compliance_fraction": (
                sum(qwen_upper_ok) / len(qwen_upper_ok) if qwen_upper_ok else None
            ),
            "recovery_path_ms": summarize(recovery_ms),
            "recovery_lower_accuracy_fraction": (
                sum(recovery_accuracy) / len(recovery_accuracy)
                if recovery_accuracy else None
            ),
            "recovery_upper_compliance_fraction": (
                sum(recovery_upper_ok) / len(recovery_upper_ok)
                if recovery_upper_ok else None
            ),
        },
        "all_attempt_diagnostics": {
            "qwen_units": len(all_qwen_rows),
            "qwen_upper_overshoots": sum(
                row["execution_ms"] > row["declared_bound_ms"]
                for row in all_qwen_rows
            ),
            "launch_control_upper_overshoots": sum(
                row["accept_after_revalidation_ms"] > 5.0
                for row in all_qwen_rows
            ),
            "recovery_units": len(all_recovery_ms),
            "recovery_upper_overshoots": sum(
                value > 25.0 for value in all_recovery_ms
            ),
        },
        "claim_boundary": protocol["analysis_role"],
        "inventory": inventory,
        "artifact_sha256": {
            "protocol": sha256(args.protocol),
            "peer_spec": sha256(args.peer_spec),
            "coordinator": sha256(args.coordinator),
            "nrx_worker": sha256(args.nrx_worker),
            "qwen": sha256(args.qwen),
            "inventory": sha256(args.inventory),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
