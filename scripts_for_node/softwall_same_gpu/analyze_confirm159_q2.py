#!/usr/bin/env python3.11
"""Gate one frozen C159-Q2 variable-context P180 qualification run."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_confirm159_q2_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def percentile(values: list[float], fraction: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[round((len(ordered) - 1) * fraction)]


def summary(values: list[float]) -> dict:
    return {
        "count": len(values),
        "mean": (sum(values) / len(values)) if values else None,
        "p50": percentile(values, 0.50),
        "p99": percentile(values, 0.99),
        "max": max(values) if values else None,
    }


def gpu_inventory(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    valid = (
        len(rows) == 4
        and all("A100" in row.get("name", "") for row in rows)
        and all(row.get("mig.mode.current", "").lower() == "disabled" for row in rows)
    )
    return valid, rows


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

    protocol = load(args.protocol)
    spec = load(args.peer_spec)
    coordinator = load(args.coordinator)
    nrx = load(args.nrx_worker)
    qwen = load(args.qwen)
    owners = [load(Path(row["owner_output"])) for row in spec["peers"]]
    inventory_ok, inventory = gpu_inventory(args.inventory)
    iterations = int(protocol["iterations"])
    accepted_keys = [tuple(key) for key in protocol["accepted_physical_keys"]]
    accepted_set = set(accepted_keys)
    expected_units = iterations * len(accepted_keys)
    mode = protocol["mode"]

    same_host = (
        nrx.get("host") == coordinator.get("host")
        and qwen.get("host") == coordinator.get("host")
        and all(row.get("host") == coordinator.get("host") for row in owners)
    )
    owner_by_key = {tuple(row["key"]): row for row in owners}
    owner_records = [
        (tuple(owner["key"]), record)
        for owner in owners for record in owner.get("records", [])
    ]
    nrx_records = nrx.get("records", [])
    rounds = coordinator.get("rounds", [])
    round_by_sequence = {int(row["sequence"]): row for row in rounds}
    expected_pairs = {
        (sequence, key) for sequence in range(1, iterations + 1)
        for key in accepted_keys
    }
    owner_pairs = {
        (int(record["sequence"]), key) for key, record in owner_records
    }
    nrx_pairs = {
        (int(record["sequence"]), tuple(record["key"]))
        for record in nrx_records
    }

    outcome_consistent = True
    staging_consistent = True
    for key, record in owner_records:
        sequence = int(record["sequence"])
        round_row = round_by_sequence.get(sequence, {})
        success = {tuple(value) for value in round_row.get("success_keys", [])}
        recovery = {
            tuple(value) for value in round_row.get("physical_recoveries", [])
        }
        timely = bool(record.get("timely_success"))
        outcome_consistent &= (
            (timely and key in success and key not in recovery
             and record.get("commit_source") == "actual_nrx")
            or ((not timely) and key not in success and key in recovery
                and record.get("commit_source") == "shared_conventional")
        )
        staging_consistent &= (
            record.get("preparation_started_ns", -1)
            >= owner_by_key[key].get("prestage_completed_ns", 2**63)
            and record.get("preparation_completed_ns", 2**63)
            <= record.get("release_wall_ns", -1)
            and record.get("bank_copy_gpu_ms", float("inf"))
            <= mode["post_expiry_copy_budget_ms"]
        )
        if sequence > 1:
            previous_release = record["release_wall_ns"] - round(
                mode["period_ms"] * 1e6
            )
            staging_consistent &= (
                record.get("preparation_started_ns", -1)
                >= previous_release + round(mode["expiry_ms"] * 1e6)
            )

    round_structure = all(
        int(row.get("sequence", -1)) == sequence
        and row.get("missing_outcomes_at_cutoff") == []
        and {tuple(key) for key in row.get("success_keys", [])}.issubset(accepted_set)
        and {tuple(key) for key in row.get("physical_recoveries", [])}
            == accepted_set - {tuple(key) for key in row.get("success_keys", [])}
        and row.get("rejected_keys") == [protocol["expected_global_reject"]]
        for sequence, row in enumerate(rounds, start=1)
    )
    atomic_outcome_batch = all(
        row.get("outcome_transition", {}).get("kind")
            == "atomic_observed_success_batch"
        and {
            tuple(key) for key in row["outcome_transition"].get(
                "observed_successes", []
            )
        } == {tuple(key) for key in row.get("success_keys", [])}
        and {
            tuple(key) for key in row["outcome_transition"].get(
                "unresolved_obligations", []
            )
        } == accepted_set - {tuple(key) for key in row.get("success_keys", [])}
        for row in rounds
    )
    qwen_rounds = [row for row in rounds if row.get("qwen") is not None]
    context_lengths = tuple(int(value) for value in protocol["ai_offer"]["order"])
    class_bounds = {
        int(key): float(value)
        for key, value in mode["ai_class_bounds_ms"].items()
    }
    offered_by_class = {
        length: sum(int(row.get("context_length", -1)) == length for row in rounds)
        for length in context_lengths
    }
    executed_by_class = {
        length: sum(
            row.get("qwen") is not None
            and int(row.get("context_length", -1)) == length
            for row in rounds
        )
        for length in context_lengths
    }
    certificate_rejected_by_class = {
        length: sum(
            row.get("lease_accepted") is False
            and int(row.get("context_length", -1)) == length
            for row in rounds
        )
        for length in context_lengths
    }
    launch_guard_rejected_by_class = {
        length: sum(
            row.get("lease_accepted") is True
            and row.get("qwen") is None
            and row.get("qwen_attempt", {}).get("launched") is False
            and int(row.get("context_length", -1)) == length
            for row in rounds
        )
        for length in context_lengths
    }
    class_schedule_consistent = all(
        int(row.get("context_length", -1))
            == context_lengths[(sequence - 1) % len(context_lengths)]
        and float(row.get("ai_bound_ms", -1))
            == class_bounds[context_lengths[(sequence - 1) % len(context_lengths)]]
        and (
            row.get("qwen") is None
            or int(row["qwen"].get("context_length", -1))
                == int(row["context_length"])
        )
        and (
            row.get("qwen_attempt") is None
            or int(row["qwen_attempt"].get("context_length", -1))
                == int(row["context_length"])
        )
        for sequence, row in enumerate(rounds, start=1)
    )
    qwen_ids = {
        row["qwen"]["request_id"] for row in qwen_rounds
    }
    worker_ids = {
        row["request_id"] for row in qwen.get("records", [])
    }
    revalidation_consistent = all(
        1 <= len(row.get("launch_revalidation_attempts", [])) <= 3
        and row["launch_revalidation_attempts"][-1].get("within_budget") is True
        and all(attempt.get("within_budget") is False
                for attempt in row["launch_revalidation_attempts"][:-1])
        and float(row.get("launch_revalidation_host_ms", float("inf")))
            <= mode["launch_control_bound_ms"]
        for row in rounds
    )
    physical_start_guard_ok = all(
        int(row["qwen"]["worker_accepted_ns"])
            <= int(row["qwen"]["latest_start_ns"])
        for row in qwen_rounds
    ) and all(
        int(row["accepted_ns"]) <= int(row["latest_start_ns"])
        for row in qwen.get("records", [])
    )
    rejected_start_guard_ok = all(
        int(row["accepted_ns"]) > int(row["latest_start_ns"])
        and row.get("reason") == "latest_start_expired_before_gpu_launch"
        for row in qwen.get("launch_guard_rejections", [])
    )
    lease_consistent = all(
        (
            row.get("lease_accepted") is True
            and row.get("lease_reason") == "lease_committed"
            and row.get("qwen", {}).get("fence_confirmed") is True
            and row["qwen"].get("bound_violation") is False
            and row["qwen"].get("launch_control_bound_violation") is False
            and row["qwen"].get("lease_interval_violation") is False
            and row.get("lease_retire", {}).get("accepted") is True
        ) if row.get("qwen") is not None else (
            row.get("lease_accepted") is True
            and row.get("lease_reason") == "lease_committed"
            and row.get("qwen_attempt", {}).get("launched") is False
            and row.get("qwen_attempt", {}).get("reason")
                == "latest_start_expired_before_gpu_launch"
            and row.get("qwen_attempt", {}).get("fence_confirmed") is True
            and row.get("qwen_attempt", {}).get(
                "launch_control_bound_violation"
            ) is False
            and row.get("lease_retire", {}).get("accepted") is True
        ) if row.get("qwen_attempt") is not None else (
            row.get("lease_accepted") is False
            and row.get("lease_retire") is None
        )
        for row in rounds
    )

    nrx_release_ms = [
        float(record["nrx_release_to_complete_ms"])
        for _, record in owner_records
    ]
    radio_commit_ms = [
        float(record["release_to_commit_ms"])
        for _, record in owner_records
    ]
    recovery_path_ms = [
        (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
        for row in coordinator.get("physical_recoveries", [])
    ]
    recovery_records = [
        record for _, record in owner_records
        if record.get("commit_source") == "shared_conventional"
    ]
    success_records = [
        record for _, record in owner_records
        if record.get("commit_source") == "actual_nrx"
    ]
    access_values = list(nrx.get("peer_access", {}).values()) + list(
        coordinator.get("peer_access", {}).values()
    )
    recovery_preflight = coordinator.get("recovery_preflight", [])

    gates = {
        "source_hash_match": (
            source_hashes(args.scripts_root, args.task1_root)
            == protocol.get("source_sha256")
            and sha256(Path(protocol["prespec"]["path"]))
                == protocol["prespec"]["sha256"]
            and protocol["prespec"].get("holdout_materialized") is False
        ),
        "node_inventory_and_exclusion": (
            inventory_ok and same_host
            and coordinator.get("host") not in protocol.get("excluded_nodes", [])
        ),
        "complete_request_sample": (
            coordinator.get("completed_rounds") == iterations
            and len(rounds) == iterations
            and nrx.get("completed_units") == expected_units
            and nrx.get("finite_units") == expected_units
            and owner_pairs == expected_pairs
            and nrx_pairs == expected_pairs
            and all(owner.get("completed_iterations") == iterations for owner in owners)
        ),
        "nrx_bound": (
            len(nrx_release_ms) == expected_units
            and max(nrx_release_ms, default=float("inf")) <= mode["nrx_bound_ms"]
            and all(record.get("output_finite") for record in nrx_records)
            and all(record.get("backward_echo_equal") is True
                    for record in nrx_records)
            and all(record.get("nrx_input_roundtrip_equal") is True
                    for _, record in owner_records)
        ),
        "global_certificate_each_round": round_structure,
        "outcome_driven_transition": outcome_consistent,
        "atomic_observed_outcome_batch": atomic_outcome_batch,
        "pre_staged_input_and_p180_copy": (
            staging_consistent
            and nrx.get("lifecycle")
                == "startup preflight only; no per-epoch activation"
            and all(owner.get("pre_staged_iterations") == iterations
                    for owner in owners)
            and all(owner.get("prestage_completed_ns") is not None
                    for owner in owners)
            and all(owner.get("bank_bytes", 0) > 0 for owner in owners)
        ),
        "full_recovery_path_preflight": (
            len(recovery_preflight) == len(accepted_keys)
            and all(row.get("backward_echo_equal") is True
                    for row in recovery_preflight)
            and all(float(row.get("path_ms", 0.0)) > 0.0
                    for row in recovery_preflight)
        ),
        "atomic_qwen_lease": (
            lease_consistent
            and qwen.get("completed_units") == len(qwen_rounds)
            and qwen.get("rejected") == []
            and len(qwen.get("launch_guard_rejections", []))
                == sum(launch_guard_rejected_by_class.values())
            and coordinator.get("qwen_units") == len(qwen_rounds)
        ),
        "bounded_revalidation_and_physical_start_guard": (
            revalidation_consistent
            and physical_start_guard_ok
            and rejected_start_guard_ok
        ),
        "variable_context_class_qualification": (
            tuple(int(value) for value in qwen.get("allowed_context_lengths", []))
                == context_lengths
            and tuple(int(value) for value in coordinator.get(
                "context_lengths", []
            )) == context_lengths
            and {
                int(key): float(value)
                for key, value in coordinator.get("class_bounds_ms", {}).items()
            } == class_bounds
            and class_schedule_consistent
            and all(
                offered_by_class[length]
                    == protocol["ai_offer"]["offered_per_class"]
                for length in context_lengths
            )
            and all(
                executed_by_class[length]
                    >= protocol["ai_offer"]["minimum_executed_per_class"]
                for length in context_lengths
            )
            and all(
                offered_by_class[length]
                == executed_by_class[length]
                    + certificate_rejected_by_class[length]
                    + launch_guard_rejected_by_class[length]
                for length in context_lengths
            )
            and all(
                float(row["qwen"]["execution_ms"])
                    <= class_bounds[int(row["context_length"])]
                for row in qwen_rounds
            )
            and qwen_ids == worker_ids
        ),
        "certificate_recovery_conformance": (
            len(recovery_records)
            == coordinator.get("physical_recovery_count")
            and all(row.get("recovery_contract_valid") is True
                    for row in recovery_records)
            and all(row.get("recovery_input_roundtrip_equal") is True
                    for row in recovery_records)
            and all(
                row.get("recovery_crc") != 0
                or row.get("recovery_payload_matches_reference") is True
                for row in recovery_records
            )
            and not any(row.get("declared_path_bound_violation")
                        for row in coordinator.get("physical_recoveries", []))
            and all(row.get("backward_echo_equal") is True
                    for row in coordinator.get("physical_recoveries", []))
            and all(row.get("serialized_crc_status")
                    == int(row.get("crc_failures", 0) != 0)
                    for row in coordinator.get("physical_recoveries", []))
            and max(recovery_path_ms, default=0.0)
                <= mode["conventional_bound_ms"]
        ),
        "single_timely_radio_commit": (
            len(owner_records) == expected_units
            and all(row.get("commit_count") == 1 for _, row in owner_records)
            and all(row.get("input_ready_before_release") for _, row in owner_records)
            and not any(row.get("deadline_miss") for _, row in owner_records)
            and max(radio_commit_ms, default=float("inf")) <= mode["expiry_ms"]
        ),
        "physical_p2p_ipc_lifecycle": (
            bool(access_values) and all(value == 1 for value in access_values)
            and all(nrx.get("preflight_echo_equal", []))
            and len(nrx.get("preflight_echo_equal", [])) == len(accepted_keys)
            and all(coordinator.get("preflight_echo_equal", []))
            and len(coordinator.get("preflight_echo_equal", []))
                == len(accepted_keys)
            and nrx.get("ipc_handles_closed_before_ack") is True
            and coordinator.get("ipc_handles_closed_before_ack") is True
            and coordinator.get("qwen_stop_acknowledged") is True
            and all(owner.get("nrx_termination_acknowledged") is True
                    and owner.get("recovery_termination_acknowledged") is True
                    for owner in owners)
        ),
        "no_process_error": (
            coordinator.get("error") is None
            and nrx.get("error") is None
            and all(owner.get("error") is None for owner in owners)
        ),
    }

    artifacts = [
        args.protocol, args.peer_spec, args.coordinator, args.nrx_worker,
        args.qwen, args.inventory, Path(protocol["prespec"]["path"]),
    ] + [Path(row["owner_output"]) for row in spec["peers"]]
    value = {
        "schema": "softwall-confirm159-q2-result-v1",
        "status": (
            "C159_Q2_VARIABLE_P180_PASS" if all(gates.values())
            else "C159_Q2_VARIABLE_P180_FAIL"
        ),
        "campaign": protocol["campaign"],
        "job_id": coordinator.get("slurm_job_id"),
        "node": coordinator.get("host"),
        "protocol": str(args.protocol),
        "gates": gates,
        "all_pass": all(gates.values()),
        "summary": {
            "rounds": len(rounds),
            "actual_nrx_requests": len(owner_records),
            "actual_nrx_successes": len(success_records),
            "physical_recoveries": len(recovery_records),
            "oracle_equivalent_recoveries": sum(
                row.get("recovery_local_class_equal") is True
                for row in recovery_records
            ),
            "local_shared_class_disagreements": sum(
                row.get("recovery_local_class_equal") is False
                for row in recovery_records
            ),
            "raw_zero_normalized_failures": sum(
                row.get("raw_response_crc") == 0
                and row.get("serialized_crc_status") == 1
                for row in coordinator.get("physical_recoveries", [])
            ),
            "qwen_units": len(qwen_rounds),
            "qwen_offered_by_class": {
                str(key): value for key, value in offered_by_class.items()
            },
            "qwen_executed_by_class": {
                str(key): value for key, value in executed_by_class.items()
            },
            "qwen_certificate_rejected_by_class": {
                str(key): value
                for key, value in certificate_rejected_by_class.items()
            },
            "qwen_launch_guard_rejected_by_class": {
                str(key): value
                for key, value in launch_guard_rejected_by_class.items()
            },
            "launch_revalidation_retry_rounds": sum(
                len(row.get("launch_revalidation_attempts", [])) > 1
                for row in rounds
            ),
            "launch_revalidation_attempts": summary([
                float(len(row.get("launch_revalidation_attempts", [])))
                for row in rounds
            ]),
            "qwen_execution_ms_by_class": {
                str(length): summary([
                    float(row["qwen"]["execution_ms"])
                    for row in qwen_rounds
                    if int(row["context_length"]) == length
                ])
                for length in context_lengths
            },
            "radio_commits": sum(row.get("commit_count", 0)
                                 for _, row in owner_records),
            "deadline_misses": sum(bool(row.get("deadline_miss"))
                                   for _, row in owner_records),
            "bank_copy_gpu_ms": summary([
                float(row["bank_copy_gpu_ms"]) for _, row in owner_records
            ]),
            "prestage_ms": summary([
                float(owner["prestage_ms"]) for owner in owners
            ]),
            "recovery_preflight_path_ms": summary([
                float(row["path_ms"]) for row in recovery_preflight
            ]),
            "nrx_release_to_complete_ms": summary(nrx_release_ms),
            "radio_release_to_commit_ms": summary(radio_commit_ms),
            "recovery_path_ms": summary(recovery_path_ms),
            "qwen_execution_ms": summary([
                float(row["qwen"]["execution_ms"]) for row in qwen_rounds
            ]),
        },
        "inventory": inventory,
        "scope": protocol["scope"],
        "artifact_sha256": {str(path): sha256(path) for path in artifacts},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"],
        "gates": gates,
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
