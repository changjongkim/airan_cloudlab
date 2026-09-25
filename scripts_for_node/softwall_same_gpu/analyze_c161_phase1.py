#!/usr/bin/env python3.11
"""Gate one frozen C161 phase-1 P180 physical fault campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

from build_c161_phase1_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory_ok(path: Path) -> bool:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    return (
        len(rows) == 4
        and all("A100" in row.get("name", "") for row in rows)
        and all(row.get("mig.mode.current", "").lower() == "disabled"
                for row in rows)
    )


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
    owner_records = [
        (tuple(owner["key"]), record)
        for owner in owners for record in owner.get("records", [])
    ]
    rounds = coordinator.get("rounds", [])
    round_by_sequence = {int(row["sequence"]): row for row in rounds}
    accepted = {tuple(key) for key in protocol["accepted_physical_keys"]}
    iterations = int(protocol["iterations"])
    expected_units = iterations * len(accepted)
    mode = protocol["mode"]
    bounds = {
        int(key): float(value)
        for key, value in mode["ai_class_bounds_ms"].items()
    }
    arm_order = tuple(protocol["fault_plan"]["arm_order"])
    arm_length = int(protocol["fault_plan"]["arm_length"])
    arm_counts = Counter(row.get("fault_arm") for row in rounds)

    transition_ok = True
    for key, record in owner_records:
        row = round_by_sequence.get(int(record["sequence"]), {})
        effective = {
            tuple(value) for value in row.get("effective_timely_success_keys", [])
        }
        recovery = {
            tuple(value) for value in row.get("physical_recoveries", [])
        }
        transition_ok &= (
            (key in effective and key not in recovery
             and record.get("commit_source") == "actual_nrx")
            or (key not in effective and key in recovery
                and record.get("commit_source") == "shared_conventional")
        )

    arm_semantics = True
    guarded_faults = 0
    nofault_qwen = 0
    for index, row in enumerate(rounds):
        expected_arm = arm_order[index // arm_length]
        actual = {tuple(key) for key in row["actual_timely_success_keys"]}
        effective = {tuple(key) for key in row["effective_timely_success_keys"]}
        recoveries = {tuple(key) for key in row["physical_recoveries"]}
        arm_semantics &= row["fault_arm"] == expected_arm
        arm_semantics &= recoveries == accepted - effective
        arm_semantics &= {
            tuple(key) for key in row["success_keys"]
        } == effective
        arm_semantics &= row["rejected_keys"] == [
            protocol["expected_global_reject"]
        ]
        if expected_arm == "no_fault":
            arm_semantics &= effective == actual
            if row.get("lease_accepted"):
                arm_semantics &= (
                    row.get("qwen", {}).get("launched") is True
                    and row.get("qwen_attempt", {}).get("launched") is True
                    and row.get("lease_retire", {}).get("accepted") is True
                )
                nofault_qwen += 1
        elif expected_arm == "correlated_all_fail":
            arm_semantics &= not effective
            arm_semantics &= len(recoveries) == 4
            arm_semantics &= row.get("lease_accepted") is False
            arm_semantics &= row.get("qwen") is None
            arm_semantics &= row.get("qwen_attempt") is None
        elif expected_arm == "latest_start_nonlaunch":
            arm_semantics &= effective == actual
            if row.get("lease_accepted"):
                attempt = row.get("qwen_attempt", {})
                arm_semantics &= (
                    row.get("injected_host_hold_ms", 0.0)
                        >= protocol["fault_plan"]["fault_hold_ms"] * 0.90
                    and attempt.get("launched") is False
                    and attempt.get("reason")
                        == "latest_start_expired_before_gpu_launch"
                    and attempt.get("fence_confirmed") is True
                    and row.get("qwen") is None
                    and row.get("lease_retire", {}).get("accepted") is True
                )
                guarded_faults += 1

    recovery_rows = coordinator.get("physical_recoveries", [])
    qwen_rounds = [row for row in rounds if row.get("qwen") is not None]
    qwen_ids = {row["qwen"]["request_id"] for row in qwen_rounds}
    worker_ids = {row["request_id"] for row in qwen.get("records", [])}
    nrx_records = nrx.get("records", [])
    access = list(nrx.get("peer_access", {}).values()) + list(
        coordinator.get("peer_access", {}).values()
    )
    owner_by_key = {tuple(owner["key"]): owner for owner in owners}
    staging_ok = all(
        record.get("preparation_started_ns", -1)
            >= owner_by_key[key].get("prestage_completed_ns", 2**63)
        and record.get("preparation_completed_ns", 2**63)
            <= record.get("release_wall_ns", -1)
        and record.get("bank_copy_gpu_ms", float("inf"))
            <= mode["post_expiry_copy_budget_ms"]
        for key, record in owner_records
    )

    gates = {
        "source_hash_match": (
            source_hashes(args.scripts_root, args.task1_root)
                == protocol["source_sha256"]
            and sha256(Path(protocol["prespec"]["path"]))
                == protocol["prespec"]["sha256"]
            and protocol["prespec"]["holdout_materialized"] is False
        ),
        "node_inventory_and_exclusion": (
            inventory_ok(args.inventory)
            and coordinator.get("host") not in protocol["excluded_nodes"]
            and nrx.get("host") == coordinator.get("host")
            and qwen.get("host") == coordinator.get("host")
            and all(owner.get("host") == coordinator.get("host")
                    for owner in owners)
        ),
        "complete_sample_and_arm_order": (
            len(rounds) == coordinator.get("completed_rounds") == iterations
            and len(owner_records) == len(nrx_records) == expected_units
            and all(arm_counts[arm] == arm_length for arm in arm_order)
        ),
        "actual_nrx_bound_and_transport": (
            all(record["nrx_release_to_complete_ms"] <= mode["nrx_bound_ms"]
                and record.get("nrx_input_roundtrip_equal") is True
                for _, record in owner_records)
            and all(record.get("output_finite") is True
                    and record.get("backward_echo_equal") is True
                    for record in nrx_records)
        ),
        "pre_staged_input_and_full_preflight": (
            staging_ok
            and len(coordinator.get("recovery_preflight", [])) == len(accepted)
            and all(row.get("backward_echo_equal") is True
                    for row in coordinator.get("recovery_preflight", []))
        ),
        "fault_arm_semantics": arm_semantics,
        "correlated_all_fail_physically_recovered": (
            sum(
                len(row["physical_recoveries"]) for row in rounds
                if row["fault_arm"] == "correlated_all_fail"
            ) == arm_length * 4
        ),
        "latest_start_nonlaunch_exercised": guarded_faults > 0,
        "nofault_qwen_exercised": nofault_qwen > 0,
        "atomic_batch_all_rounds": all(
            row.get("outcome_transition", {}).get("kind")
                == "atomic_observed_success_batch"
            for row in rounds
        ),
        "qwen_bound_identity_and_guard": (
            qwen_ids == worker_ids
            and all(
                float(row["qwen"]["execution_ms"])
                    <= bounds[int(row["context_length"])]
                and int(row["qwen"]["worker_accepted_ns"])
                    <= int(row["qwen"]["latest_start_ns"])
                for row in qwen_rounds
            )
            and len(qwen.get("launch_guard_rejections", [])) == guarded_faults
        ),
        "recovery_bound_transport_and_contract": (
            all(
                (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
                    <= mode["conventional_bound_ms"]
                and row.get("backward_echo_equal") is True
                and row.get("declared_path_bound_violation") is False
                for row in recovery_rows
            )
            and all(
                record.get("recovery_contract_valid") is True
                and record.get("recovery_input_roundtrip_equal") is True
                for _, record in owner_records
                if record.get("commit_source") == "shared_conventional"
            )
        ),
        "single_timely_radio_commit": (
            transition_ok
            and all(record.get("commit_count") == 1
                    and record.get("deadline_miss") is False
                    and record.get("release_to_commit_ms") <= mode["expiry_ms"]
                    for _, record in owner_records)
        ),
        "physical_p2p_ipc_lifecycle": (
            access and all(value == 1 for value in access)
            and nrx.get("ipc_handles_closed_before_ack") is True
            and coordinator.get("ipc_handles_closed_before_ack") is True
            and coordinator.get("qwen_stop_acknowledged") is True
            and all(owner.get("nrx_termination_acknowledged") is True
                    and owner.get("recovery_termination_acknowledged") is True
                    for owner in owners)
        ),
        "no_process_error": (
            coordinator.get("error") is None and nrx.get("error") is None
            and all(owner.get("error") is None for owner in owners)
            and qwen.get("rejected") == []
        ),
    }
    artifacts = [
        args.protocol, args.peer_spec, args.coordinator, args.nrx_worker,
        args.qwen, args.inventory, Path(protocol["prespec"]["path"]),
    ] + [Path(row["owner_output"]) for row in spec["peers"]]
    value = {
        "schema": "softwall-c161-phase1-result-v1",
        "status": "C161_PHASE1_PASS" if all(gates.values()) else "C161_PHASE1_FAIL",
        "campaign": protocol["campaign"],
        "job_id": coordinator.get("slurm_job_id"),
        "node": coordinator.get("host"),
        "protocol": str(args.protocol),
        "all_pass": all(gates.values()),
        "gates": gates,
        "summary": {
            "rounds": len(rounds),
            "actual_nrx_requests": len(owner_records),
            "actual_nrx_successes": sum(
                bool(record.get("timely_success")) for _, record in owner_records
            ),
            "effective_successes": sum(
                len(row["effective_timely_success_keys"]) for row in rounds
            ),
            "physical_recoveries": len(recovery_rows),
            "qwen_units": len(qwen_rounds),
            "guarded_nonlaunches": guarded_faults,
            "radio_commits": sum(record.get("commit_count", 0)
                                 for _, record in owner_records),
            "deadline_misses": sum(bool(record.get("deadline_miss"))
                                   for _, record in owner_records),
            "arm_counts": dict(arm_counts),
            "correlated_recoveries": sum(
                len(row["physical_recoveries"]) for row in rounds
                if row["fault_arm"] == "correlated_all_fail"
            ),
            "nrx_max_ms": max(
                record["nrx_release_to_complete_ms"]
                for _, record in owner_records
            ),
            "recovery_max_ms": max(
                (row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
                for row in recovery_rows
            ),
            "qwen_max_ms": max(
                (row["qwen"]["execution_ms"] for row in qwen_rounds),
                default=0.0,
            ),
            "commit_max_ms": max(
                record["release_to_commit_ms"] for _, record in owner_records
            ),
        },
        "scope": protocol["scope"],
        "artifact_sha256": {
            str(path): sha256(path) for path in artifacts if path.is_file()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({
        "status": value["status"], "gates": gates,
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
