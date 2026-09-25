#!/usr/bin/env python3.11
"""Gate one physical arm in a frozen C161 phase-2 campaign."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_c161_phase2_protocol import source_hashes


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


def injected_audit_ok(audit: dict, *, duplicate_reason: str) -> bool:
    return (
        audit.get("injected") is True
        and audit.get("duplicate", {}).get("accepted") is True
        and audit.get("duplicate", {}).get("reason") == duplicate_reason
        and audit.get("duplicate", {}).get("state_unchanged") is True
        and audit.get("stale", {}).get("accepted") is False
        and audit.get("stale", {}).get("reason") == "stale_or_future_epoch"
        and audit.get("stale", {}).get("state_unchanged") is True
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--arm-index", type=int, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--nrx-worker", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = load(args.protocol)
    arm_spec = protocol["arms"][args.arm_index]
    arm = arm_spec["arm"]
    spec_path = Path(arm_spec["peer_spec"])
    spec = load(spec_path)
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
    iterations = int(arm_spec["iterations"])
    expected_units = iterations * len(accepted)
    mode = protocol["mode"]
    bounds = {int(key): float(value) for key, value in mode["ai_class_bounds_ms"].items()}
    recovery_rows = coordinator.get("physical_recoveries", [])
    nrx_records = nrx.get("records", [])

    transition_ok = True
    for key, record in owner_records:
        row = round_by_sequence.get(int(record["sequence"]), {})
        effective = {tuple(value) for value in row.get("effective_timely_success_keys", [])}
        recovery = {tuple(value) for value in row.get("physical_recoveries", [])}
        transition_ok &= (
            (key in effective and key not in recovery
             and record.get("commit_source") == "actual_nrx")
            or (key not in effective and key in recovery
                and record.get("commit_source") == "shared_conventional")
        )

    common_semantics = all(
        row.get("fault_arm") == arm
        and {tuple(key) for key in row.get("effective_timely_success_keys", [])}
            == {tuple(key) for key in row.get("actual_timely_success_keys", [])}
        and {tuple(key) for key in row.get("physical_recoveries", [])}
            == accepted - {tuple(key) for key in row.get("success_keys", [])}
        and row.get("rejected_keys") == [protocol["expected_global_reject"]]
        and row.get("outcome_transition", {}).get("kind")
            == "atomic_observed_success_batch"
        for row in rounds
    )
    nrx_fault_audits = [
        row["nrx_event_audit"] for row in rounds
        if row.get("nrx_event_audit", {}).get("injected")
    ]
    recovery_fault_audits = [
        audit for row in rounds for audit in row.get("recovery_event_audits", [])
        if audit.get("injected")
    ]
    terminal = coordinator.get("terminal_fault")
    terminal_sequence = None if terminal is None else int(terminal["sequence"])
    post_fault_rounds = 0 if terminal_sequence is None else iterations - terminal_sequence
    phase2_semantics = False
    if arm == "stale_duplicate_nrx":
        phase2_semantics = (
            len(nrx_fault_audits) == mode["event_fault_target"]
            and coordinator.get("nrx_fault_count") == mode["event_fault_target"]
            and all(injected_audit_ok(audit, duplicate_reason="duplicate_transaction")
                    for audit in nrx_fault_audits)
            and not recovery_fault_audits and terminal is None
            and coordinator.get("ai_quarantined") is False
        )
    elif arm == "stale_duplicate_recovery":
        phase2_semantics = (
            len(recovery_fault_audits) == mode["event_fault_target"]
            and coordinator.get("recovery_fault_count") == mode["event_fault_target"]
            and all(injected_audit_ok(audit, duplicate_reason="duplicate_radio_commit")
                    for audit in recovery_fault_audits)
            and not nrx_fault_audits and terminal is None
            and coordinator.get("ai_quarantined") is False
        )
    elif arm == "post_fence_reply_delay":
        phase2_semantics = (
            terminal is not None and terminal.get("arm") == arm
            and terminal.get("fence_confirmed") is True
            and terminal.get("lease_retired") is True
            and coordinator.get("ambiguous_lease") is None
            and coordinator.get("ai_quarantined") is True
            and coordinator.get("qwen_launches_after_quarantine") == 0
            and post_fault_rounds >= mode["post_fault_min_radio_epochs"]
            and len(qwen.get("fault_events", [])) == 1
            and qwen["fault_events"][0].get("completion_marker_published") is True
        )
    elif arm == "pre_fence_channel_loss":
        phase2_semantics = (
            terminal is not None and terminal.get("arm") == arm
            and terminal.get("fence_confirmed") is False
            and terminal.get("lease_retired") is False
            and terminal.get("lease_retire_reason") == "physical_fence_required"
            and coordinator.get("ambiguous_lease") == terminal.get("request_id")
            and coordinator.get("ai_quarantined") is True
            and coordinator.get("qwen_launches_after_quarantine") == 0
            and post_fault_rounds >= mode["post_fault_min_radio_epochs"]
            and len(qwen.get("fault_events", [])) == 1
            and qwen["fault_events"][0].get("completion_marker_published") is False
        )

    attempts = [
        row["qwen_attempt"] for row in rounds if row.get("qwen_attempt") is not None
    ]
    attempt_ids = {row["request_id"] for row in attempts if row.get("launched")}
    worker_ids = {row["request_id"] for row in qwen.get("records", [])}
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
    terminal_arm = arm in {"post_fence_reply_delay", "pre_fence_channel_loss"}
    gates = {
        "source_hash_match": (
            source_hashes(args.scripts_root, args.task1_root) == protocol["source_sha256"]
            and sha256(Path(protocol["prespec"]["path"])) == protocol["prespec"]["sha256"]
            and protocol["prespec"]["holdout_materialized"] is False
        ),
        "node_inventory_and_exclusion": (
            inventory_ok(args.inventory)
            and coordinator.get("host") not in protocol["excluded_nodes"]
            and nrx.get("host") == coordinator.get("host")
            and qwen.get("host") == coordinator.get("host")
            and all(owner.get("host") == coordinator.get("host") for owner in owners)
        ),
        "complete_sample": (
            len(rounds) == coordinator.get("completed_rounds") == iterations
            and len(owner_records) == len(nrx_records) == expected_units
        ),
        "actual_nrx_bound_and_transport": (
            all(record["nrx_release_to_complete_ms"] <= mode["nrx_bound_ms"]
                and record.get("nrx_input_roundtrip_equal") is True
                for _, record in owner_records)
            and all(record.get("output_finite") is True
                    and record.get("backward_echo_equal") is True
                    for record in nrx_records)
        ),
        "prestage_and_recovery_preflight": (
            staging_ok
            and len(coordinator.get("recovery_preflight", [])) == len(accepted)
            and all(row.get("backward_echo_equal") is True
                    for row in coordinator.get("recovery_preflight", []))
        ),
        "common_transition_semantics": common_semantics,
        "phase2_fault_semantics": phase2_semantics,
        "qwen_identity_bound_and_start": (
            bool(attempts) and attempt_ids == worker_ids
            and all(row.get("launched") is True for row in attempts)
            and all(
                float(row["gpu_ms"]) <= bounds[int(row["context_length"])]
                and int(row["accepted_ns"]) <= int(row["latest_start_ns"])
                for row in qwen.get("records", [])
            )
            and not qwen.get("launch_guard_rejections")
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
            and ((not terminal_arm and coordinator.get("qwen_stop_acknowledged") is True)
                 or (terminal_arm and coordinator.get("qwen_fault_exit_expected") is True))
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
    marker_files = list(Path(arm_spec["completion_dir"]).glob("*.json"))
    artifacts = [
        args.protocol, spec_path, args.coordinator, args.nrx_worker,
        args.qwen, args.inventory, Path(protocol["prespec"]["path"]),
    ] + [Path(row["owner_output"]) for row in spec["peers"]] + marker_files
    summary = {
        "arm": arm,
        "rounds": len(rounds),
        "actual_nrx_requests": len(owner_records),
        "actual_nrx_successes": sum(bool(record.get("timely_success"))
                                    for _, record in owner_records),
        "physical_recoveries": len(recovery_rows),
        "qwen_worker_units": len(qwen.get("records", [])),
        "nrx_faults": len(nrx_fault_audits),
        "recovery_faults": len(recovery_fault_audits),
        "terminal_faults": int(terminal is not None),
        "post_fault_rounds": post_fault_rounds,
        "radio_commits": sum(record.get("commit_count", 0) for _, record in owner_records),
        "deadline_misses": sum(bool(record.get("deadline_miss")) for _, record in owner_records),
        "nrx_max_ms": max(record["nrx_release_to_complete_ms"] for _, record in owner_records),
        "recovery_max_ms": max(
            ((row["actual_completed_ns"] - row["actual_start_ns"]) / 1e6
             for row in recovery_rows), default=0.0,
        ),
        "qwen_gpu_max_ms": max((row["gpu_ms"] for row in qwen.get("records", [])), default=0.0),
        "commit_max_ms": max(record["release_to_commit_ms"] for _, record in owner_records),
    }
    value = {
        "schema": "softwall-c161-phase2-arm-result-v1",
        "status": "C161_PHASE2_ARM_PASS" if all(gates.values()) else "C161_PHASE2_ARM_FAIL",
        "campaign": protocol["campaign"],
        "arm_index": args.arm_index,
        "arm": arm,
        "job_id": coordinator.get("slurm_job_id"),
        "node": coordinator.get("host"),
        "all_pass": all(gates.values()),
        "gates": gates,
        "summary": summary,
        "scope": protocol["scope"],
        "artifact_sha256": {str(path): sha256(path) for path in artifacts if path.is_file()},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps({"status": value["status"], "gates": gates, "summary": summary},
                     indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
