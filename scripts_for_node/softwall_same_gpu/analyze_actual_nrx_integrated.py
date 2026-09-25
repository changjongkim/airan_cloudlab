#!/usr/bin/env python3.11
"""Gate one frozen actual-NeuralRx integrated execution."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_actual_nrx_integrated_protocol import source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    expected_success = protocol["expected_actual_success_keys"]
    expected_recovery = protocol["expected_actual_recovery_keys"]
    actual_success = [
        row["key"] for row in coordinator.get("actual_success_outcomes", [])
    ]
    actual_recovery = [
        row["key"] for row in coordinator.get("physical_recoveries", [])
    ]
    decisions = coordinator.get("reservation_decisions", [])
    mode = protocol["mode"]
    same_host = (
        nrx.get("host") == coordinator.get("host")
        and qwen.get("host") == coordinator.get("host")
        and all(row.get("host") == coordinator.get("host") for row in owners)
    )
    success_owners = [row for row in owners if row.get("key") in expected_success]
    recovery_owners = [row for row in owners if row.get("key") in expected_recovery]

    gates = {
        "source_hash_match": (
            source_hashes(args.scripts_root, args.task1_root)
            == protocol.get("source_sha256")
        ),
        "node_inventory_and_exclusion": (
            inventory_ok and same_host
            and coordinator.get("host") not in protocol.get("excluded_nodes", [])
        ),
        "local_global_divergence": (
            coordinator.get("local_certificates") == {"home0": True, "home1": True}
            and [row.get("key") for row in decisions] == protocol["offered_keys"]
            and sum(bool(row.get("accepted")) for row in decisions) == 4
            and [row.get("key") for row in decisions if not row.get("accepted")]
            == [protocol["expected_global_reject"]]
            and all(row.get("state_unchanged_on_reject") for row in decisions)
        ),
        "actual_nrx_path": (
            nrx.get("completed_units") == 4
            and nrx.get("finite_units") == 4
            and [row.get("key") for row in nrx.get("records", [])]
            == protocol["accepted_physical_keys"]
            and nrx.get("ipc_handles_closed_before_ack") is True
            and nrx.get("release_prewarm_lead_ms")
            == mode["nrx_release_prewarm_lead_ms"]
            and nrx.get("release_prewarm_completed_ns") is not None
            and nrx.get("error") is None
        ),
        "frozen_actual_outcome_pattern": (
            coordinator.get("missing_outcomes_at_cutoff") == []
            and actual_success == expected_success
            and [row.get("key") for row in coordinator.get(
                "observed_outcomes_at_cutoff", []
            ) if row.get("timely_success")] == expected_success
            and all(row.get("timely_success") for row in success_owners)
            and not any(row.get("timely_success") for row in recovery_owners)
        ),
        "v17_1_atomic_lease": (
            coordinator.get("lease_decision", {}).get("accepted") is True
            and coordinator.get("lease_decision", {}).get("reason") == "lease_committed"
            and coordinator.get("launch_revalidation", {}).get("host_ms")
            <= mode["launch_control_bound_ms"]
            and coordinator.get("qwen") is not None
            and coordinator["qwen"].get("fence_confirmed") is True
            and coordinator["qwen"].get("bound_violation") is False
            and coordinator["qwen"].get("launch_control_bound_violation") is False
            and coordinator["qwen"].get("lease_interval_violation") is False
            and coordinator.get("lease_retire", {}).get("accepted") is True
            and qwen.get("completed_units") == 1
            and not qwen.get("rejected")
        ),
        "certificate_recovery_conformance": (
            actual_recovery == expected_recovery
            and coordinator.get("certificate_order") == expected_recovery
            and not any(row.get("declared_path_bound_violation")
                        for row in coordinator.get("physical_recoveries", []))
        ),
        "same_tb_recovery_oracle": (
            len(recovery_owners) == 2
            and all(row.get("commit_source") == "shared_conventional"
                    for row in recovery_owners)
            and all(row.get("recovery_equivalent_to_local_oracle") is True
                    for row in recovery_owners)
            and all(row.get("dispatch_recovery") is True for row in recovery_owners)
        ),
        "actual_nrx_commit": (
            len(success_owners) == 2
            and all(row.get("commit_source") == "actual_nrx" for row in success_owners)
            and all(row.get("neural_correct") is True for row in success_owners)
            and all(row.get("dispatch_success") is True for row in success_owners)
        ),
        "single_timely_radio_commit": (
            all(row.get("commit_count") == 1 for row in owners)
            and all(row.get("input_ready_before_release") for row in owners)
            and not any(row.get("deadline_miss") for row in owners)
            and all(row.get("nrx_release_to_complete_ms", float("inf"))
                    <= mode["nrx_bound_ms"] for row in owners)
        ),
        "physical_lifecycle": (
            coordinator.get("ipc_handles_closed_before_ack") is True
            and coordinator.get("qwen_stop_acknowledged") is True
            and all(row.get("nrx_termination_acknowledged") is True
                    and row.get("recovery_termination_acknowledged") is True
                    for row in owners)
        ),
        "no_process_error": (
            coordinator.get("error") is None
            and nrx.get("error") is None
            and all(row.get("error") is None for row in owners)
        ),
    }
    artifacts = [
        args.protocol, args.peer_spec, args.coordinator, args.nrx_worker,
        args.qwen, args.inventory,
    ] + [Path(row["owner_output"]) for row in spec["peers"]]
    value = {
        "schema": "softwall-actual-nrx-integrated-result-v1",
        "status": (
            "ACTUAL_NRX_TRANSITION_PASS" if all(gates.values())
            else "ACTUAL_NRX_TRANSITION_FAIL"
        ),
        "campaign": protocol["campaign"],
        "job_id": coordinator.get("slurm_job_id"),
        "node": coordinator.get("host"),
        "protocol": str(args.protocol),
        "gates": gates,
        "all_pass": all(gates.values()),
        "summary": {
            "submitted_debts": len(decisions),
            "accepted_debts": sum(bool(row.get("accepted")) for row in decisions),
            "rejected_debts": sum(not bool(row.get("accepted")) for row in decisions),
            "actual_nrx_successes": len(actual_success),
            "physical_recoveries": len(actual_recovery),
            "oracle_equivalent_recoveries": sum(
                row.get("recovery_equivalent_to_local_oracle") is True
                for row in owners
            ),
            "qwen_units": qwen.get("completed_units"),
            "qwen_execution_ms": (coordinator.get("qwen") or {}).get(
                "execution_ms"
            ),
            "radio_commits": sum(row.get("commit_count", 0) for row in owners),
            "deadline_misses": sum(bool(row.get("deadline_miss")) for row in owners),
            "max_nrx_release_to_complete_ms": max(
                row.get("nrx_release_to_complete_ms", 0.0) for row in owners
            ),
            "max_radio_release_to_commit_ms": max(
                row.get("release_to_commit_ms", 0.0) for row in owners
            ),
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
        "status": value["status"], "gates": gates, "summary": value["summary"]
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
