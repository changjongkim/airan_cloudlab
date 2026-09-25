#!/usr/bin/env python3.11
"""Analyze one prospective C154/C155 integrated holdout branch."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_confirm154_holdout_protocol import source_hashes
from integrated_shared_recovery_holdout_plan_v1 import build_branch


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inventory(path: Path) -> tuple[bool, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(handle)
        ]
    return (
        len(rows) == 4
        and all("A100" in row.get("name", "") for row in rows)
        and all(row.get("mig.mode.current", "").lower() == "disabled" for row in rows),
        rows,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.protocol)
    spec = load(args.peer_spec)
    worker = load(args.worker)
    qwen = load(args.qwen)
    owners = [load(Path(row["owner_output"])) for row in spec["peers"]]
    branch = spec["branch"]
    expected = build_branch(branch)
    inventory_ok, inventory_rows = inventory(args.inventory)
    expected_keys = [list(key) for key in expected["physical_recovery_keys"]]
    physical = worker.get("physical_recoveries", [])
    physical_keys = [row.get("key") for row in physical]
    decisions = worker.get("reservation_decisions", [])
    decision_keys = [row.get("key") for row in decisions]
    rejected_keys = [row.get("key") for row in decisions if not row.get("accepted")]
    success_keys = [row.get("key") for row in worker.get("success_outcomes", [])]
    owner_keys = [[f"home{row.get('home_id')}", row.get("request_id")] for row in owners]
    same_host = (
        all(row.get("host") == worker.get("host") for row in owners)
        and qwen.get("host") == worker.get("host")
    )
    mode = protocol["mode"]

    if expected["ai_expected"]:
        qwen_gate = (
            worker.get("qwen") is not None
            and worker["qwen"].get("fence_confirmed") is True
            and worker["qwen"].get("bound_violation") is False
            and worker["qwen"].get("lease_interval_violation") is False
            and worker.get("lease_retire", {}).get("accepted") is True
            and qwen.get("completed_units") == 1
            and not qwen.get("rejected")
        )
    else:
        qwen_gate = (
            worker.get("qwen") is None
            and worker.get("lease_retire") is None
            and qwen.get("completed_units") == 0
            and not qwen.get("rejected")
        )

    peer_gate = worker.get("ipc_handles_closed_before_ack") is True
    if owners:
        peer_gate = (
            peer_gate
            and bool(worker.get("peer_access"))
            and all(bool(value) for value in worker.get("peer_access", {}).values())
        )
    else:
        peer_gate = peer_gate and worker.get("peer_access") == {}

    gates = {
        "source_hash_match": (
            source_hashes(args.scripts_root, args.task1_root)
            == protocol.get("source_sha256")
        ),
        "node_and_inventory": (
            inventory_ok and same_host
            and worker.get("host") not in protocol.get("excluded_nodes", [])
        ),
        "configuration_match": (
            worker.get("branch") == branch
            and mode["period_ms"] == worker["config"]["period_ms"]
            and mode["expiry_ms"] == worker["config"]["expiry_ms"]
            and mode["nrx_bound_ms"] == worker["config"]["nrx_bound_ms"]
            and mode["conventional_bound_ms"]
            == worker["config"]["conventional_bound_ms"]
            and mode["ai_bound_ms"] == worker["config"]["ai_bound_ms"]
            and mode["guard_ms"] == worker["config"]["guard_ms"]
            and mode["release_lead_ms"] == worker["release_lead_ms"]
        ),
        "local_and_global_decisions": (
            worker.get("local_certificates") == {"home0": True, "home1": True}
            and decision_keys == [list(key) for key in expected["submitted_keys"]]
            and sum(bool(row.get("accepted")) for row in decisions) == 4
            and rejected_keys == [list(key) for key in expected["rejected_keys"]]
            and all(row.get("state_unchanged_on_reject") for row in decisions)
        ),
        "outcomes_and_lease": (
            success_keys == [list(key) for key in expected["success_keys"]]
            and worker.get("lease_decision", {}).get("accepted")
            == expected["ai_expected"]
            and worker.get("lease_decision", {}).get("reason")
            == ("lease_committed" if expected["ai_expected"]
                else "lease_breaks_global_certificate")
            and worker.get("lease_decision", {}).get(
                "state_unchanged_on_reject", True
            )
        ),
        "qwen_contract": qwen_gate and worker.get("qwen_stop_acknowledged") is True,
        "certificate_physical_conformance": (
            physical_keys == expected_keys
            and worker.get("certificate_order") == expected_keys
            and all(row.get("correct") for row in physical)
            and not any(row.get("declared_path_bound_violation") for row in physical)
        ),
        "home_commit_correct_and_timely": (
            owner_keys == expected_keys
            and all(row.get("published") and row.get("correct") for row in owners)
            and all(row.get("input_ready_before_release") for row in owners)
            and not any(row.get("deadline_miss") for row in owners)
            and all(row.get("termination_acknowledged") for row in owners)
        ),
        "p2p_ipc_lifecycle": peer_gate,
        "no_process_error": (
            worker.get("error") is None
            and all(row.get("error") is None for row in owners)
        ),
    }
    value = {
        "schema": "softwall-confirm154-integrated-holdout-arm-result-v1",
        "protocol": str(args.protocol),
        "peer_spec": str(args.peer_spec),
        "branch": branch,
        "arm_index": spec["arm_index"],
        "job_id": worker.get("slurm_job_id"),
        "node": worker.get("host"),
        "gates": gates,
        "all_pass": all(gates.values()),
        "summary": {
            "submitted_debts": len(decisions),
            "accepted_debts": sum(bool(row.get("accepted")) for row in decisions),
            "rejected_debts": len(rejected_keys),
            "success_outcomes": len(success_keys),
            "qwen_units": qwen.get("completed_units"),
            "qwen_execution_ms": (
                worker.get("qwen", {}).get("execution_ms")
                if worker.get("qwen") else None
            ),
            "physical_recoveries": len(physical),
            "correct_home_commits": sum(bool(row.get("correct")) for row in owners),
            "deadline_misses": sum(bool(row.get("deadline_miss")) for row in owners),
            "max_home_release_to_commit_ms": max(
                (row.get("release_to_commit_ms", 0.0) for row in owners),
                default=0.0,
            ),
        },
        "inventory": inventory_rows,
        "artifact_sha256": {
            str(path): sha256(path)
            for path in (args.protocol, args.peer_spec, args.worker, args.qwen, args.inventory)
        } | {
            str(Path(row["owner_output"])): sha256(Path(row["owner_output"]))
            for row in spec["peers"]
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps({
        "branch": branch,
        "all_pass": value["all_pass"],
        "summary": value["summary"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

