#!/usr/bin/env python3.11
"""Frozen-gate analyzer for the C153 shared-recovery integration canary."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

from build_confirm153_protocol import source_hashes
from integrated_shared_recovery_plan_v1 import (
    PHYSICAL_RECOVERY_KEYS,
    REJECTED_KEY,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def inventory_gate(path: Path) -> tuple[bool, list[dict]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    normalized = [
        {key.strip(): value.strip() for key, value in row.items()}
        for row in rows
    ]
    gate = (
        len(normalized) == 4
        and all("A100" in row.get("name", "") for row in normalized)
        and all(row.get("mig.mode.current", "").lower() == "disabled"
                for row in normalized)
    )
    return gate, normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--owner0", type=Path, required=True)
    parser.add_argument("--owner1", type=Path, required=True)
    parser.add_argument("--worker", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    protocol = load(args.protocol)
    owners = [load(args.owner0), load(args.owner1)]
    worker = load(args.worker)
    qwen = load(args.qwen)
    inventory_ok, inventory = inventory_gate(args.inventory)
    current_hashes = source_hashes(args.scripts_root, args.task1_root)
    decisions = worker.get("reservation_decisions", [])
    accepted = [row for row in decisions if row.get("accepted")]
    rejected = [row for row in decisions if not row.get("accepted")]
    rejected_keys = [(row.get("home_id"), row.get("request_id")) for row in rejected]
    physical = worker.get("physical_recoveries", [])
    physical_keys = [(row.get("home_id"), row.get("request_id")) for row in physical]
    transition = worker.get("outcome_transition", {})
    first = transition.get("lease_after_one_success", {})
    second = transition.get("lease_after_two_successes", {})
    config = worker.get("config", {})

    gates = {
        "source_hash_match": current_hashes == protocol.get("source_sha256"),
        "configuration_match": (
            config.get("period_ms") == protocol["mode"]["period_ms"]
            and config.get("expiry_ms") == protocol["mode"]["expiry_ms"]
            and config.get("nrx_bound_ms") == protocol["mode"]["nrx_bound_ms"]
            and config.get("conventional_bound_ms")
            == protocol["mode"]["conventional_bound_ms"]
            and config.get("ai_bound_ms") == protocol["mode"]["ai_bound_ms"]
            and config.get("guard_ms") == protocol["mode"]["guard_ms"]
            and config.get("release_lead_ms")
            == protocol["mode"]["release_lead_ms"]
        ),
        "four_a100_mig_off": inventory_ok,
        "local_feasible_global_fifth_rejected": (
            worker.get("local_certificates") == {"home0": True, "home1": True}
            and len(accepted) == 4
            and rejected_keys == [REJECTED_KEY]
            and rejected[0].get("reason") == "global_all_fail_infeasible"
            and rejected[0].get("state_unchanged_on_reject") is True
        ),
        "conditional_ai_threshold": (
            first.get("accepted") is False
            and first.get("reason") == "lease_breaks_global_certificate"
            and first.get("state_unchanged") is True
            and second.get("accepted") is True
            and second.get("reason") == "lease_committed"
        ),
        "qwen_physical_fence_and_bound": (
            worker.get("qwen", {}).get("fence_confirmed") is True
            and worker.get("qwen", {}).get("bound_violation") is False
            and worker.get("qwen", {}).get("lease_interval_violation") is False
            and worker.get("lease_retire", {}).get("accepted") is True
            and worker.get("qwen_stop_acknowledged") is True
            and qwen.get("completed_units") == 1
            and not qwen.get("rejected")
        ),
        "certificate_drives_physical_order": (
            physical_keys == list(PHYSICAL_RECOVERY_KEYS)
            and worker.get("certificate_order")
            == [list(key) for key in PHYSICAL_RECOVERY_KEYS]
            and all(row.get("correct") for row in physical)
            and not any(row.get("declared_path_bound_violation") for row in physical)
            and REJECTED_KEY not in physical_keys
        ),
        "home_commits_correct_and_timely": (
            len(owners) == 2
            and {(f"home{row.get('home_id')}", row.get("request_id"))
                 for row in owners} == set(PHYSICAL_RECOVERY_KEYS)
            and all(row.get("published") and row.get("correct") for row in owners)
            and all(row.get("input_ready_before_release") for row in owners)
            and not any(row.get("deadline_miss") for row in owners)
            and all(row.get("termination_acknowledged") for row in owners)
        ),
        "p2p_ipc_lifecycle": (
            worker.get("ipc_handles_closed_before_ack") is True
            and bool(worker.get("peer_access"))
            and all(bool(value) for value in worker.get("peer_access", {}).values())
        ),
        "no_process_error": (
            worker.get("error") is None
            and all(row.get("error") is None for row in owners)
        ),
    }
    value = {
        "schema": "softwall-confirm153-integrated-shared-recovery-result-v1",
        "protocol": str(args.protocol),
        "status": (
            "DEVELOPMENT_CANARY_PASS_V17_REMAINS_UQ"
            if all(gates.values()) else "DEVELOPMENT_CANARY_FAIL"
        ),
        "gates": gates,
        "all_pass": all(gates.values()),
        "inventory": inventory,
        "summary": {
            "accepted_debts": len(accepted),
            "rejected_debts": len(rejected),
            "physical_recoveries": len(physical),
            "correct_home_commits": sum(bool(row.get("correct")) for row in owners),
            "deadline_misses": sum(bool(row.get("deadline_miss")) for row in owners),
            "qwen_execution_ms": worker.get("qwen", {}).get("execution_ms"),
            "qwen_gpu_ms": worker.get("qwen", {}).get("gpu_ms"),
            "max_recovery_release_to_complete_ms": max(
                (row.get("release_to_complete_ms", 0.0) for row in physical),
                default=0.0,
            ),
            "max_home_release_to_commit_ms": max(
                (row.get("release_to_commit_ms", 0.0) for row in owners),
                default=0.0,
            ),
        },
        "artifact_sha256": {
            str(path): sha256(path)
            for path in (
                args.protocol, args.owner0, args.owner1, args.worker,
                args.qwen, args.inventory,
            )
        },
        "scope": (
            "Injected-outcome C153 mechanism canary only. Passing closes the "
            "model-to-shared-cuPHY/Qwen glue path but does not qualify actual "
            "NeuralRx outcomes, independent nodes, WCET, production d_MAC, "
            "fault coverage, throughput, or V17 QSU."
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
