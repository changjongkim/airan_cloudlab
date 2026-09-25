#!/usr/bin/env python3.11
"""Audit one C164 first-RAN-request-after-MPS-restart campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_c164_restart_protocol import restart_source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def evaluate(protocol: dict, base: dict, marker: dict,
             coordinator: dict, nrx: dict, qwen: dict,
             current_source_hashes: dict[str, str]) -> dict:
    contract = protocol.get("c164_restart", {})
    first = coordinator.get("rounds", [{}])[0] if coordinator.get("rounds") else {}
    after = marker.get("after", {})
    same_host_job = (
        marker.get("before", {}).get("host")
        == after.get("host") == coordinator.get("host")
        == nrx.get("host") == qwen.get("host")
        and marker.get("before", {}).get("slurm_job_id")
        == after.get("slurm_job_id") == coordinator.get("slurm_job_id")
        == nrx.get("slurm_job_id") == qwen.get("slurm_job_id")
    )
    gates = {
        "base_c162_boundary_all_pass": base.get("all_pass") is True,
        "frozen_restart_contract": (
            contract.get("schema") == "softwall-c164-mps-restart-protocol-v1"
            and contract.get("lifecycle") == "mps_restart_first"
            and "uninterrupted service" in contract.get("availability_scope", "")
        ),
        "c164_restart_source_hash_match": (
            contract.get("source_sha256") == current_source_hashes
        ),
        "physical_restart_marker_pass": (
            marker.get("schema") == "softwall-c164-mps-restart-marker-v1"
            and marker.get("status") == "MPS_RESTART_PROVEN"
            and marker.get("all_pass") is True
            and all(marker.get("checks", {}).values())
        ),
        "same_host_job_clock_domain": same_host_job,
        "new_epoch_before_first_release": (
            after.get("snapshot_ns", 0) < first.get("release_wall_ns", 0)
            and marker.get("created_ns", 0) < first.get("release_wall_ns", 0)
        ),
        "first_round_actual_nrx_complete": (
            len(first.get("physical_timely_successes", [])) == 4
            and not first.get("missing_outcomes_at_cutoff")
        ),
        "first_round_certificate_and_commit_safe": (
            bool(first.get("certificate_order"))
            and base.get("counts", {}).get("deadline_misses") == 0
            and base.get("gates", {}).get("single_commit_d155_semantics") is True
        ),
    }
    return {
        "schema": "softwall-c164-mps-restart-result-v1",
        "status": "C164_MPS_RESTART_BOUNDARY_PASS" if all(gates.values()) else "C164_MPS_RESTART_BOUNDARY_FAIL",
        "all_pass": all(gates.values()),
        "gates": gates,
        "campaign": base.get("campaign"), "label": base.get("label"),
        "host": coordinator.get("host"),
        "slurm_job_id": coordinator.get("slurm_job_id"),
        "first_round_case": first.get("boundary_case"),
        "counts": base.get("counts"), "maxima_ms": base.get("maxima_ms"),
        "restart_identity": {
            "before_control_pids": marker.get("before", {}).get("control_pids"),
            "after_control_pids": after.get("control_pids"),
            "before_server_pids": marker.get("before", {}).get("server_pids"),
            "after_server_pids": after.get("server_pids"),
            "before_control_inode": marker.get("before", {}).get("control_socket", {}).get("inode"),
            "after_control_inode": after.get("control_socket", {}).get("inode"),
        },
        "qualification_scope": (
            "First RAN request and frozen C162 boundary subset after a quiescent "
            "MPS daemon restart and full fresh-client requalification. Optional "
            "service is closed during restart; restart availability, full six-class "
            "qualification, production d_MAC, cross-family, and WCET remain UQ."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--nrx-worker", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    value = evaluate(
        load(args.protocol), load(args.base_result), load(args.marker),
        load(args.coordinator), load(args.nrx_worker), load(args.qwen),
        restart_source_hashes(args.scripts_root),
    )
    value["artifact_sha256"] = {
        "protocol": sha256(args.protocol), "base_result": sha256(args.base_result),
        "marker": sha256(args.marker), "coordinator": sha256(args.coordinator),
        "nrx_worker": sha256(args.nrx_worker), "qwen": sha256(args.qwen),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
