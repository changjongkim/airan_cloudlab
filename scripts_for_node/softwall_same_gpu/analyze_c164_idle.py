#!/usr/bin/env python3.11
"""Audit one C164 first-request-after-idle physical boundary campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from build_c164_idle_protocol import c164_source_hashes


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--base-result", type=Path, required=True)
    parser.add_argument("--marker", type=Path, required=True)
    parser.add_argument("--coordinator", type=Path, required=True)
    parser.add_argument("--nrx-worker", type=Path, required=True)
    parser.add_argument("--qwen", type=Path, required=True)
    parser.add_argument("--lifecycle-model", type=Path, required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol, base, marker = map(load, (
        args.protocol, args.base_result, args.marker
    ))
    coordinator, nrx, qwen, model = map(load, (
        args.coordinator, args.nrx_worker, args.qwen, args.lifecycle_model
    ))
    c164 = protocol.get("c164", {})
    first = coordinator["rounds"][0] if coordinator.get("rounds") else {}
    required_ns = round(float(c164.get("required_idle_s", 0)) * 1e9)
    same_host_job = (
        marker.get("host") == coordinator.get("host") == nrx.get("host")
        == qwen.get("host")
        and marker.get("slurm_job_id") == coordinator.get("slurm_job_id")
        == nrx.get("slurm_job_id") == qwen.get("slurm_job_id")
    )
    gates = {
        "base_c162_boundary_all_pass": base.get("all_pass") is True,
        "lifecycle_model_pass": model.get("status") == "C164_LIFECYCLE_MODEL_PASS",
        "c164_source_hash_match": (
            c164.get("source_sha256") == c164_source_hashes(args.scripts_root)
        ),
        "frozen_idle_mode": (
            c164.get("schema") == "softwall-c164-idle-boundary-protocol-v1"
            and c164.get("lifecycle") == "idle_30s_first"
            and required_ns > 0
        ),
        "same_host_job_clock_domain": same_host_job,
        "idle_after_readiness_before_schedule": (
            marker.get("schema") == "softwall-c164-lifecycle-idle-marker-v1"
            and marker.get("clock") == "time.perf_counter_ns"
            and marker.get("status") == "idle_complete_before_schedule_publish"
            and marker.get("first_release_wall_ns")
                == first.get("release_wall_ns")
            and marker.get("idle_started_ns", 0)
                < marker.get("idle_completed_ns", 0)
                < marker.get("first_release_wall_ns", 0)
        ),
        "required_idle_physically_observed": (
            marker.get("idle_gate") is True
            and marker.get("observed_idle_ns", -1) >= required_ns
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
    value = {
        "schema": "softwall-c164-idle-boundary-result-v1",
        "status": "C164_IDLE30_BOUNDARY_PASS" if all(gates.values()) else "C164_IDLE30_BOUNDARY_FAIL",
        "all_pass": all(gates.values()), "gates": gates,
        "campaign": base.get("campaign"), "label": base.get("label"),
        "host": coordinator.get("host"),
        "slurm_job_id": coordinator.get("slurm_job_id"),
        "required_idle_s": required_ns / 1e9,
        "observed_idle_s": marker.get("observed_idle_ns", 0) / 1e9,
        "first_round_case": first.get("boundary_case"),
        "counts": base.get("counts"), "maxima_ms": base.get("maxima_ms"),
        "qualification_scope": (
            "First-request-after-30s-idle boundary subset on this node. Full "
            "six-class lifecycle qualification, restart, and WCET remain UQ."
        ),
        "artifact_sha256": {
            "protocol": sha256(args.protocol),
            "base_result": sha256(args.base_result),
            "marker": sha256(args.marker),
            "coordinator": sha256(args.coordinator),
            "nrx_worker": sha256(args.nrx_worker), "qwen": sha256(args.qwen),
            "lifecycle_model": sha256(args.lifecycle_model),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(value, indent=2, sort_keys=True))
    raise SystemExit(0 if value["all_pass"] else 1)


if __name__ == "__main__":
    main()
