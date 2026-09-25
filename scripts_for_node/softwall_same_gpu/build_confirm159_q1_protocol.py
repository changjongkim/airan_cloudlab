#!/usr/bin/env python3.11
"""Freeze one C159-Q1 pre-staged P180 qualification campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


SCRIPT_SOURCES = (
    "c159_prestaged_owner.py",
    "c159_persistent_nrx_worker.py",
    "c159_q1_coordinator.py",
    "actual_nrx_repeated_control.py",
    "build_confirm159_q1_protocol.py",
    "analyze_confirm159_q1.py",
    "run_confirm159_q1.sh",
    "actual_nrx_shared_recovery_plan_v1.py",
    "actual_nrx_integrated_owner.py",
    "actual_nrx_multi_peer_worker.py",
    "actual_nrx_integrated_coordinator.py",
    "integrated_shared_recovery_launch_plan_v1.py",
    "integrated_shared_recovery_plan_v1.py",
    "shared_recovery_certificate_v1.py",
    "integrated_shared_recovery_worker.py",
    "shared_conventional_owner.py",
    "shared_conventional_worker.py",
    "multigpu_p2p_nrx_worker.py",
    "multigpu_p2p_ipc_gate.py",
    "dual_receiver_phy.py",
    "trace_qwen_worker.py",
)
TASK1_SOURCES = ("isca_v2/cuda_ipc_channel.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in SCRIPT_SOURCES}
    paths.update({f"task1/{name}": task1_root / name for name in TASK1_SOURCES})
    return {key: sha256(path) for key, path in paths.items()}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--peer-tsv", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign", choices=("development", "holdout"), required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--prespec", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--excluded-nodes", required=True)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--period-ms", type=float, default=180.0)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    config = IntegratedScenarioConfig(period_ms=round(args.period_ms))
    config.validate()
    if args.period_ms <= config.expiry_ms:
        parser.error("period must leave positive post-expiry bank-copy time")
    prespec = json.loads(args.prespec.read_text(encoding="utf-8"))
    if prespec.get("status") != "DESIGN_PRESPEC_HOLDOUT_NOT_MATERIALIZED":
        parser.error("C159 prespec status mismatch")
    if prespec.get("dataset", {}).get("holdout_materialized") is not False:
        parser.error("C159-Q1 requires an unopened holdout")

    accepted = REQUEST_ORDER[:4]
    expected_high_snr = (REQUEST_ORDER[0], REQUEST_ORDER[3])
    snr_by_key = {
        REQUEST_ORDER[0]: 20.0,
        REQUEST_ORDER[1]: -8.75,
        REQUEST_ORDER[2]: -8.75,
        REQUEST_ORDER[3]: 20.0,
    }
    peers = []
    for index, key in enumerate(accepted):
        home_index = int(key[0].removeprefix("home"))
        safe = f"{key[0]}_{key[1]}"
        peers.append({
            "key": list(key),
            "source_device": home_index,
            "receiver_seed": args.seed_base + index * 100_000 + 11,
            "channel_seed_base": args.seed_base + index * 100_000 + 10_001,
            "snr_db": snr_by_key[key],
            "nrx_tag": f"{args.label}_{safe}_nrx",
            "recovery_tag": f"{args.label}_{safe}_recovery",
            "ready_file": str(args.state_dir / f"{safe}_ready.json"),
            "decision_control": str(
                args.state_dir / f"{safe}_decision.ctrl"
            ),
            "owner_output": str(args.raw_dir / f"{args.label}_{safe}_owner.json"),
        })

    spec = {
        # Keep the established wire schema because the sealed P2P/cuPHY
        # loaders validate it before opening any CUDA IPC handle. C159 changes
        # lifecycle semantics in the protocol, not the peer descriptor shape.
        "schema": "softwall-actual-nrx-peer-spec-v1",
        "label": args.label,
        "peers": peers,
    }
    total_requests = args.iterations * len(peers)
    protocol = {
        "schema": "softwall-confirm159-q1-protocol-v1",
        "status": "frozen-before-run",
        "campaign": args.campaign,
        "label": args.label,
        "excluded_nodes": sorted({
            value for value in args.excluded_nodes.split(",") if value
        }),
        "seed_base": args.seed_base,
        "iterations": args.iterations,
        "actual_nrx_requests": total_requests,
        "mode": {
            "period_ms": args.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "ai_bound_ms": config.ai_bound_ms,
            "guard_ms": config.guard_ms,
            "launch_control_bound_ms": 5,
            "capacity": config.capacity,
            "release_lead_ms": args.release_lead_ms,
            "post_expiry_copy_budget_ms": args.period_ms - config.expiry_ms,
            "warmup": args.warmup,
            "qwen_context_length": 64,
            "qwen_model": "Qwen/Qwen2.5-1.5B",
            "lifecycle": (
                "persistent endpoints with full P2P/window/cuPHY/response-path "
                "startup preflight; all synthetic "
                "channels and validation-only local oracles are built before "
                "readiness; only GPU-bank to stable IPC-buffer copy occurs in "
                "the post-expiry interval before the next P180 release"
            ),
        },
        "placement": {
            "home0": "GPU0",
            "home1": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS 80/20",
            "persistent_actual_nrx": "GPU3 under MPS 80",
        },
        "offered_keys_per_round": [list(key) for key in REQUEST_ORDER],
        "accepted_physical_keys": [list(key) for key in accepted],
        "expected_global_reject": list(REQUEST_ORDER[4]),
        "high_snr_keys": [list(key) for key in expected_high_snr],
        "low_snr_keys": [
            list(key) for key in accepted if key not in expected_high_snr
        ],
        "snr_assignment_rule": (
            "two +20 dB stable-success keys and two -8.75 dB transition keys; "
            "the transition point was fixed from the earlier independent "
            "Confirm18 500-sample sweep (conventional 0/500, NeuralRx 271/500)"
        ),
        "outcome_rule": (
            "Every CRC-correct result completed by 45 ms resolves its mandatory "
            "debt; every other accepted request executes shared recovery. Outcome "
            "counts are measurements, not pass criteria."
        ),
        "recovery_oracle": (
            "worker echoes the received input to the owner for exact comparison; "
            "shared CRC pass requires exact transmitted-TB payload while CRC "
            "fail is a valid radio failure result; local/shared decoder-class "
            "disagreement is reported separately and is not a transport error"
        ),
        "gates": {
            "sample": f"all {total_requests} actual NeuralRx requests complete",
            "prestage": (
                "all input/oracle generation completes before readiness; each "
                "bank copy starts after prior D155 and completes before release"
            ),
            "warm_recovery_lifecycle": (
                "every shared recovery context completes the full P2P input, "
                "window install, cuPHY decode, and P2P response path before "
                "readiness; preflight latency is reported but excluded from "
                "the warm service bound"
            ),
            "nrx_bound": "zero release-to-complete values above 45 ms",
            "transition": "physical commit agrees with each observed outcome",
            "recovery": "zero same-input conventional semantic mismatches",
            "safety": "single commit and zero D155/path/lease/control violations",
            "lifecycle": "all CUDA IPC handles close before termination ack",
            "response_fence": (
                "NRx and recovery responses are copied back to the worker and "
                "byte-compared before the completion doorbell"
            ),
            "crc_wire_contract": (
                "shared recovery serializes the verified CRC decision as "
                "0=pass and 1=fail; raw cuPHY CRC bytes remain telemetry"
            ),
        },
        "scope": (
            "C159-Q1 P180 qualification, excluded from performance confidence "
            "intervals. Passing two nodes supports only the frozen context64 "
            "warm synthetic mode; variable-context trace performance, WCET, "
            "cold/long-idle, production d_MAC and integrated faults remain UQ."
        ),
        "prespec": {
            "path": str(args.prespec),
            "sha256": sha256(args.prespec),
            "status": prespec["status"],
            "holdout_materialized": False,
        },
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    atomic_json(args.peer_spec, spec)
    atomic_json(args.output, protocol)
    args.peer_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.peer_tsv.write_text("".join(
        "\t".join(map(str, (
            row["key"][0].removeprefix("home"), row["key"][1],
            row["source_device"], row["receiver_seed"],
            row["channel_seed_base"], row["snr_db"], row["nrx_tag"],
            row["recovery_tag"], row["ready_file"], row["decision_control"],
            row["owner_output"],
        ))) + "\n" for row in peers
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
