#!/usr/bin/env python3.11
"""Freeze one actual-NeuralRx integrated campaign before GPU execution."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


SCRIPT_SOURCES = (
    "actual_nrx_shared_recovery_plan_v1.py",
    "actual_nrx_integrated_owner.py",
    "actual_nrx_multi_peer_worker.py",
    "actual_nrx_integrated_coordinator.py",
    "build_actual_nrx_integrated_protocol.py",
    "analyze_actual_nrx_integrated.py",
    "run_actual_nrx_integrated.sh",
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
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--excluded-nodes", required=True)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    config = IntegratedScenarioConfig()
    accepted = REQUEST_ORDER[:4]
    expected_success = (REQUEST_ORDER[0], REQUEST_ORDER[3])
    snr_by_key = {
        REQUEST_ORDER[0]: 20.0,
        REQUEST_ORDER[1]: -15.0,
        REQUEST_ORDER[2]: -15.0,
        REQUEST_ORDER[3]: 20.0,
    }
    peers = []
    for index, key in enumerate(accepted):
        home_index = int(key[0].removeprefix("home"))
        safe = f"{key[0]}_{key[1]}"
        peers.append({
            "key": list(key),
            "source_device": home_index,
            "receiver_seed": args.seed_base + index * 100 + 11,
            "channel_seed": args.seed_base + index * 100 + 51,
            "snr_db": snr_by_key[key],
            "nrx_tag": f"{args.label}_{safe}_nrx",
            "recovery_tag": f"{args.label}_{safe}_recovery",
            "ready_file": str(args.state_dir / f"{safe}_ready.json"),
            "outcome_file": str(args.state_dir / f"{safe}_outcome.json"),
            "owner_output": str(args.raw_dir / f"{args.label}_{safe}_owner.json"),
        })
    spec = {
        "schema": "softwall-actual-nrx-peer-spec-v1",
        "label": args.label,
        "peers": peers,
    }
    protocol = {
        "schema": "softwall-actual-nrx-integrated-protocol-v1",
        "status": "frozen-before-run",
        "campaign": args.campaign,
        "label": args.label,
        "excluded_nodes": sorted({
            value for value in args.excluded_nodes.split(",") if value
        }),
        "seed_base": args.seed_base,
        "mode": {
            "period_ms": config.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "ai_bound_ms": config.ai_bound_ms,
            "guard_ms": config.guard_ms,
            "launch_control_bound_ms": 5,
            "capacity": config.capacity,
            "release_lead_ms": args.release_lead_ms,
            "warmup": args.warmup,
            "nrx_release_prewarm_lead_ms": 100.0,
            "lifecycle": "persistent warm endpoint with per-peer activation 100 ms before release",
            "qwen_context_length": 64,
            "qwen_model": "Qwen/Qwen2.5-1.5B",
        },
        "placement": {
            "home0": "GPU0",
            "home1": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS 80/20",
            "persistent_actual_nrx": "GPU3 under MPS 80",
        },
        "offered_keys": [list(key) for key in REQUEST_ORDER],
        "accepted_physical_keys": [list(key) for key in accepted],
        "expected_global_reject": list(REQUEST_ORDER[4]),
        "expected_actual_success_keys": [list(key) for key in expected_success],
        "expected_actual_recovery_keys": [
            list(key) for key in accepted if key not in expected_success
        ],
        "snr_assignment_rule": (
            "one +20 dB expected-success request per home and two -15 dB "
            "expected-failure requests; fixed before outcomes are opened"
        ),
        "outcome_source": "actual TensorRT NeuralRx CRC before the frozen 45 ms cutoff",
        "recovery_oracle": (
            "same-input local conventional semantics: successful CRC requires "
            "payload equality; failed CRC requires the shared path to fail CRC, "
            "because failed-CRC payload bits are not radio-valid output"
        ),
        "gates": {
            "actual_outcome_pattern": "exactly the two frozen +20 dB keys succeed",
            "global_reject": "the fifth 3+2 debt is rejected before GPU launch",
            "qwen": "one context-64 unit inside a V17.1 launch-time lease",
            "recovery": "two -15 dB requests match local conventional CRC semantics",
            "safety": "single commit, zero deadline/bound/credit/lifecycle violations",
        },
        "scope": (
            "Finite-sample actual-NeuralRx transition canary. Passing does not "
            "establish WCET, cold/long-idle qualification, production d_MAC, "
            "fault coverage, throughput superiority, or cross-GPU-family generality."
        ),
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    for path, value in ((args.peer_spec, spec), (args.output, protocol)):
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        temporary.replace(path)
    args.peer_tsv.parent.mkdir(parents=True, exist_ok=True)
    args.peer_tsv.write_text("".join(
        "\t".join(map(str, (
            row["key"][0].removeprefix("home"), row["key"][1],
            row["source_device"], row["receiver_seed"], row["channel_seed"],
            row["snr_db"], row["nrx_tag"], row["recovery_tag"],
            row["ready_file"], row["outcome_file"], row["owner_output"],
        ))) + "\n" for row in peers
    ), encoding="utf-8")


if __name__ == "__main__":
    main()
