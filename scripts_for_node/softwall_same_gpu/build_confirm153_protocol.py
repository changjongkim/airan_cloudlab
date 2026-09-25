#!/usr/bin/env python3.11
"""Freeze the C153 integrated shared-recovery mechanism protocol."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrated_shared_recovery_plan_v1 import (
    PHYSICAL_RECOVERY_KEYS,
    REJECTED_KEY,
    REQUEST_ORDER,
    SUCCESS_KEYS,
    IntegratedScenarioConfig,
)


SCRIPT_SOURCES = (
    "integrated_shared_recovery_plan_v1.py",
    "integrated_shared_recovery_owner.py",
    "integrated_shared_recovery_worker.py",
    "shared_recovery_certificate_v1.py",
    "shared_conventional_owner.py",
    "shared_conventional_worker.py",
    "multigpu_p2p_ipc_gate.py",
    "dual_receiver_phy.py",
    "trace_qwen_worker.py",
    "run_confirm153_integrated_shared_recovery.sh",
)
TASK1_SOURCES = ("isca_v2/cuda_ipc_channel.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {
        f"scripts/{name}": scripts_root / name for name in SCRIPT_SOURCES
    }
    paths.update({
        f"task1/{name}": task1_root / name for name in TASK1_SOURCES
    })
    return {key: sha256(path) for key, path in paths.items()}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--receiver-seed0", type=int, required=True)
    parser.add_argument("--receiver-seed1", type=int, required=True)
    parser.add_argument("--channel-seed0", type=int, required=True)
    parser.add_argument("--channel-seed1", type=int, required=True)
    parser.add_argument("--snr-db", type=float, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    args = parser.parse_args()
    config = IntegratedScenarioConfig()
    value = {
        "schema": "softwall-confirm153-integrated-shared-recovery-protocol-v1",
        "status": "frozen-before-run",
        "label": args.label,
        "development_or_confirmatory": "development_canary",
        "mode": {
            "period_ms": config.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "ai_bound_ms": config.ai_bound_ms,
            "guard_ms": config.guard_ms,
            "capacity": config.capacity,
            "qwen_context_length": 64,
            "qwen_model": "Qwen/Qwen2.5-1.5B",
            "warmup": args.warmup,
            "release_lead_ms": 1000.0,
            "release_semantics": "t_IQ_ready_after_synthetic_channel_preparation",
        },
        "placement": {
            "home0_owner": "GPU0",
            "home1_owner": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS",
            "GPU3": "unused in injected-outcome mechanism canary",
        },
        "mps_active_thread_percentage": {
            "shared_cuphy_worker": 80,
            "qwen_worker": 20,
        },
        "request_order": [list(key) for key in REQUEST_ORDER],
        "injected_success_keys": [list(key) for key in SUCCESS_KEYS],
        "physical_recovery_keys": [list(key) for key in PHYSICAL_RECOVERY_KEYS],
        "expected_rejected_key": list(REJECTED_KEY),
        "workload": {
            "receiver_seeds": [args.receiver_seed0, args.receiver_seed1],
            "channel_seeds": [args.channel_seed0, args.channel_seed1],
            "snr_db": args.snr_db,
            "noise_reference": "pre_fading",
        },
        "required_semantics": (
            "local 3+2 feasible; global fifth reject before launch; one success "
            "still rejects AI35; two successes atomically open AI35; Qwen fence "
            "retires lease; coordinator placements dispatch two real cuPHY/P2P "
            "recoveries and home commits before D155"
        ),
        "scope": (
            "C153 development integration canary with injected NRx outcomes; "
            "not a V17 QSU, WCET, production d_MAC, actual-NRx, or throughput claim"
        ),
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
