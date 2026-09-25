#!/usr/bin/env python3.11
"""Freeze the C156 conditional-open Nsight semantics arm."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from integrated_shared_recovery_launch_plan_v1 import LAUNCH_CONTROL_BOUND_MS
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


BRANCH = "conditional_open"
SCRIPT_SOURCES = (
    "integrated_shared_recovery_plan_v1.py",
    "integrated_shared_recovery_holdout_plan_v1.py",
    "integrated_shared_recovery_launch_plan_v1.py",
    "integrated_shared_recovery_owner.py",
    "integrated_shared_recovery_worker.py",
    "integrated_shared_recovery_timeline_worker.py",
    "shared_recovery_certificate_v1.py",
    "shared_conventional_owner.py",
    "shared_conventional_worker.py",
    "multigpu_p2p_ipc_gate.py",
    "dual_receiver_phy.py",
    "trace_qwen_worker.py",
    "build_confirm154_peer_spec.py",
    "build_confirm156_timeline_protocol.py",
    "analyze_confirm156_timeline.py",
    "run_confirm156_timeline.sh",
)
TASK1_SOURCES = ("isca_v2/cuda_ipc_channel.py",)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in SCRIPT_SOURCES}
    paths.update({f"task1/{name}": task1_root / name for name in TASK1_SOURCES})
    return {key: sha256(path) for key, path in paths.items()}


def arm_seeds(seed_base: int) -> dict[str, dict[str, int]]:
    receivers = {}
    channels = {}
    for request_index, key in enumerate(REQUEST_ORDER):
        name = "/".join(key)
        receivers[name] = seed_base + request_index * 100 + 11
        channels[name] = seed_base + request_index * 100 + 51
    return {"receiver": receivers, "channel": channels}


def build_protocol(
    *, label: str, scripts_root: Path, task1_root: Path, seed_base: int,
    excluded_nodes: str, snr_db: float, warmup: int, release_lead_ms: float,
) -> dict:
    config = IntegratedScenarioConfig()
    return {
        "schema": "softwall-confirm156-timeline-protocol-v1",
        "status": "frozen-before-run",
        "label": label,
        "branch_order": [BRANCH],
        "excluded_nodes": sorted({
            value for value in excluded_nodes.split(",") if value
        }),
        "mode": {
            "period_ms": config.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "ai_bound_ms": config.ai_bound_ms,
            "launch_control_bound_ms": LAUNCH_CONTROL_BOUND_MS,
            "guard_ms": config.guard_ms,
            "capacity": config.capacity,
            "qwen_context_length": 64,
            "qwen_model": "Qwen/Qwen2.5-1.5B",
            "snr_db": snr_db,
            "warmup": warmup,
            "release_lead_ms": release_lead_ms,
            "release_semantics": "t_IQ_ready_after_synthetic_channel_preparation",
        },
        "placement": {
            "home0_owners": "GPU0",
            "home1_owners": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS 80/20",
            "GPU3": "unused in controlled-outcome timeline arm",
        },
        "arms": [{
            "arm_index": 0,
            "branch": BRANCH,
            "seeds": arm_seeds(seed_base),
        }],
        "outcome_source": "prospectively controlled conditional-open injection",
        "profiler": {
            "tool": "NVIDIA Nsight Systems",
            "trace": "cuda,nvtx",
            "cuda_graph_trace": "node",
            "export": "sqlite",
            "semantic_gates": [
                "qwen_kernels_within_committed_lease",
                "qwen_recovery_kernel_overlap_zero",
                "certificate_recovery_order",
                "forward_install_cuphy_backward_gpu_causality",
            ],
        },
        "scope": (
            "Profiler-perturbed one-node controlled conditional-open arm. Passing "
            "validates GPU timeline semantics only; it is not a service bound, "
            "WCET, actual NeuralRx, throughput or production d_MAC result."
        ),
        "source_sha256": source_hashes(scripts_root, task1_root),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--excluded-nodes", required=True)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    args = parser.parse_args()
    value = build_protocol(
        label=args.label,
        scripts_root=args.scripts_root,
        task1_root=args.task1_root,
        seed_base=args.seed_base,
        excluded_nodes=args.excluded_nodes,
        snr_db=args.snr_db,
        warmup=args.warmup,
        release_lead_ms=args.release_lead_ms,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
