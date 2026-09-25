#!/usr/bin/env python3.11
"""Freeze one four-arm C161 phase-2 physical fault campaign."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from c159_q2_classes import CLASS_BOUNDS_MS
from c161_phase2_faults import ARMS, PHASE2_CONTEXTS, parse_arm_order
from integrated_shared_recovery_plan_v1 import REQUEST_ORDER, IntegratedScenarioConfig


PHASE2_SOURCES = (
    "c159_q2_classes.py",
    "c160_fault_state_model_v1.py",
    "c161_phase2_faults.py",
    "c161_phase2_plan.py",
    "c161_phase2_qwen_worker.py",
    "c161_phase2_coordinator.py",
    "build_c161_phase2_protocol.py",
    "analyze_c161_phase2_arm.py",
    "analyze_c161_phase2_campaign.py",
    "analyze_c161_phase2_two_node.py",
    "run_c161_phase2.sh",
    "test_c161_phase2_faults.py",
    "test_c161_phase2_plan.py",
    "bounded_launch_revalidation.py",
    "actual_nrx_batch_recovery_plan_v1.py",
    "c159_prestaged_owner.py",
    "c159_persistent_nrx_worker.py",
    "actual_nrx_repeated_control.py",
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
)
TASK1_SOURCES = ("isca_v2/cuda_ipc_channel.py",)
ITERATIONS = {
    "stale_duplicate_nrx": 60,
    "post_fence_reply_delay": 70,
    "pre_fence_channel_loss": 70,
    "stale_duplicate_recovery": 60,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def source_hashes(scripts_root: Path, task1_root: Path) -> dict[str, str]:
    paths = {f"scripts/{name}": scripts_root / name for name in PHASE2_SOURCES}
    paths.update({f"task1/{name}": task1_root / name for name in TASK1_SOURCES})
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"missing C161 phase-2 source: {missing}")
    return {key: sha256(path) for key, path in paths.items()}


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--label", required=True)
    parser.add_argument("--campaign", choices=("development", "holdout"), required=True)
    parser.add_argument("--scripts-root", type=Path, required=True)
    parser.add_argument("--task1-root", type=Path, required=True)
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--state-root", type=Path, required=True)
    parser.add_argument("--prespec", type=Path, required=True)
    parser.add_argument("--seed-base", type=int, required=True)
    parser.add_argument("--excluded-nodes", required=True)
    parser.add_argument("--arm-order", required=True)
    parser.add_argument("--period-ms", type=float, default=180.0)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--event-fault-interval", type=int, default=6)
    parser.add_argument("--event-fault-target", type=int, default=10)
    parser.add_argument("--terminal-fault-after-epoch", type=int, default=5)
    parser.add_argument("--response-delay-ms", type=float, default=100.0)
    parser.add_argument("--response-timeout-margin-ms", type=float, default=8.0)
    args = parser.parse_args()
    try:
        arms = parse_arm_order(args.arm_order)
    except ValueError as error:
        parser.error(str(error))
    if (args.period_ms <= 0 or args.seed_base <= 0
            or args.event_fault_interval <= 0 or args.event_fault_target != 10
            or args.terminal_fault_after_epoch <= 0
            or args.response_delay_ms <= 0
            or args.response_timeout_margin_ms <= 0):
        parser.error("invalid C161 phase-2 configuration")
    config = IntegratedScenarioConfig(period_ms=round(args.period_ms))
    config.validate()
    prespec = json.loads(args.prespec.read_text(encoding="utf-8"))
    if (prespec.get("status") != "DESIGN_PRESPEC_HOLDOUT_NOT_MATERIALIZED"
            or prespec.get("dataset", {}).get("holdout_materialized") is not False):
        parser.error("C159 prespec status mismatch")
    accepted = REQUEST_ORDER[:4]
    snr = {accepted[0]: 20.0, accepted[1]: -8.75,
           accepted[2]: -8.75, accepted[3]: 20.0}
    arm_specs = []
    for arm_index, arm in enumerate(arms):
        arm_label = f"{args.label}_a{arm_index}_{arm}"
        state = args.state_root / arm_label
        peers = []
        for index, key in enumerate(accepted):
            home_index = int(key[0].removeprefix("home"))
            safe = f"{key[0]}_{key[1]}"
            peers.append({
                "key": list(key),
                "source_device": home_index,
                "receiver_seed": args.seed_base + arm_index * 1_000_000 + index * 100_000 + 11,
                "channel_seed_base": args.seed_base + arm_index * 1_000_000 + index * 100_000 + 10_001,
                "snr_db": snr[key],
                "nrx_tag": f"{arm_label}_{safe}_nrx",
                "recovery_tag": f"{arm_label}_{safe}_recovery",
                "ready_file": str(state / f"{safe}_ready.json"),
                "decision_control": str(state / f"{safe}_decision.ctrl"),
                "owner_output": str(args.raw_dir / f"{arm_label}_{safe}_owner.json"),
            })
        peer_spec = args.result_root / f"{arm_label}_peer_spec.json"
        peer_tsv = state / "peers.tsv"
        atomic_json(peer_spec, {
            "schema": "softwall-actual-nrx-peer-spec-v1",
            "label": arm_label,
            "peers": peers,
        })
        peer_tsv.parent.mkdir(parents=True, exist_ok=True)
        peer_tsv.write_text("".join(
            "\t".join(map(str, (
                row["key"][0].removeprefix("home"), row["key"][1],
                row["source_device"], row["receiver_seed"],
                row["channel_seed_base"], row["snr_db"], row["nrx_tag"],
                row["recovery_tag"], row["ready_file"],
                row["decision_control"], row["owner_output"],
            ))) + "\n" for row in peers
        ), encoding="utf-8")
        arm_specs.append({
            "index": arm_index,
            "arm": arm,
            "label": arm_label,
            "iterations": ITERATIONS[arm],
            "seed_base": args.seed_base + arm_index * 1_000_000,
            "state_dir": str(state),
            "peer_spec": str(peer_spec),
            "peer_tsv": str(peer_tsv),
            "schedule_file": str(state / "schedule.json"),
            "completion_dir": str(state / "qwen_markers"),
            "qwen_socket_name": f"c2{arm_index}.sock",
            "qwen_output": str(args.raw_dir / f"{arm_label}_qwen.json"),
            "nrx_output": str(args.raw_dir / f"{arm_label}_nrx_worker.json"),
            "coordinator_output": str(args.raw_dir / f"{arm_label}_coordinator.json"),
            "inventory": str(args.raw_dir / f"{arm_label}_gpu_inventory.csv"),
            "result": str(args.result_root / f"{arm_label}_result.json"),
        })
    protocol = {
        "schema": "softwall-c161-phase2-campaign-protocol-v1",
        "status": "frozen-before-run",
        "campaign": args.campaign,
        "label": args.label,
        "excluded_nodes": sorted({value for value in args.excluded_nodes.split(",") if value}),
        "seed_base": args.seed_base,
        "arm_order": arms,
        "arms": arm_specs,
        "mode": {
            "period_ms": args.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "ai_class_bounds_ms": CLASS_BOUNDS_MS,
            "qwen_context_lengths": PHASE2_CONTEXTS,
            "guard_ms": config.guard_ms,
            "launch_control_bound_ms": 5,
            "event_fault_interval": args.event_fault_interval,
            "event_fault_target": args.event_fault_target,
            "terminal_fault_after_epoch": args.terminal_fault_after_epoch,
            "post_fault_min_radio_epochs": 50,
            "response_delay_ms": args.response_delay_ms,
            "response_timeout_margin_ms": args.response_timeout_margin_ms,
            "post_expiry_copy_budget_ms": args.period_ms - config.expiry_ms,
            "warmup": args.warmup,
            "release_lead_ms": args.release_lead_ms,
            "lifecycle": "one fresh persistent actual-NRx/shared-cuPHY/Qwen process set per fault arm",
        },
        "placement": {
            "home0": "GPU0", "home1": "GPU1",
            "shared_cuphy_and_qwen": "GPU2 under MPS 80/20",
            "persistent_actual_nrx": "GPU3 under MPS 80",
        },
        "accepted_physical_keys": [list(key) for key in accepted],
        "expected_global_reject": list(REQUEST_ORDER[4]),
        "fault_contract": {
            "stale_duplicate_nrx": "10 physical current outcome batches followed by duplicate and prior-epoch delivery; both injected deliveries are no-op",
            "post_fence_reply_delay": "one Qwen CUDA completion marker precedes a response delayed 100 ms; retire matching lease, quarantine AI, continue at least 50 radio epochs",
            "pre_fence_channel_loss": "one Qwen CUDA launch marker precedes channel close and no visible completion marker; retain lease token, quarantine AI, continue at least 50 radio epochs",
            "stale_duplicate_recovery": "10 physical conventional responses followed by duplicate and prior-epoch delivery; no second commit or physical execution",
        },
        "gates": {
            "common": "all actual NRx/transport/recovery/single-commit/deadline/bound/source/lifecycle gates pass",
            "event_faults": "A2 and A6 each inject exactly 10 duplicate+stale events with state mutation zero",
            "post_fence": "matching marker retires exactly one lease, future AI launch zero, at least 50 later radio epochs",
            "pre_fence": "completion marker absent at decision, retire rejected with physical_fence_required, ambiguous lease retained, future AI launch zero, at least 50 later radio epochs",
        },
        "scope": "Finite-sample warm P180 phase-2 containment for event replay and Qwen channel faults. GPU/driver hang, process restart/durable reconciliation, cold lifecycle, production d_MAC and cross-family behavior remain UQ.",
        "prespec": {
            "path": str(args.prespec), "sha256": sha256(args.prespec),
            "status": prespec["status"], "holdout_materialized": False,
        },
        "source_sha256": source_hashes(args.scripts_root, args.task1_root),
    }
    atomic_json(args.output, protocol)
    print(json.dumps({
        "protocol": str(args.output), "campaign": args.campaign,
        "arm_order": arms, "arm_labels": [row["label"] for row in arm_specs],
        "source_count": len(protocol["source_sha256"]),
    }, indent=2))


if __name__ == "__main__":
    main()
