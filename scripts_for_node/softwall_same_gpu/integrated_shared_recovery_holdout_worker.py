#!/usr/bin/env python3
"""Four-branch certificate-driven physical worker for C154/C155."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver
from integrated_shared_recovery_holdout_plan_v1 import BRANCHES, build_branch
from integrated_shared_recovery_worker import (
    connect_qwen,
    qwen_request,
    wait_until_ns,
)
from isca_v2.cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from multigpu_p2p_ipc_gate import (
    P2PCopier,
    close_peer_before_ack,
    enable_peer_access,
)
from shared_conventional_worker import (
    RX_ELEMENTS,
    atomic_json,
    install_window,
    run_conventional,
    wait_sequence,
)


def load_peer_specs(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "softwall-c154-peer-spec-v1":
        raise ValueError("invalid peer specification schema")
    specs = value.get("peers", [])
    keys = [tuple(row["key"]) for row in specs]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate peer key")
    return specs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--branch", choices=BRANCHES, required=True)
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--qwen-socket", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination-device", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ready-timeout-s", type=float, default=240.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    args = parser.parse_args()

    preview = build_branch(args.branch)
    expected_keys = preview["physical_recovery_keys"]
    specs = load_peer_specs(args.peer_spec)
    if tuple(tuple(row["key"]) for row in specs) != expected_keys:
        raise RuntimeError(
            f"peer spec does not match certificate preview: "
            f"{[row['key'] for row in specs]} != {expected_keys}"
        )
    peers = []
    for row in specs:
        source = int(row["source_device"])
        cp.cuda.runtime.setDevice(source)
        peers.append(
            CudaIpcPeer(row["tag"], args.ready_timeout_s, directory=args.ipc_dir)
        )

    peer_access = {}
    for source in sorted({int(row["source_device"]) for row in specs}):
        peer_access.update(enable_peer_access(source, args.destination_device))

    contexts = []
    with cp.cuda.Device(args.destination_device):
        for row in specs:
            receiver = PairedDualReceiver(
                args.engine,
                seed=int(row["receiver_seed"]),
                device=args.destination_device,
                enable_local_neural=False,
            )
            remote_forward = cp.empty(2 * RX_ELEMENTS, dtype=cp.float32)
            remote_backward = cp.empty(
                len(receiver.reference_tb) + 1, dtype=cp.uint8
            )
            for _ in range(args.warmup):
                if not receiver.run_conventional()[1]:
                    raise RuntimeError(f"warmup failed for {row['key']}")
            contexts.append({
                "receiver": receiver,
                "remote_forward": remote_forward,
                "remote_backward": remote_backward,
                "forward_copy": P2PCopier(
                    int(row["source_device"]), args.destination_device
                ),
                "backward_copy": P2PCopier(
                    args.destination_device, int(row["source_device"])
                ),
            })

    qwen_socket = None
    qwen_channel = None
    result = {
        "schema": "softwall-c154-integrated-holdout-worker-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "branch": args.branch,
        "peer_specs": specs,
        "peer_access": peer_access,
        "release_lead_ms": args.release_lead_ms,
        "physical_recoveries": [],
        "ipc_handles_closed_before_ack": False,
        "qwen_stop_acknowledged": False,
        "error": None,
    }
    try:
        qwen_socket, qwen_channel = connect_qwen(
            args.qwen_socket, args.ready_timeout_s
        )
        for peer in peers:
            peer.mark_ready()
        release_wall_ns = time.perf_counter_ns() + round(args.release_lead_ms * 1e6)
        atomic_json(args.release_file, {
            "schema": "softwall-c154-release-v1",
            "branch": args.branch,
            "release_wall_ns": release_wall_ns,
            "release_semantics": "t_IQ_ready_after_synthetic_channel_preparation",
            "physical_recovery_keys": [list(key) for key in expected_keys],
        })
        wait_until_ns(release_wall_ns + preview["config"].recovery_release_ns)
        plan = build_branch(args.branch)
        if plan["physical_recovery_keys"] != expected_keys:
            raise RuntimeError("certificate plan changed between preview and release")
        coordinator = plan["coordinator"]
        config = plan["config"]
        result.update({
            "config": {
                "period_ms": config.period_ms,
                "expiry_ms": config.expiry_ms,
                "nrx_bound_ms": config.nrx_bound_ms,
                "conventional_bound_ms": config.conventional_bound_ms,
                "ai_bound_ms": config.ai_bound_ms,
                "guard_ms": config.guard_ms,
                "capacity": config.capacity,
            },
            "local_certificates": plan["local_certificates"],
            "reservation_decisions": plan["reservation_decisions"],
            "success_outcomes": plan["success_outcomes"],
            "lease_decision": plan["lease_decision"],
            "ai_expected": plan["ai_expected"],
            "release_wall_ns": release_wall_ns,
        })

        if plan["ai_expected"]:
            qwen = qwen_request(
                qwen_channel, f"{plan['lease_id']}-{args.branch}", 64
            )
            qwen["release_to_send_ms"] = (
                qwen["sent_ns"] - release_wall_ns
            ) / 1e6
            qwen["release_to_return_ms"] = (
                qwen["returned_ns"] - release_wall_ns
            ) / 1e6
            qwen["bound_violation"] = qwen["execution_ms"] > config.ai_bound_ms
            qwen["lease_interval_violation"] = not (
                config.ai_start_ns
                <= qwen["sent_ns"] - release_wall_ns
                and qwen["returned_ns"] - release_wall_ns
                <= config.ai_finish_ns
            )
            result["qwen"] = qwen
            retire_now_ns = max(0, qwen["returned_ns"] - release_wall_ns)
            retire = coordinator.retire_lease(
                plan["lease_id"],
                gpu_fence_confirmed=qwen["fence_confirmed"],
                expected_generation=coordinator.generation,
                now_ns=retire_now_ns,
            )
            result["lease_retire"] = {
                "accepted": retire.accepted,
                "reason": retire.reason,
                "generation": retire.generation,
                "now_ns": retire_now_ns,
            }
            if not retire.accepted:
                raise RuntimeError(f"lease retire failed: {retire.reason}")
        else:
            result["qwen"] = None
            result["lease_retire"] = None

        final_snapshot = coordinator.snapshot()
        result["final_snapshot"] = final_snapshot
        context_by_key = {
            tuple(row["key"]): (peer, context)
            for row, peer, context in zip(specs, peers, contexts)
        }
        for placement in sorted(
            final_snapshot["placements"],
            key=lambda row: (
                row["start_ns"], row["lane"], row["home_id"], row["request_id"]
            ),
        ):
            key = (placement["home_id"], placement["request_id"])
            peer, context = context_by_key[key]
            wait_sequence(peer, 1, args.unit_timeout_s)
            wait_until_ns(release_wall_ns + placement["start_ns"])
            actual_start_ns = time.perf_counter_ns()
            forward = context["forward_copy"].copy(
                context["remote_forward"], peer.forward.view(cp.float32)
            )
            install_window(context["receiver"], context["remote_forward"])
            decoded = run_conventional(
                context["receiver"], context["remote_backward"]
            )
            if not decoded["correct"]:
                raise RuntimeError(f"decode failed for {key}")
            backward = context["backward_copy"].copy(
                peer.backward.view(cp.uint8), context["remote_backward"]
            )
            completed_ns = time.perf_counter_ns()
            peer.publish_backward(1)
            result["physical_recoveries"].append({
                "key": list(key),
                "model_lane": placement["lane"],
                "model_start_ns": placement["start_ns"],
                "model_finish_ns": placement["finish_ns"],
                "actual_start_ns": actual_start_ns,
                "actual_completed_ns": completed_ns,
                "release_to_start_ms": (actual_start_ns - release_wall_ns) / 1e6,
                "release_to_complete_ms": (completed_ns - release_wall_ns) / 1e6,
                "forward_gpu_us": forward["gpu_us"],
                "conventional_gpu_ms": decoded["gpu_ms"],
                "conventional_host_ms": decoded["host_ms"],
                "backward_gpu_us": backward["gpu_us"],
                "correct": decoded["correct"],
                "declared_path_bound_violation": (
                    (completed_ns - actual_start_ns) / 1e6
                    > config.conventional_bound_ms
                ),
            })

        for peer in peers:
            wait_sequence(peer, TERMINATE_SEQ, args.ready_timeout_s)
        for row, peer in zip(specs, peers):
            cp.cuda.runtime.setDevice(int(row["source_device"]))
            close_peer_before_ack(peer)
        result["ipc_handles_closed_before_ack"] = True
        qwen_channel.write(b'{"op":"stop"}\n')
        stop = json.loads(qwen_channel.readline())
        result["qwen_stop_acknowledged"] = bool(stop.get("stopped"))
        result["certificate_order"] = [
            [row["home_id"], row["request_id"]]
            for row in sorted(
                final_snapshot["placements"],
                key=lambda row: (
                    row["start_ns"], row["lane"], row["home_id"], row["request_id"]
                ),
            )
        ]
    except BaseException as error:
        result["error"] = repr(error)
        raise
    finally:
        result["completed_ns"] = time.perf_counter_ns()
        atomic_json(args.output, result)
        if qwen_channel is not None:
            try:
                qwen_channel.close()
            except OSError:
                pass
        if qwen_socket is not None:
            qwen_socket.close()
        if not result["ipc_handles_closed_before_ack"]:
            for peer in peers:
                try:
                    peer.close()
                except BaseException:
                    pass


if __name__ == "__main__":
    main()

