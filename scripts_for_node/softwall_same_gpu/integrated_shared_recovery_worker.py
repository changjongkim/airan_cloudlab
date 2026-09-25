#!/usr/bin/env python3
"""Certificate-driven cuPHY/Qwen worker for the C153 integration canary."""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import time
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver
from integrated_shared_recovery_plan_v1 import (
    LEASE_ID,
    PHYSICAL_RECOVERY_KEYS,
    IntegratedScenarioConfig,
    local_certificates,
    placement_keys,
    reserve_global,
    resolve_successes_and_lease,
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


def wait_until_ns(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1e9)
        else:
            time.sleep(0)


def connect_qwen(path: str, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    last_error = None
    while time.monotonic() < deadline:
        client = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            client.connect(path)
            client.settimeout(timeout_s)
            return client, client.makefile("rwb", buffering=0)
        except OSError as error:
            last_error = error
            client.close()
            time.sleep(0.01)
    raise TimeoutError(f"could not connect to Qwen worker: {last_error}")


def qwen_request(channel, request_id: str, context_length: int) -> dict:
    sent_ns = time.perf_counter_ns()
    channel.write(json.dumps({
        "op": "run",
        "request_id": request_id,
        "context_length": context_length,
    }).encode() + b"\n")
    raw = channel.readline()
    returned_ns = time.perf_counter_ns()
    if not raw:
        raise ConnectionError("Qwen worker closed before response")
    response = json.loads(raw)
    if not response.get("ok"):
        raise RuntimeError(f"Qwen worker rejected request: {response}")
    return {
        "request_id": request_id,
        "context_length": context_length,
        "sent_ns": sent_ns,
        "returned_ns": returned_ns,
        "execution_ms": (returned_ns - sent_ns) / 1e6,
        "gpu_ms": float(response["gpu_ms"]),
        "worker_completed_ns": int(response["completed_ns"]),
        "fence_confirmed": True,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag0", required=True)
    parser.add_argument("--tag1", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--qwen-socket", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-device0", type=int, default=0)
    parser.add_argument("--source-device1", type=int, default=1)
    parser.add_argument("--destination-device", type=int, default=2)
    parser.add_argument("--receiver-seed0", type=int, required=True)
    parser.add_argument("--receiver-seed1", type=int, required=True)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ready-timeout-s", type=float, default=240.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--release-lead-ms", type=float, default=200.0)
    args = parser.parse_args()
    source_devices = (args.source_device0, args.source_device1)
    if len(set(source_devices + (args.destination_device,))) != 3:
        parser.error("three distinct GPU devices are required")

    config = IntegratedScenarioConfig()
    coordinator, reservation_decisions = reserve_global(config)
    pre_outcome_snapshot = coordinator.snapshot()
    local = local_certificates(config)
    peers = []
    for tag, source in zip((args.tag0, args.tag1), source_devices):
        cp.cuda.runtime.setDevice(source)
        peers.append(CudaIpcPeer(tag, args.ready_timeout_s, directory=args.ipc_dir))

    peer_access = {}
    for source in source_devices:
        peer_access.update(enable_peer_access(source, args.destination_device))

    contexts = []
    with cp.cuda.Device(args.destination_device):
        for home, seed in enumerate((args.receiver_seed0, args.receiver_seed1)):
            receiver = PairedDualReceiver(
                args.engine,
                seed=seed,
                device=args.destination_device,
                enable_local_neural=False,
            )
            remote_forward = cp.empty(2 * RX_ELEMENTS, dtype=cp.float32)
            remote_backward = cp.empty(
                len(receiver.reference_tb) + 1, dtype=cp.uint8
            )
            for _ in range(args.warmup):
                if not receiver.run_conventional()[1]:
                    raise RuntimeError(f"home {home} warmup failed")
            contexts.append({
                "receiver": receiver,
                "remote_forward": remote_forward,
                "remote_backward": remote_backward,
                "forward_copy": P2PCopier(source_devices[home], args.destination_device),
                "backward_copy": P2PCopier(args.destination_device, source_devices[home]),
            })

    qwen_socket = None
    qwen_channel = None
    result = {
        "schema": "softwall-c153-integrated-worker-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "config": {
            "period_ms": config.period_ms,
            "expiry_ms": config.expiry_ms,
            "nrx_bound_ms": config.nrx_bound_ms,
            "conventional_bound_ms": config.conventional_bound_ms,
            "ai_bound_ms": config.ai_bound_ms,
            "guard_ms": config.guard_ms,
            "capacity": config.capacity,
            "release_lead_ms": args.release_lead_ms,
        },
        "local_certificates": local,
        "reservation_decisions": reservation_decisions,
        "pre_outcome_snapshot": pre_outcome_snapshot,
        "peer_access": peer_access,
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
            "schema": "softwall-c153-release-v1",
            "release_wall_ns": release_wall_ns,
            "release_semantics": "t_IQ_ready_after_synthetic_channel_preparation",
            "physical_recovery_keys": [list(key) for key in PHYSICAL_RECOVERY_KEYS],
            "accepted_keys": [
                [row["home_id"], row["request_id"]]
                for row in reservation_decisions if row["accepted"]
            ],
            "rejected_keys": [
                [row["home_id"], row["request_id"]]
                for row in reservation_decisions if not row["accepted"]
            ],
        })
        wait_until_ns(release_wall_ns + config.recovery_release_ns)
        transition = resolve_successes_and_lease(coordinator, config)
        result["outcome_transition"] = transition
        result["lease_snapshot"] = coordinator.snapshot()
        if not transition["lease_after_two_successes"]["accepted"]:
            raise RuntimeError("conditional Qwen lease was not admitted")

        qwen = qwen_request(qwen_channel, LEASE_ID, 64)
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
            LEASE_ID,
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
        final_snapshot = coordinator.snapshot()
        result["post_retire_snapshot"] = final_snapshot

        context_by_key = {
            PHYSICAL_RECOVERY_KEYS[0]: (0, peers[0], contexts[0]),
            PHYSICAL_RECOVERY_KEYS[1]: (1, peers[1], contexts[1]),
        }
        for placement in sorted(
            final_snapshot["placements"],
            key=lambda row: (
                row["start_ns"], row["lane"], row["home_id"], row["request_id"]
            ),
        ):
            key = (placement["home_id"], placement["request_id"])
            home, peer, context = context_by_key[key]
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
                "home_id": key[0],
                "request_id": key[1],
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
        for source, peer in zip(source_devices, peers):
            cp.cuda.runtime.setDevice(source)
            close_peer_before_ack(peer)
        result["ipc_handles_closed_before_ack"] = True
        qwen_channel.write(b'{"op":"stop"}\n')
        stop = json.loads(qwen_channel.readline())
        result["qwen_stop_acknowledged"] = bool(stop.get("stopped"))
        result["certificate_order"] = [list(key) for key in placement_keys(final_snapshot)]
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
