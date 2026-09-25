#!/usr/bin/env python3
"""Actual-NeuralRx-driven V17.1 coordinator and shared recovery worker."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from actual_nrx_shared_recovery_plan_v1 import build_actual_outcome_plan
from dual_receiver_phy import PairedDualReceiver
from integrated_shared_recovery_plan_v1 import IntegratedScenarioConfig
from integrated_shared_recovery_worker import connect_qwen, qwen_request, wait_until_ns
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


def load_spec(path: Path) -> list[dict]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("schema") != "softwall-actual-nrx-peer-spec-v1":
        raise ValueError("invalid actual-NRx peer specification")
    return value["peers"]


def wait_json(path: Path, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.002)
    raise TimeoutError(f"timed out waiting for {path}")


def read_json_if_complete(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--dispatch-file", type=Path, required=True)
    parser.add_argument("--qwen-socket", required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--destination-device", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--ready-timeout-s", type=float, default=240.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--release-lead-ms", type=float, default=3000.0)
    args = parser.parse_args()
    specs = load_spec(args.peer_spec)
    config = IntegratedScenarioConfig()

    peers = []
    for row in specs:
        source = int(row["source_device"])
        cp.cuda.runtime.setDevice(source)
        peers.append(CudaIpcPeer(
            row["recovery_tag"], args.ready_timeout_s, directory=args.ipc_dir
        ))
    access = {}
    for source in sorted({int(row["source_device"]) for row in specs}):
        access.update(enable_peer_access(source, args.destination_device))

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
                    raise RuntimeError(f"recovery warmup failed for {row['key']}")
            contexts.append({
                "receiver": receiver,
                "remote_forward": remote_forward,
                "remote_backward": remote_backward,
                "forward": P2PCopier(
                    int(row["source_device"]), args.destination_device
                ),
                "backward": P2PCopier(
                    args.destination_device, int(row["source_device"])
                ),
            })

    qwen_socket = None
    qwen_channel = None
    handles_closed = False
    result = {
        "schema": "softwall-actual-nrx-integrated-coordinator-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "peer_specs": specs,
        "peer_access": access,
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
        for row in specs:
            ready = wait_json(Path(row["ready_file"]), args.ready_timeout_s)
            if ready.get("key") != row["key"] or not ready.get("prepared"):
                raise RuntimeError(f"invalid owner readiness {row['key']}")

        release_wall_ns = time.perf_counter_ns() + round(args.release_lead_ms * 1e6)
        atomic_json(args.release_file, {
            "schema": "softwall-actual-nrx-release-v1",
            "release_wall_ns": release_wall_ns,
            "release_semantics": "t_IQ_ready_after_synthetic_channel_preparation",
            "accepted_keys": [row["key"] for row in specs],
        })
        cutoff_wall_ns = release_wall_ns + config.recovery_release_ns
        wait_until_ns(cutoff_wall_ns)
        observed = []
        missing_at_cutoff = []
        timely_successes = []
        for row in specs:
            value = read_json_if_complete(Path(row["outcome_file"]))
            if value is None:
                missing_at_cutoff.append(row["key"])
                continue
            observed.append(value)
            if value.get("timely_success") is True:
                timely_successes.append(tuple(row["key"]))

        revalidation_started_ns = time.perf_counter_ns()
        launch_now_ns = max(
            config.recovery_release_ns,
            revalidation_started_ns - release_wall_ns,
        )
        plan = build_actual_outcome_plan(timely_successes, launch_now_ns, config)
        revalidation_completed_ns = time.perf_counter_ns()
        coordinator = plan["coordinator"]
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
            "release_wall_ns": release_wall_ns,
            "cutoff_wall_ns": cutoff_wall_ns,
            "observed_outcomes_at_cutoff": observed,
            "missing_outcomes_at_cutoff": missing_at_cutoff,
            "local_certificates": plan["local_certificates"],
            "reservation_decisions": plan["reservation_decisions"],
            "actual_success_outcomes": plan["actual_success_outcomes"],
            "lease_decision": plan["lease_decision"],
            "lease_interval": plan["lease_interval"],
            "launch_control_bound_ms": plan["launch_control_bound_ms"],
            "launch_revalidation": {
                "started_ns": revalidation_started_ns,
                "completed_ns": revalidation_completed_ns,
                "now_ns": launch_now_ns,
                "host_ms": (
                    revalidation_completed_ns - revalidation_started_ns
                ) / 1e6,
            },
        })
        dispatch = {
            "schema": "softwall-actual-nrx-dispatch-v1",
            "success_keys": [list(key) for key in plan["success_keys"]],
            "physical_recovery_keys": [
                list(key) for key in plan["physical_recovery_keys"]
            ],
            "rejected_keys": [list(key) for key in plan["rejected_keys"]],
            "generation": coordinator.generation,
        }
        atomic_json(args.dispatch_file, dispatch)

        if plan["ai_expected"]:
            qwen = qwen_request(qwen_channel, plan["lease_id"], 64)
            qwen["release_to_send_ms"] = (qwen["sent_ns"] - release_wall_ns) / 1e6
            qwen["release_to_return_ms"] = (
                qwen["returned_ns"] - release_wall_ns
            ) / 1e6
            qwen["bound_violation"] = qwen["execution_ms"] > config.ai_bound_ms
            qwen["dispatch_after_revalidation_ms"] = (
                qwen["sent_ns"] - revalidation_started_ns
            ) / 1e6
            qwen["launch_control_bound_violation"] = (
                qwen["dispatch_after_revalidation_ms"]
                > plan["launch_control_bound_ms"]
            )
            lease = plan["lease_interval"]
            qwen["lease_interval_violation"] = not (
                lease["start_ns"] <= qwen["sent_ns"] - release_wall_ns
                and qwen["returned_ns"] - release_wall_ns <= lease["finish_ns"]
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
                raise RuntimeError(f"actual-NRx lease retire failed: {retire.reason}")
        else:
            result["qwen"] = None
            result["lease_retire"] = None

        final_snapshot = coordinator.snapshot()
        result["final_snapshot"] = final_snapshot
        by_key = {
            tuple(row["key"]): (row, peer, context)
            for row, peer, context in zip(specs, peers, contexts)
        }
        ordered = sorted(final_snapshot["placements"], key=lambda row: (
            row["start_ns"], row["lane"], row["home_id"], row["request_id"]
        ))
        for placement in ordered:
            key = (placement["home_id"], placement["request_id"])
            row, peer, context = by_key[key]
            wait_sequence(peer, 1, args.unit_timeout_s)
            wait_until_ns(release_wall_ns + placement["start_ns"])
            started_ns = time.perf_counter_ns()
            forward = context["forward"].copy(
                context["remote_forward"], peer.forward.view(cp.float32)
            )
            install_window(context["receiver"], context["remote_forward"])
            decoded = run_conventional(
                context["receiver"], context["remote_backward"]
            )
            backward = context["backward"].copy(
                peer.backward.view(cp.uint8), context["remote_backward"]
            )
            completed_ns = time.perf_counter_ns()
            peer.publish_backward(1)
            result["physical_recoveries"].append({
                "key": list(key),
                "model_lane": placement["lane"],
                "model_start_ns": placement["start_ns"],
                "model_finish_ns": placement["finish_ns"],
                "actual_start_ns": started_ns,
                "actual_completed_ns": completed_ns,
                "release_to_start_ms": (started_ns - release_wall_ns) / 1e6,
                "release_to_complete_ms": (completed_ns - release_wall_ns) / 1e6,
                "forward_gpu_us": forward["gpu_us"],
                "conventional_gpu_ms": decoded["gpu_ms"],
                "conventional_host_ms": decoded["host_ms"],
                "conventional_reference_correct": decoded["correct"],
                "crc_failures": decoded["crc_failures"],
                "payload_mismatches": decoded["payload_mismatches"],
                "backward_gpu_us": backward["gpu_us"],
                "declared_path_bound_violation": (
                    (completed_ns - started_ns) / 1e6
                    > config.conventional_bound_ms
                ),
            })

        for peer in peers:
            wait_sequence(peer, TERMINATE_SEQ, args.ready_timeout_s)
        for row, peer in zip(specs, peers):
            cp.cuda.runtime.setDevice(int(row["source_device"]))
            close_peer_before_ack(peer)
        handles_closed = True
        result["ipc_handles_closed_before_ack"] = True
        qwen_channel.write(b'{"op":"stop"}\n')
        stopped = json.loads(qwen_channel.readline())
        result["qwen_stop_acknowledged"] = bool(stopped.get("stopped"))
        result["certificate_order"] = [
            [row["home_id"], row["request_id"]] for row in ordered
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
        if not handles_closed:
            for peer in peers:
                try:
                    peer.close()
                except BaseException:
                    pass


if __name__ == "__main__":
    main()
