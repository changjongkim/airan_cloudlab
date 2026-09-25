#!/usr/bin/env python3
"""Persistent four-peer TensorRT NeuralRx worker for the C158 pilot."""

from __future__ import annotations

import argparse
import os
import platform
import time
from pathlib import Path

import cupy as cp

from actual_nrx_multi_peer_worker import load_spec, wait_json, wait_until_ns
from isca_v2.cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from multigpu_p2p_ipc_gate import (
    P2PCopier,
    close_peer_before_ack,
    enable_peer_access,
    summarize,
)
from multigpu_p2p_nrx_worker import (
    CE_ELEMENTS,
    CE_SHAPE,
    FWD_ELEMENTS,
    LLR_ELEMENTS,
    LLR_SHAPE,
    RX_ELEMENTS,
    RX_SHAPE,
    atomic_json,
    section,
)
from nrx_trt_direct import DirectNrx
from shared_conventional_worker import wait_sequence


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--peer-spec", type=Path, required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--schedule-file", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--destination-device", type=int, default=3)
    parser.add_argument("--ready-timeout-s", type=float, default=300.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--graph-warmup", type=int, default=20)
    parser.add_argument("--release-prewarm-lead-ms", type=float, default=100.0)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("iterations must be positive")
    specs = load_spec(args.peer_spec)

    peers = []
    for row in specs:
        source = int(row["source_device"])
        cp.cuda.runtime.setDevice(source)
        peers.append(CudaIpcPeer(
            row["nrx_tag"], args.ready_timeout_s, directory=args.ipc_dir
        ))
    access = {}
    for source in sorted({int(row["source_device"]) for row in specs}):
        access.update(enable_peer_access(source, args.destination_device))

    with cp.cuda.Device(args.destination_device):
        remote_forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
        remote_backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
        verify_backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
        runtime = DirectNrx(args.engine)
        offset = 0
        for name, count, shape in (
            ("rx_slot_real", RX_ELEMENTS, RX_SHAPE),
            ("rx_slot_imag", RX_ELEMENTS, RX_SHAPE),
            ("h_hat_real", CE_ELEMENTS, CE_SHAPE),
            ("h_hat_imag", CE_ELEMENTS, CE_SHAPE),
        ):
            runtime.bind_tensor(name, section(remote_forward, offset, count, shape))
            offset += count
        runtime.bind_tensor("output_1", remote_backward.reshape((1,) + LLR_SHAPE))
        runtime.inputs["active_dmrs_ports"].fill(1)
        runtime.inputs["dmrs_ofdm_pos"][:] = cp.asarray([[0, 5, 9]], dtype=cp.int32)
        runtime.inputs["dmrs_subcarrier_pos"][:] = cp.asarray(
            [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
        )
        runtime.capture_graph()
        for _ in range(args.graph_warmup):
            runtime.launch(use_graph=True)
        runtime.stream.synchronize()

    contexts = []
    for row, peer in zip(specs, peers):
        source = int(row["source_device"])
        contexts.append({
            "spec": row,
            "peer": peer,
            "forward": P2PCopier(source, args.destination_device),
            "backward": P2PCopier(args.destination_device, source),
        })

    # Compile all path and monitor work before readiness.
    preflight_echo = []
    for context in contexts:
        peer = context["peer"]
        context["forward"].copy(remote_forward, peer.forward.view(cp.float32))
        with cp.cuda.Device(args.destination_device):
            runtime.launch(use_graph=True)
            runtime.stream.synchronize()
        context["backward"].copy(peer.backward.view(cp.float32), remote_backward)
        context["forward"].copy(
            verify_backward, peer.backward.view(cp.float32)
        )
        with cp.cuda.Device(args.destination_device):
            preflight_echo.append(bool(
                cp.array_equal(verify_backward, remote_backward).item()
            ))
    if not all(preflight_echo):
        raise RuntimeError("NRx response echo preflight failed")
    with cp.cuda.Device(args.destination_device):
        if not bool(cp.all(cp.isfinite(remote_backward)).item()):
            raise RuntimeError("non-finite output during monitor warmup")

    records = []
    activations = []
    handles_closed = False
    error = None
    try:
        for peer in peers:
            peer.mark_ready()
        schedule = wait_json(args.schedule_file, args.ready_timeout_s)
        first_release_ns = int(schedule["first_release_wall_ns"])
        period_ns = int(schedule["period_ns"])
        if int(schedule["iterations"]) != args.iterations:
            raise RuntimeError("schedule iteration mismatch")

        for index in range(args.iterations):
            sequence = index + 1
            release_ns = first_release_ns + index * period_ns
            wait_until_ns(
                release_ns - round(args.release_prewarm_lead_ms * 1e6)
            )
            activation_started_ns = time.perf_counter_ns()
            for context in contexts:
                peer = context["peer"]
                context["forward"].copy(
                    remote_forward, peer.forward.view(cp.float32)
                )
                with cp.cuda.Device(args.destination_device):
                    runtime.launch(use_graph=True)
                    runtime.stream.synchronize()
                context["backward"].copy(
                    peer.backward.view(cp.float32), remote_backward
                )
            activation_completed_ns = time.perf_counter_ns()
            activations.append({
                "sequence": sequence,
                "started_ns": activation_started_ns,
                "completed_ns": activation_completed_ns,
                "release_to_start_ms": (
                    activation_started_ns - release_ns
                ) / 1e6,
                "release_to_complete_ms": (
                    activation_completed_ns - release_ns
                ) / 1e6,
                "host_ms": (
                    activation_completed_ns - activation_started_ns
                ) / 1e6,
            })
            for context in contexts:
                peer = context["peer"]
                row = context["spec"]
                wait_sequence(peer, sequence, args.unit_timeout_s)
                observed_ns = time.perf_counter_ns()
                forward = context["forward"].copy(
                    remote_forward, peer.forward.view(cp.float32)
                )
                input_echo = context["backward"].copy(
                    peer.forward.view(cp.float32), remote_forward
                )
                with cp.cuda.Device(args.destination_device):
                    begin = cp.cuda.Event()
                    end = cp.cuda.Event()
                    wall_begin_ns = time.perf_counter_ns()
                    with runtime.stream:
                        begin.record()
                        runtime.launch(use_graph=True)
                        end.record()
                    end.synchronize()
                    nrx_gpu_ms = float(cp.cuda.get_elapsed_time(begin, end))
                    nrx_host_ms = (time.perf_counter_ns() - wall_begin_ns) / 1e6
                    finite = bool(cp.all(cp.isfinite(remote_backward)).item())
                if not finite:
                    raise RuntimeError(
                        f"non-finite NRx output sequence={sequence} key={row['key']}"
                    )
                backward = context["backward"].copy(
                    peer.backward.view(cp.float32), remote_backward
                )
                echo = context["forward"].copy(
                    verify_backward, peer.backward.view(cp.float32)
                )
                with cp.cuda.Device(args.destination_device):
                    backward_echo_equal = bool(
                        cp.array_equal(verify_backward, remote_backward).item()
                    )
                if not backward_echo_equal:
                    raise RuntimeError(
                        f"NRx response echo mismatch sequence={sequence} "
                        f"key={row['key']}"
                    )
                completed_ns = time.perf_counter_ns()
                peer.publish_backward(sequence)
                records.append({
                    "sequence": sequence,
                    "key": row["key"],
                    "release_wall_ns": release_ns,
                    "observed_ns": observed_ns,
                    "completed_ns": completed_ns,
                    "release_to_complete_ms": (completed_ns - release_ns) / 1e6,
                    "forward_gpu_us": forward["gpu_us"],
                    "forward_host_us": forward["host_us"],
                    "input_echo_gpu_us": input_echo["gpu_us"],
                    "input_echo_host_us": input_echo["host_us"],
                    "nrx_gpu_ms": nrx_gpu_ms,
                    "nrx_host_ms": nrx_host_ms,
                    "backward_gpu_us": backward["gpu_us"],
                    "backward_host_us": backward["host_us"],
                    "backward_echo_gpu_us": echo["gpu_us"],
                    "backward_echo_host_us": echo["host_us"],
                    "backward_echo_equal": backward_echo_equal,
                    "worker_path_ms": (completed_ns - observed_ns) / 1e6,
                    "output_finite": finite,
                })

        for peer in peers:
            wait_sequence(peer, TERMINATE_SEQ, args.ready_timeout_s)
        for row, peer in zip(specs, peers):
            cp.cuda.runtime.setDevice(int(row["source_device"]))
            close_peer_before_ack(peer)
        handles_closed = True
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        result = {
            "schema": "softwall-actual-nrx-repeated-worker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "destination_device": args.destination_device,
            "iterations": args.iterations,
            "peer_count": len(specs),
            "release_prewarm_lead_ms": args.release_prewarm_lead_ms,
            "peer_access": access,
            "preflight_echo_equal": preflight_echo,
            "completed_units": len(records),
            "finite_units": sum(row["output_finite"] for row in records),
            "ipc_handles_closed_before_ack": handles_closed,
            "forward_gpu_us": summarize([row["forward_gpu_us"] for row in records]),
            "input_echo_gpu_us": summarize([
                row["input_echo_gpu_us"] for row in records
            ]),
            "nrx_gpu_ms": summarize([row["nrx_gpu_ms"] for row in records]),
            "backward_gpu_us": summarize([row["backward_gpu_us"] for row in records]),
            "backward_echo_gpu_us": summarize([
                row["backward_echo_gpu_us"] for row in records
            ]),
            "worker_path_ms": summarize([row["worker_path_ms"] for row in records]),
            "release_to_complete_ms": summarize([
                row["release_to_complete_ms"] for row in records
            ]),
            "activation_host_ms": summarize([
                row["host_ms"] for row in activations
            ]),
            "activations": activations,
            "records": records,
            "error": error,
        }
        atomic_json(args.output, result)
        if not handles_closed:
            for peer in peers:
                try:
                    peer.close()
                except BaseException:
                    pass


if __name__ == "__main__":
    main()
