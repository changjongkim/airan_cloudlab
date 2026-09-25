#!/usr/bin/env python3
"""One persistent cuPHY conventional worker shared by two RAN homes."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np

from dual_receiver_phy import PairedDualReceiver
from isca_v2.cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from multigpu_p2p_ipc_gate import (
    P2PCopier,
    close_peer_before_ack,
    enable_peer_access,
    summarize,
)


RX_SHAPE = (1, 3276, 12, 4)
RX_ELEMENTS = int(np.prod(RX_SHAPE))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def wait_sequence(peer, expected, timeout_s):
    deadline = time.monotonic() + timeout_s
    observed = peer.read_forward()
    while time.monotonic() < deadline:
        observed = peer.read_forward()
        if observed == expected:
            return
        if observed == TERMINATE_SEQ:
            raise RuntimeError(
                f"home terminated before sequence {expected}"
            )
        if observed > expected:
            raise RuntimeError(
                f"home doorbell skipped expected={expected} observed={observed}"
            )
        time.sleep(0)
    raise TimeoutError(
        f"home request timeout expected={expected} observed={observed}"
    )


def install_window(receiver, remote_forward):
    with cp.cuda.Device(receiver.device), receiver.stream:
        real = remote_forward[:RX_ELEMENTS].reshape(RX_SHAPE)
        imag = remote_forward[RX_ELEMENTS:].reshape(RX_SHAPE)
        window = real + 1j * imag
        # Keep the configured full-slot shape and replace the PUSCH region.
        receiver.rx_slot = cp.array(receiver.clean_rx_slot, order="F", copy=True)
        receiver.rx_slot[
            :, receiver.start_sym:receiver.start_sym + receiver.num_symbols, :
        ] = window[0]
        receiver.stream.synchronize()


def run_conventional(receiver, remote_backward):
    with cp.cuda.Device(receiver.device), receiver.stream:
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        wall_begin_ns = time.perf_counter_ns()
        begin.record()
        output = receiver.conventional_once()
        blocks, crc_values = output
        payload = cp.asarray(blocks[0]).reshape(-1).astype(cp.uint8)
        crc = cp.asarray(crc_values[0]).reshape(-1).astype(cp.uint8)
        if remote_backward.size != payload.size + 1:
            raise RuntimeError(
                f"shared conventional backward size mismatch "
                f"buffer={remote_backward.size} payload={payload.size}"
            )
        remote_backward[0] = crc[0]
        remote_backward[1:] = payload
        end.record()
        end.synchronize()
        correct, crc_failures, payload_mismatches = receiver._verify(output)
        return {
            "gpu_ms": float(cp.cuda.get_elapsed_time(begin, end)),
            "host_ms": (time.perf_counter_ns() - wall_begin_ns) / 1e6,
            "correct": bool(correct),
            "crc_failures": int(crc_failures),
            "payload_mismatches": int(payload_mismatches),
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag0", required=True)
    parser.add_argument("--tag1", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--source-device0", type=int, default=0)
    parser.add_argument("--source-device1", type=int, default=1)
    parser.add_argument("--destination-device", type=int, default=2)
    parser.add_argument("--receiver-seed0", type=int, required=True)
    parser.add_argument("--receiver-seed1", type=int, required=True)
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    parser.add_argument("--warmup", type=int, default=10)
    args = parser.parse_args()
    source_devices = (args.source_device0, args.source_device1)
    if (args.iterations <= 0 or args.warmup < 0
            or len(set(source_devices + (args.destination_device,))) != 3):
        parser.error("positive counts and three distinct devices are required")

    peers = []
    for tag, source in zip((args.tag0, args.tag1), source_devices):
        cp.cuda.runtime.setDevice(source)
        peers.append(CudaIpcPeer(tag, args.ready_timeout_s, directory=args.ipc_dir))

    access = {}
    for source in source_devices:
        access.update(enable_peer_access(source, args.destination_device))

    receiver_seeds = (args.receiver_seed0, args.receiver_seed1)
    contexts = []
    with cp.cuda.Device(args.destination_device):
        for home, seed in enumerate(receiver_seeds):
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
                    raise RuntimeError(f"home {home} shared worker warmup failed")
            contexts.append({
                "receiver": receiver,
                "remote_forward": remote_forward,
                "remote_backward": remote_backward,
                "forward_copy": P2PCopier(source_devices[home], args.destination_device),
                "backward_copy": P2PCopier(args.destination_device, source_devices[home]),
            })

    records = []
    error = None
    handles_closed = False
    started = time.perf_counter()
    try:
        for peer in peers:
            peer.mark_ready()
        # This is a physical global-order canary. Each round is ordered by the
        # global key (sequence, home), independent of host arrival order.
        for sequence in range(1, args.iterations + 1):
            for home, (peer, context) in enumerate(zip(peers, contexts)):
                wait_sequence(peer, sequence, args.unit_timeout_s)
                observed_ns = time.perf_counter_ns()
                forward = context["forward_copy"].copy(
                    context["remote_forward"], peer.forward.view(cp.float32)
                )
                install_window(context["receiver"], context["remote_forward"])
                decoded = run_conventional(
                    context["receiver"], context["remote_backward"]
                )
                if not decoded["correct"]:
                    raise RuntimeError(
                        f"shared conventional decode failed home={home} sequence={sequence}"
                    )
                backward = context["backward_copy"].copy(
                    peer.backward.view(cp.uint8), context["remote_backward"]
                )
                completed_ns = time.perf_counter_ns()
                peer.publish_backward(sequence)
                records.append({
                    "global_position": len(records),
                    "sequence": sequence,
                    "home_id": home,
                    "observed_ns": observed_ns,
                    "completed_ns": completed_ns,
                    "forward_gpu_us": forward["gpu_us"],
                    "conventional_gpu_ms": decoded["gpu_ms"],
                    "conventional_host_ms": decoded["host_ms"],
                    "backward_gpu_us": backward["gpu_us"],
                    "worker_path_ms": (completed_ns - observed_ns) / 1e6,
                    "correct": decoded["correct"],
                })
        for peer in peers:
            wait_sequence(peer, TERMINATE_SEQ, args.ready_timeout_s)
        for source, peer in zip(source_devices, peers):
            cp.cuda.runtime.setDevice(source)
            close_peer_before_ack(peer)
        handles_closed = True
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        result = {
            "schema": "softwall-shared-conventional-worker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_devices": list(source_devices),
            "destination_device": args.destination_device,
            "iterations": args.iterations,
            "warmup": args.warmup,
            "peer_access": access,
            "completed_units": len(records),
            "correct_units": sum(row["correct"] for row in records),
            "global_order": [
                [row["sequence"], row["home_id"]] for row in records
            ],
            "global_order_exact": [
                [sequence, home]
                for sequence in range(1, args.iterations + 1)
                for home in (0, 1)
            ] == [[row["sequence"], row["home_id"]] for row in records],
            "ipc_handles_closed_before_ack": handles_closed,
            "forward_gpu_us": summarize([row["forward_gpu_us"] for row in records]),
            "conventional_gpu_ms": summarize([
                row["conventional_gpu_ms"] for row in records
            ]),
            "backward_gpu_us": summarize([row["backward_gpu_us"] for row in records]),
            "worker_path_ms": summarize([row["worker_path_ms"] for row in records]),
            "wall_s": time.perf_counter() - started,
            "error": error,
            "records": records,
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
