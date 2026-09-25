#!/usr/bin/env python3
"""Remote full NeuralRx pipeline from raw frequency-domain IQ."""

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
from multigpu_p2p_ipc_gate import P2PCopier, enable_peer_access, summarize


SLOT_SHAPE = (3276, 14, 4)
SLOT_ELEMENTS = int(np.prod(SLOT_SHAPE))
FWD_ELEMENTS = 2 * SLOT_ELEMENTS
BACKWARD_ELEMENTS = 4


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def pin_to_allowed_cpu(index: int | None) -> list[int]:
    allowed = sorted(os.sched_getaffinity(0))
    if index is not None:
        if not allowed:
            raise RuntimeError("no CPUs in process affinity mask")
        selected = allowed[index % len(allowed)]
        os.sched_setaffinity(0, {selected})
    return sorted(os.sched_getaffinity(0))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--destination-device", type=int, default=1)
    parser.add_argument("--same-stream", action="store_true")
    parser.add_argument("--busy-poll", action="store_true")
    parser.add_argument("--cpu-affinity-index", type=int)
    parser.add_argument("--profile-stages", action="store_true")
    parser.add_argument(
        "--input-mode",
        choices=("transient_sync", "persistent_sync", "persistent_ordered"),
        default="transient_sync",
        help=(
            "How raw real/imag P2P input becomes the cuPHY complex slot. "
            "persistent_ordered keeps the buffer and relies on the receiver "
            "stream dependency instead of a host synchronization."
        ),
    )
    args = parser.parse_args()

    cpu_affinity = pin_to_allowed_cpu(args.cpu_affinity_index)

    cp.cuda.runtime.setDevice(args.source_device)
    peer_access = enable_peer_access(args.source_device, args.destination_device)
    peer = CudaIpcPeer(args.tag, 120.0, directory=args.ipc_dir)
    with cp.cuda.Device(args.destination_device):
        remote_forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
        remote_backward = cp.empty(BACKWARD_ELEMENTS, dtype=cp.float32)
        receiver = PairedDualReceiver(
            args.engine,
            seed=args.seed,
            device=args.destination_device,
            direct_nrx_same_stream=args.same_stream,
        )
        persistent_rx_slot = None
        if args.input_mode != "transient_sync":
            persistent_rx_slot = cp.empty(
                SLOT_SHAPE, dtype=cp.complex64, order="F"
            )
    forward_copy = P2PCopier(args.source_device, args.destination_device)
    backward_copy = P2PCopier(args.destination_device, args.source_device)

    # Prewarm lazy mappings and the full remote receiver before readiness.
    forward_copy.copy(remote_forward, peer.forward.view(cp.float32))
    with cp.cuda.Device(args.destination_device):
        receiver.run_neural()
        remote_backward.fill(0)
    backward_copy.copy(peer.backward.view(cp.float32), remote_backward)

    records = []
    error = None
    try:
        peer.mark_ready()
        last = 0
        while True:
            sequence = peer.read_forward()
            if sequence == TERMINATE_SEQ:
                break
            if sequence <= last:
                if not args.busy_poll:
                    time.sleep(0)
                continue
            if sequence != last + 1:
                raise RuntimeError(f"sequence skip {last + 1} -> {sequence}")
            observed_ns = time.perf_counter_ns()
            forward = forward_copy.copy(
                remote_forward, peer.forward.view(cp.float32)
            )
            with cp.cuda.Device(args.destination_device), receiver.stream:
                real = remote_forward[:SLOT_ELEMENTS].reshape(SLOT_SHAPE)
                imag = remote_forward[SLOT_ELEMENTS:].reshape(SLOT_SHAPE)
                if persistent_rx_slot is None:
                    receiver.rx_slot = cp.asfortranarray(real + 1j * imag)
                else:
                    # Preserve the Fortran layout expected by cuPHY without
                    # allocating a complex temporary on every request.
                    cp.copyto(persistent_rx_slot.real, real)
                    cp.copyto(persistent_rx_slot.imag, imag)
                    receiver.rx_slot = persistent_rx_slot
            if args.input_mode != "persistent_ordered":
                receiver.stream.synchronize()
            stage_profile = None
            if args.profile_stages:
                stage_profile = receiver.profile_neural_direct_once()
                neural = (
                    stage_profile["total_gpu_ms"],
                    stage_profile["correct"],
                    stage_profile["crc_failures"],
                    stage_profile["payload_mismatches"],
                )
            else:
                neural = receiver.run_neural()
            with cp.cuda.Device(args.destination_device):
                remote_backward.set(np.asarray([
                    float(neural[1]),
                    float(neural[2]),
                    float(neural[3]),
                    float(neural[0]),
                ], dtype=np.float32))
            backward = backward_copy.copy(
                peer.backward.view(cp.float32), remote_backward
            )
            completed_ns = time.perf_counter_ns()
            peer.publish_backward(sequence)
            records.append({
                "sequence": sequence,
                "forward_gpu_us": forward["gpu_us"],
                "forward_host_us": forward["host_us"],
                "neural_gpu_ms": neural[0],
                "neural_correct": bool(neural[1]),
                "neural_crc_failures": neural[2],
                "neural_payload_mismatches": neural[3],
                "backward_gpu_us": backward["gpu_us"],
                "backward_host_us": backward["host_us"],
                "worker_path_ms": (completed_ns - observed_ns) / 1e6,
                "stage_profile": stage_profile,
            })
            last = sequence
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        result = {
            "schema": "softwall-c163-raw-p2p-nrx-worker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "source_device": args.source_device,
            "destination_device": args.destination_device,
            "peer_access": peer_access,
            "same_stream": args.same_stream,
            "busy_poll": args.busy_poll,
            "profile_stages": args.profile_stages,
            "input_mode": args.input_mode,
            "cpu_affinity": cpu_affinity,
            "completed_units": len(records),
            "sequences_contiguous": [x["sequence"] for x in records]
            == list(range(1, len(records) + 1)),
            "correct_units": sum(x["neural_correct"] for x in records),
            "forward_gpu_us": summarize([x["forward_gpu_us"] for x in records]),
            "neural_gpu_ms": summarize([x["neural_gpu_ms"] for x in records]),
            "backward_gpu_us": summarize([x["backward_gpu_us"] for x in records]),
            "worker_path_ms": summarize([x["worker_path_ms"] for x in records]),
            "error": error,
            "records": records,
        }
        atomic_json(args.output, result)
        peer.close()


if __name__ == "__main__":
    main()
