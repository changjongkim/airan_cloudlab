#!/usr/bin/env python3
"""Cross-process CUDA-IPC plus full-GPU P2P payload integrity gate.

The owner process creates the exact forward/backward buffers used by the
SoftWall NeuralRx path on GPU0 and exports them with CUDA IPC.  The worker
imports those GPU0 allocations, copies the forward payload to GPU1 with
cudaMemcpyPeerAsync, validates it on GPU1, produces a deterministic LLR-sized
reply, and copies that reply back to the imported GPU0 allocation.

Only scalar integrity results and sequence doorbells reach host memory; the
payload stays in GPU memory.  This is a transport gate, not an NRx service or
deadline qualification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from isca_v2.cuda_ipc_channel import (
    BWD_OFFSET,
    CudaIpcOwner,
    CudaIpcPeer,
    TERMINATE_SEQ,
)


RX_ELEMENTS = 1 * 3276 * 12 * 4
CE_ELEMENTS = 1 * 4914 * 1 * 4
FWD_ELEMENTS = 2 * RX_ELEMENTS + 2 * CE_ELEMENTS
LLR_ELEMENTS = 2 * 1 * 3276 * 12


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pattern(sequence: int) -> float:
    # All values are exactly representable in float32 and change every unit.
    return float((sequence * 17) % 251 + 1)


def summarize(values: list[float]) -> dict[str, float | int]:
    if not values:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p99": 0.0, "max": 0.0}
    ordered = sorted(values)

    def percentile(fraction: float) -> float:
        return ordered[round((len(ordered) - 1) * fraction)]

    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.50),
        "p99": percentile(0.99),
        "max": ordered[-1],
    }


def device_name(index: int) -> str:
    value = cp.cuda.runtime.getDeviceProperties(index)["name"]
    return value.decode() if isinstance(value, bytes) else str(value)


class P2PCopier:
    def __init__(self, source_device: int, destination_device: int) -> None:
        self.source_device = source_device
        self.destination_device = destination_device
        with cp.cuda.Device(source_device):
            self.stream = cp.cuda.Stream(non_blocking=True)

    def copy(self, destination: cp.ndarray, source: cp.ndarray) -> dict[str, float]:
        if destination.nbytes != source.nbytes:
            raise RuntimeError(
                f"P2P size mismatch destination={destination.nbytes} source={source.nbytes}"
            )
        with cp.cuda.Device(self.source_device):
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            host_begin_ns = time.perf_counter_ns()
            begin.record(self.stream)
            cp.cuda.runtime.memcpyPeerAsync(
                destination.data.ptr,
                self.destination_device,
                source.data.ptr,
                self.source_device,
                source.nbytes,
                self.stream.ptr,
            )
            end.record(self.stream)
            end.synchronize()
            host_end_ns = time.perf_counter_ns()
        return {
            "gpu_us": float(cp.cuda.get_elapsed_time(begin, end) * 1000.0),
            "host_us": (host_end_ns - host_begin_ns) / 1000.0,
        }


def enable_peer_access(first: int, second: int) -> dict[str, int]:
    result = {}
    for source, destination in ((first, second), (second, first)):
        key = f"{source}_to_{destination}"
        available = int(cp.cuda.runtime.deviceCanAccessPeer(source, destination))
        result[key] = available
        if not available:
            raise RuntimeError(f"CUDA P2P unavailable {key}")
        with cp.cuda.Device(source):
            try:
                cp.cuda.runtime.deviceEnablePeerAccess(destination)
            except cp.cuda.runtime.CUDARuntimeError as error:
                already_enabled = getattr(
                    cp.cuda.runtime, "errorPeerAccessAlreadyEnabled", 704
                )
                if error.status != already_enabled:
                    raise
    return result


def common_result(args: argparse.Namespace) -> dict:
    script = Path(__file__).resolve()
    channel = Path(args.channel_source).resolve()
    return {
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "role": args.role,
        "tag": args.tag,
        "iterations": args.iterations,
        "source_device": args.source_device,
        "destination_device": args.destination_device,
        "forward_elements": FWD_ELEMENTS,
        "forward_bytes": FWD_ELEMENTS * 4,
        "backward_elements": LLR_ELEMENTS,
        "backward_bytes": LLR_ELEMENTS * 4,
        "source_sha256": {
            str(script): sha256(script),
            str(channel): sha256(channel),
        },
    }


def run_owner(args: argparse.Namespace) -> None:
    cp.cuda.runtime.setDevice(args.source_device)
    forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
    owner = CudaIpcOwner(args.tag, forward, backward, directory=args.ipc_dir)
    records = []
    error = None
    termination_acknowledged = False
    started = time.perf_counter()
    try:
        owner.wait_ready(args.ready_timeout_s)
        for sequence in range(1, args.iterations + 1):
            expected = pattern(sequence)
            with cp.cuda.Device(args.source_device):
                forward.fill(cp.float32(expected))
                cp.cuda.get_current_stream().synchronize()
            published_ns = time.perf_counter_ns()
            owner.publish_forward(sequence)
            owner.wait_backward(sequence, args.unit_timeout_s)
            returned_ns = time.perf_counter_ns()
            with cp.cuda.Device(args.source_device):
                backward_ok = bool(cp.all(backward == cp.float32(expected + 0.5)).item())
            records.append({
                "sequence": sequence,
                "expected_forward": expected,
                "backward_ok": backward_ok,
                "round_trip_us": (returned_ns - published_ns) / 1000.0,
            })
            if not backward_ok:
                raise RuntimeError(f"backward integrity failure sequence={sequence}")
        owner.publish_forward(TERMINATE_SEQ)
        owner.wait_backward(TERMINATE_SEQ, args.ready_timeout_s)
        termination_acknowledged = True
    except BaseException as caught:
        error = repr(caught)
        try:
            owner.publish_forward(TERMINATE_SEQ)
        except BaseException:
            pass
        raise
    finally:
        result = common_result(args)
        result.update({
            "schema": "softwall-multigpu-p2p-ipc-owner-v1",
            "device_name": device_name(args.source_device),
            "completed_units": len(records),
            "integrity_errors": sum(not row["backward_ok"] for row in records),
            "sequences_contiguous": [row["sequence"] for row in records]
            == list(range(1, len(records) + 1)),
            "termination_acknowledged": termination_acknowledged,
            "round_trip_us": summarize([row["round_trip_us"] for row in records]),
            "wall_s": time.perf_counter() - started,
            "error": error,
            "records": records,
        })
        atomic_json(args.output, result)
        owner.close()


def close_peer_before_ack(peer: CudaIpcPeer) -> None:
    # The exporting process must not free an allocation while an importing
    # context still has it open. Close both GPU handles first, then publish the
    # termination acknowledgement through the independent host mmap control.
    cp.cuda.runtime.ipcCloseMemHandle(peer.forward_ptr)
    cp.cuda.runtime.ipcCloseMemHandle(peer.backward_ptr)
    peer.control.write(BWD_OFFSET, TERMINATE_SEQ)
    peer.control.close()


def run_worker(args: argparse.Namespace) -> None:
    if args.source_device == args.destination_device:
        raise ValueError("P2P gate requires distinct source and destination devices")
    cp.cuda.runtime.setDevice(args.source_device)
    peer_access = enable_peer_access(args.source_device, args.destination_device)
    peer = CudaIpcPeer(args.tag, args.ready_timeout_s, directory=args.ipc_dir)
    with cp.cuda.Device(args.destination_device):
        remote_forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
        remote_backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
    forward_copy = P2PCopier(args.source_device, args.destination_device)
    backward_copy = P2PCopier(args.destination_device, args.source_device)
    records = []
    error = None
    handles_closed = False
    started = time.perf_counter()
    try:
        peer.mark_ready()
        last = 0
        while True:
            sequence = peer.read_forward()
            if sequence == TERMINATE_SEQ:
                break
            if sequence <= last:
                time.sleep(0)
                continue
            if sequence != last + 1:
                raise RuntimeError(
                    f"sequence skip expected={last + 1} observed={sequence}"
                )
            expected = pattern(sequence)
            forward_timing = forward_copy.copy(remote_forward, peer.forward.view(cp.float32))
            with cp.cuda.Device(args.destination_device):
                forward_ok = bool(
                    cp.all(remote_forward == cp.float32(expected)).item()
                )
                remote_backward.fill(cp.float32(expected + 0.5))
                cp.cuda.get_current_stream().synchronize()
            if not forward_ok:
                raise RuntimeError(f"forward integrity failure sequence={sequence}")
            backward_timing = backward_copy.copy(
                peer.backward.view(cp.float32), remote_backward
            )
            peer.publish_backward(sequence)
            records.append({
                "sequence": sequence,
                "forward_ok": forward_ok,
                "forward_gpu_us": forward_timing["gpu_us"],
                "forward_host_us": forward_timing["host_us"],
                "backward_gpu_us": backward_timing["gpu_us"],
                "backward_host_us": backward_timing["host_us"],
            })
            last = sequence
        close_peer_before_ack(peer)
        handles_closed = True
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        result = common_result(args)
        result.update({
            "schema": "softwall-multigpu-p2p-ipc-worker-v1",
            "source_device_name": device_name(args.source_device),
            "destination_device_name": device_name(args.destination_device),
            "peer_access": peer_access,
            "completed_units": len(records),
            "integrity_errors": sum(not row["forward_ok"] for row in records),
            "sequences_contiguous": [row["sequence"] for row in records]
            == list(range(1, len(records) + 1)),
            "ipc_handles_closed_before_ack": handles_closed,
            "forward_gpu_us": summarize([row["forward_gpu_us"] for row in records]),
            "forward_host_us": summarize([row["forward_host_us"] for row in records]),
            "backward_gpu_us": summarize([row["backward_gpu_us"] for row in records]),
            "backward_host_us": summarize([row["backward_host_us"] for row in records]),
            "wall_s": time.perf_counter() - started,
            "error": error,
            "records": records,
        })
        atomic_json(args.output, result)
        if not handles_closed:
            try:
                peer.close()
            except BaseException:
                pass


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--role", choices=("owner", "worker"), required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--destination-device", type=int, default=1)
    parser.add_argument("--ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--unit-timeout-s", type=float, default=2.0)
    parser.add_argument(
        "--channel-source",
        default="/softwall_task1/isca_v2/cuda_ipc_channel.py",
    )
    args = parser.parse_args()
    if args.iterations <= 0 or min(args.ready_timeout_s, args.unit_timeout_s) <= 0:
        parser.error("iterations and timeouts must be positive")
    args.ipc_dir.mkdir(parents=True, exist_ok=True)
    if args.role == "owner":
        run_owner(args)
    else:
        run_worker(args)


if __name__ == "__main__":
    main()

