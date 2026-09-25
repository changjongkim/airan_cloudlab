#!/usr/bin/env python3
"""Persistent remote NeuralRx using GPU0 CUDA IPC and GPU0<->GPU1 P2P."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from isca_v2.cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from multigpu_p2p_ipc_gate import P2PCopier, enable_peer_access, summarize
from nrx_trt_direct import DirectNrx


RX_SHAPE = (1, 3276, 12, 4)
CE_SHAPE = (1, 4914, 1, 4)
LLR_SHAPE = (2, 1, 3276, 12)
RX_ELEMENTS = 1 * 3276 * 12 * 4
CE_ELEMENTS = 1 * 4914 * 1 * 4
FWD_ELEMENTS = 2 * RX_ELEMENTS + 2 * CE_ELEMENTS
LLR_ELEMENTS = 2 * 1 * 3276 * 12


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def section(flat: cp.ndarray, offset: int, count: int, shape: tuple[int, ...]):
    value = flat[offset:offset + count].reshape(shape, order="C")
    if not value.flags.c_contiguous:
        raise RuntimeError(f"non-contiguous P2P NRx section {shape}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-device", type=int, default=0)
    parser.add_argument("--destination-device", type=int, default=1)
    parser.add_argument("--ready-timeout-s", type=float, default=120.0)
    parser.add_argument("--graph-warmup", type=int, default=20)
    args = parser.parse_args()
    if args.source_device == args.destination_device or args.graph_warmup < 0:
        parser.error("distinct devices and non-negative warmup are required")

    cp.cuda.runtime.setDevice(args.source_device)
    peer_access = enable_peer_access(args.source_device, args.destination_device)
    peer = CudaIpcPeer(args.tag, args.ready_timeout_s, directory=args.ipc_dir)
    with cp.cuda.Device(args.destination_device):
        remote_forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
        remote_backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
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

    forward_copy = P2PCopier(args.source_device, args.destination_device)
    backward_copy = P2PCopier(args.destination_device, args.source_device)
    # Exercise lazy P2P mapping before publishing readiness. The controller has
    # not yet issued a request, so the payload value is irrelevant here.
    forward_copy.copy(remote_forward, peer.forward.view(cp.float32))
    with cp.cuda.Device(args.destination_device):
        runtime.launch(use_graph=True)
        runtime.stream.synchronize()
    backward_copy.copy(peer.backward.view(cp.float32), remote_backward)

    records = []
    error = None
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
            observed_ns = time.perf_counter_ns()
            forward = forward_copy.copy(
                remote_forward, peer.forward.view(cp.float32)
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
                output_finite = bool(cp.all(cp.isfinite(remote_backward)).item())
            if not output_finite:
                raise RuntimeError(f"non-finite NRx output sequence={sequence}")
            backward = backward_copy.copy(
                peer.backward.view(cp.float32), remote_backward
            )
            completed_ns = time.perf_counter_ns()
            peer.publish_backward(sequence)
            records.append({
                "sequence": sequence,
                "observed_ns": observed_ns,
                "completed_ns": completed_ns,
                "published_ns": time.perf_counter_ns(),
                "forward_gpu_us": forward["gpu_us"],
                "forward_host_us": forward["host_us"],
                "nrx_gpu_ms": nrx_gpu_ms,
                "nrx_host_ms": nrx_host_ms,
                "backward_gpu_us": backward["gpu_us"],
                "backward_host_us": backward["host_us"],
                "output_finite": output_finite,
                "worker_path_ms": (completed_ns - observed_ns) / 1e6,
            })
            last = sequence
    except BaseException as caught:
        error = repr(caught)
        raise
    finally:
        result = {
            "schema": "softwall-multigpu-p2p-nrx-worker-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "tag": args.tag,
            "source_device": args.source_device,
            "destination_device": args.destination_device,
            "peer_access": peer_access,
            "forward_bytes": FWD_ELEMENTS * 4,
            "backward_bytes": LLR_ELEMENTS * 4,
            "graph_warmup": args.graph_warmup,
            "completed_units": len(records),
            "last_sequence": records[-1]["sequence"] if records else 0,
            "sequences_contiguous": [row["sequence"] for row in records]
            == list(range(1, len(records) + 1)),
            "nonfinite_outputs": sum(not row["output_finite"] for row in records),
            "forward_gpu_us": summarize([row["forward_gpu_us"] for row in records]),
            "nrx_gpu_ms": summarize([row["nrx_gpu_ms"] for row in records]),
            "backward_gpu_us": summarize([row["backward_gpu_us"] for row in records]),
            "worker_path_ms": summarize([row["worker_path_ms"] for row in records]),
            "wall_s": time.perf_counter() - started,
            "error": error,
            "records": records,
        }
        atomic_json(args.output, result)
        peer.close()


if __name__ == "__main__":
    main()

