#!/usr/bin/env python3
"""Persistent MPS NeuralRx endpoint bound to same-device CUDA IPC buffers."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from isca_v2.cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from nrx_trt_direct import DirectNrx
from softwall_phy import summary


RX_SHAPE = (1, 3276, 12, 4)
CE_SHAPE = (1, 4914, 1, 4)
LLR_SHAPE = (2, 1, 3276, 12)
RX_ELEMENTS = 1 * 3276 * 12 * 4
CE_ELEMENTS = 1 * 4914 * 1 * 4


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def section(flat, offset, count, shape):
    value = flat[offset:offset + count].reshape(shape, order="C")
    if not value.flags.c_contiguous:
        raise RuntimeError(f"non-contiguous IPC section {shape}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delay-sequence", type=int, default=0)
    parser.add_argument("--delay-ms", type=float, default=0.0)
    args = parser.parse_args()
    if args.delay_sequence < 0 or args.delay_ms < 0:
        parser.error("delay injection parameters must be non-negative")

    cp.cuda.runtime.setDevice(0)
    peer = CudaIpcPeer(args.tag, 120.0, directory=args.ipc_dir)
    gpu_ms = []
    sequences = []
    execution_windows = []
    started = time.perf_counter()
    runtime = None
    try:
        forward = peer.forward.view(cp.float32)
        backward = peer.backward.view(cp.float32)
        runtime = DirectNrx(args.engine)
        offset = 0
        for name, count, shape in (
            ("rx_slot_real", RX_ELEMENTS, RX_SHAPE),
            ("rx_slot_imag", RX_ELEMENTS, RX_SHAPE),
            ("h_hat_real", CE_ELEMENTS, CE_SHAPE),
            ("h_hat_imag", CE_ELEMENTS, CE_SHAPE),
        ):
            runtime.bind_tensor(name, section(forward, offset, count, shape))
            offset += count
        runtime.bind_tensor(
            "output_1", backward.reshape((1,) + LLR_SHAPE, order="C")
        )
        runtime.inputs["active_dmrs_ports"].fill(1)
        runtime.inputs["dmrs_ofdm_pos"][:] = cp.asarray(
            [[0, 5, 9]], dtype=cp.int32
        )
        runtime.inputs["dmrs_subcarrier_pos"][:] = cp.asarray(
            [[0, 2, 4, 6, 8, 10]], dtype=cp.int32
        )
        runtime.capture_graph()
        for _ in range(20):
            runtime.launch(use_graph=True)
        runtime.stream.synchronize()
        peer.mark_ready()
        print(f"[SAME-REQUEST-NRX] ready tag={args.tag}", flush=True)
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
            begin = cp.cuda.Event()
            end = cp.cuda.Event()
            with runtime.stream:
                begin.record()
                runtime.launch(use_graph=True)
                end.record()
            submitted_ns = time.perf_counter_ns()
            end.synchronize()
            completed_ns = time.perf_counter_ns()
            gpu_ms.append(float(cp.cuda.get_elapsed_time(begin, end)))
            sequences.append(sequence)
            if sequence == args.delay_sequence and args.delay_ms:
                time.sleep(args.delay_ms / 1000.0)
            peer.publish_backward(sequence)
            execution_windows.append({
                "sequence": sequence,
                "observed_ns": observed_ns,
                "submitted_ns": submitted_ns,
                "completed_ns": completed_ns,
                "published_ns": time.perf_counter_ns(),
                "gpu_ms": gpu_ms[-1],
            })
            last = sequence
    finally:
        wall_s = time.perf_counter() - started
        result = {
            "schema": "softwall-same-request-ipc-worker-v2",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "tag": args.tag,
            "completed_units": len(sequences),
            "last_sequence": sequences[-1] if sequences else 0,
            "gpu_ms": summary(gpu_ms),
            "execution_windows": execution_windows,
            "wall_s": wall_s,
            "visible_sm_count": int(
                cp.cuda.runtime.getDeviceProperties(0)["multiProcessorCount"]
            ),
            "delay_sequence": args.delay_sequence,
            "delay_ms": args.delay_ms,
        }
        atomic_json(args.output, result)
        peer.close()
        print(
            f"[SAME-REQUEST-NRX] stopped units={len(sequences)}",
            flush=True,
        )


if __name__ == "__main__":
    main()
