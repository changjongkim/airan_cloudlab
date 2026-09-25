#!/usr/bin/env python3
"""One RAN-home owner for the shared conventional-recovery P2P canary."""

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
from isca_v2.cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from softwall_phy import summary


RX_SHAPE = (1, 3276, 12, 4)
RX_ELEMENTS = int(np.prod(RX_SHAPE))


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def prepare_forward(receiver, forward):
    if forward.dtype != cp.float32 or forward.size != 2 * RX_ELEMENTS:
        raise RuntimeError("shared conventional forward-buffer mismatch")
    with cp.cuda.Device(receiver.device), receiver.stream:
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        begin.record()
        window = cp.asarray(receiver.rx_slot)[
            None, :, receiver.start_sym:receiver.start_sym + receiver.num_symbols, :
        ]
        cp.copyto(
            forward[:RX_ELEMENTS].reshape(RX_SHAPE),
            cp.ascontiguousarray(window.real).astype(cp.float32),
        )
        cp.copyto(
            forward[RX_ELEMENTS:].reshape(RX_SHAPE),
            cp.ascontiguousarray(window.imag).astype(cp.float32),
        )
        end.record()
        end.synchronize()
        return float(cp.cuda.get_elapsed_time(begin, end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--source-device", type=int, required=True)
    parser.add_argument("--destination-device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, required=True)
    parser.add_argument("--receiver-seed", type=int, required=True)
    parser.add_argument("--channel-seed-base", type=int, required=True)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--ready-timeout-s", type=float, default=180.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.iterations <= 0 or args.source_device == args.destination_device:
        parser.error("positive iterations and distinct devices are required")

    cp.cuda.runtime.setDevice(args.source_device)
    receiver = PairedDualReceiver(
        args.engine,
        seed=args.receiver_seed,
        device=args.source_device,
        enable_local_neural=False,
    )
    backward_bytes = len(receiver.reference_tb) + 1
    forward = cp.empty(2 * RX_ELEMENTS, dtype=cp.float32)
    backward = cp.empty(backward_bytes, dtype=cp.uint8)
    owner = CudaIpcOwner(args.tag, forward, backward, directory=args.ipc_dir)
    records = []
    error = None
    termination_acknowledged = False
    started = time.perf_counter()
    try:
        owner.wait_ready(args.ready_timeout_s)
        for sequence in range(1, args.iterations + 1):
            receiver.apply_rayleigh_awgn(
                args.snr_db,
                args.channel_seed_base + sequence,
                noise_reference="pre_fading",
            )
            prepare_gpu_ms = prepare_forward(receiver, forward)
            published_ns = time.perf_counter_ns()
            owner.publish_forward(sequence)
            owner.wait_backward(sequence, args.unit_timeout_s)
            returned_ns = time.perf_counter_ns()
            response = cp.asnumpy(backward)
            crc_value = int(response[0])
            payload = response[1:]
            payload_match = bool(np.array_equal(payload, receiver.reference_tb))
            records.append({
                "sequence": sequence,
                "published_ns": published_ns,
                "returned_ns": returned_ns,
                "prepare_gpu_ms": prepare_gpu_ms,
                "round_trip_ms": (returned_ns - published_ns) / 1e6,
                "crc_value": crc_value,
                "payload_match": payload_match,
            })
            if crc_value != 0 or not payload_match:
                raise RuntimeError(
                    f"shared conventional decode mismatch sequence={sequence}"
                )
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
        result = {
            "schema": "softwall-shared-conventional-owner-v1",
            "host": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "home_id": args.home_id,
            "tag": args.tag,
            "source_device": args.source_device,
            "destination_device": args.destination_device,
            "iterations": args.iterations,
            "receiver_seed": args.receiver_seed,
            "channel_seed_base": args.channel_seed_base,
            "snr_db": args.snr_db,
            "forward_bytes": int(forward.nbytes),
            "backward_bytes": int(backward.nbytes),
            "completed_units": len(records),
            "decode_errors": sum(
                row["crc_value"] != 0 or not row["payload_match"]
                for row in records
            ),
            "sequences_contiguous": [row["sequence"] for row in records]
            == list(range(1, len(records) + 1)),
            "termination_acknowledged": termination_acknowledged,
            "prepare_gpu_ms": summary([row["prepare_gpu_ms"] for row in records]),
            "round_trip_ms": summary([row["round_trip_ms"] for row in records]),
            "wall_s": time.perf_counter() - started,
            "error": error,
            "records": records,
        }
        atomic_json(args.output, result)
        owner.close()


if __name__ == "__main__":
    main()
