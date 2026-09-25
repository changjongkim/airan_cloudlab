#!/usr/bin/env python3
"""RAN-home side of the C153 integrated shared-recovery canary."""

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
from shared_conventional_owner import RX_ELEMENTS, atomic_json, prepare_forward


def wait_json(path: Path, timeout_s: float) -> dict:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError):
            time.sleep(0.005)
    raise TimeoutError(f"timed out waiting for {path}")


def wait_until_ns(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 2_000_000:
            time.sleep((remaining - 1_000_000) / 1e9)
        else:
            time.sleep(0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--home-id", type=int, choices=(0, 1), required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--source-device", type=int, required=True)
    parser.add_argument("--destination-device", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--release-file", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--receiver-seed", type=int, required=True)
    parser.add_argument("--channel-seed", type=int, required=True)
    parser.add_argument("--snr-db", type=float, default=20.0)
    parser.add_argument("--expiry-ms", type=float, default=155.0)
    parser.add_argument("--ready-timeout-s", type=float, default=240.0)
    parser.add_argument("--unit-timeout-s", type=float, default=5.0)
    args = parser.parse_args()
    if args.source_device == args.destination_device or args.expiry_ms <= 0:
        parser.error("source/destination must differ and expiry must be positive")

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
    result = {
        "schema": "softwall-c153-integrated-owner-v1",
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "home_id": args.home_id,
        "request_id": args.request_id,
        "source_device": args.source_device,
        "destination_device": args.destination_device,
        "receiver_seed": args.receiver_seed,
        "channel_seed": args.channel_seed,
        "snr_db": args.snr_db,
        "expiry_ms": args.expiry_ms,
        "published": False,
        "termination_acknowledged": False,
        "error": None,
    }
    try:
        owner.wait_ready(args.ready_timeout_s)
        plan = wait_json(args.release_file, args.ready_timeout_s)
        key = [f"home{args.home_id}", args.request_id]
        if key not in plan["physical_recovery_keys"]:
            raise RuntimeError(f"request is absent from physical plan: {key}")
        release_ns = int(plan["release_wall_ns"])
        preparation_started_ns = time.perf_counter_ns()
        receiver.apply_rayleigh_awgn(
            args.snr_db,
            args.channel_seed,
            noise_reference="pre_fading",
        )
        prepare_gpu_ms = prepare_forward(receiver, forward)
        preparation_completed_ns = time.perf_counter_ns()
        wait_until_ns(release_ns)
        published_ns = time.perf_counter_ns()
        owner.publish_forward(1)
        result["published"] = True
        owner.wait_backward(1, args.unit_timeout_s)
        returned_ns = time.perf_counter_ns()
        response = cp.asnumpy(backward)
        commit_ns = time.perf_counter_ns()
        crc_value = int(response[0])
        payload_match = bool(np.array_equal(response[1:], receiver.reference_tb))
        result.update({
            "release_wall_ns": release_ns,
            "preparation_started_ns": preparation_started_ns,
            "preparation_completed_ns": preparation_completed_ns,
            "input_ready_before_release": preparation_completed_ns <= release_ns,
            "published_ns": published_ns,
            "returned_ns": returned_ns,
            "commit_ns": commit_ns,
            "prepare_gpu_ms": prepare_gpu_ms,
            "preparation_ms": (
                preparation_completed_ns - preparation_started_ns
            ) / 1e6,
            "release_to_publish_ms": (published_ns - release_ns) / 1e6,
            "published_to_return_ms": (returned_ns - published_ns) / 1e6,
            "release_to_commit_ms": (commit_ns - release_ns) / 1e6,
            "crc_value": crc_value,
            "payload_match": payload_match,
            "correct": crc_value == 0 and payload_match,
            "deadline_miss": commit_ns > release_ns + round(args.expiry_ms * 1e6),
        })
        if not result["correct"]:
            raise RuntimeError("decoded conventional result is incorrect")
        owner.publish_forward(TERMINATE_SEQ)
        owner.wait_backward(TERMINATE_SEQ, args.ready_timeout_s)
        result["termination_acknowledged"] = True
    except BaseException as error:
        result["error"] = repr(error)
        try:
            owner.publish_forward(TERMINATE_SEQ)
        except BaseException:
            pass
        raise
    finally:
        atomic_json(args.output, result)
        owner.close()


if __name__ == "__main__":
    main()
