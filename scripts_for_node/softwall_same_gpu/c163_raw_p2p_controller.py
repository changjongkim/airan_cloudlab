#!/usr/bin/env python3
"""Overlap GPU0 conventional with a complete raw-IQ NeuralRx on GPU1."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np

from c163_raw_p2p_nrx_worker import BACKWARD_ELEMENTS, FWD_ELEMENTS, SLOT_ELEMENTS, SLOT_SHAPE
from dual_receiver_phy import PairedDualReceiver
from isca_v2.cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from softwall_phy import summary


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


def fill_raw(receiver: PairedDualReceiver, forward: cp.ndarray) -> float:
    with receiver.stream:
        begin = cp.cuda.Event()
        end = cp.cuda.Event()
        begin.record()
        slot = cp.asarray(receiver.rx_slot)
        if slot.shape != SLOT_SHAPE:
            raise RuntimeError(f"unexpected slot shape {slot.shape}")
        cp.copyto(forward[:SLOT_ELEMENTS].reshape(SLOT_SHAPE), slot.real)
        cp.copyto(forward[SLOT_ELEMENTS:].reshape(SLOT_SHAPE), slot.imag)
        end.record()
    end.synchronize()
    return float(cp.cuda.get_elapsed_time(begin, end))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--deadline-ms", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20358100)
    parser.add_argument("--cpu-affinity-index", type=int)
    parser.add_argument("--profile-conventional", action="store_true")
    parser.add_argument(
        "--conventional-mode",
        choices=("separable", "persistent_monolithic"),
        default="separable",
    )
    args = parser.parse_args()
    if args.profile_conventional and args.conventional_mode != "separable":
        raise ValueError("stage profiling is only defined for separable conventional mode")

    cpu_affinity = pin_to_allowed_cpu(args.cpu_affinity_index)

    cp.cuda.runtime.setDevice(0)
    forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    backward = cp.empty(BACKWARD_ELEMENTS, dtype=cp.float32)
    owner = CudaIpcOwner(args.tag, forward, backward, directory=args.ipc_dir)
    receiver = PairedDualReceiver(
        args.engine,
        seed=args.seed,
        enable_local_neural=False,
        enable_native_conventional_shadow=(
            args.conventional_mode == "persistent_monolithic"
        ),
    )
    run_conventional = (
        receiver.run_native_conventional
        if args.conventional_mode == "persistent_monolithic"
        else receiver.run_conventional
    )
    sequence = 0
    warmup_correct = 0
    records = []
    try:
        owner.wait_ready(120.0)
        for _ in range(args.warmup):
            sequence += 1
            fill_raw(receiver, forward)
            owner.publish_forward(sequence)
            conventional = run_conventional()
            owner.wait_backward(sequence, 2.0)
            marker = cp.asnumpy(backward)
            warmup_correct += int(conventional[1] and marker[0] == 1.0)
        if warmup_correct != args.warmup:
            raise RuntimeError(f"warmup correctness {warmup_correct}/{args.warmup}")

        for iteration in range(args.iterations):
            release_ns = time.perf_counter_ns()
            sequence += 1
            raw_copy_gpu_ms = fill_raw(receiver, forward)
            owner.publish_forward(sequence)
            conventional_stage_profile = None
            if args.profile_conventional:
                conventional_stage_profile = receiver.profile_conventional_once()
                conventional = (
                    conventional_stage_profile["total_gpu_ms"],
                    conventional_stage_profile["correct"],
                    conventional_stage_profile["crc_failures"],
                    conventional_stage_profile["payload_mismatches"],
                )
            else:
                conventional = run_conventional()
            conventional_done_ns = time.perf_counter_ns()
            owner.wait_backward(sequence, 0.1)
            marker = cp.asnumpy(backward)
            completed_ns = time.perf_counter_ns()
            records.append({
                "iteration": iteration,
                "sequence": sequence,
                "raw_copy_gpu_ms": raw_copy_gpu_ms,
                "conventional_gpu_ms": conventional[0],
                "conventional_correct": bool(conventional[1]),
                "conventional_done_ms": (conventional_done_ns - release_ns) / 1e6,
                "neural_correct": bool(marker[0] == 1.0),
                "neural_crc_failures": int(marker[1]),
                "neural_payload_mismatches": int(marker[2]),
                "remote_neural_gpu_ms": float(marker[3]),
                "parallel_pair_wall_ms": (completed_ns - release_ns) / 1e6,
                "conventional_stage_profile": conventional_stage_profile,
            })
    finally:
        try:
            owner.publish_forward(TERMINATE_SEQ)
            time.sleep(0.2)
        finally:
            owner.close()

    pair = [x["parallel_pair_wall_ms"] for x in records]
    result = {
        "schema": "softwall-c163-raw-p2p-controller-v1",
        "analysis_role": (
            "Optimistic two-GPU raw-IQ speculative diagnostic against testMAC's "
            "threshold; not a production qualification or WCET result."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "deadline_ms": args.deadline_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "warmup_correct": warmup_correct,
        "cpu_affinity": cpu_affinity,
        "profile_conventional": args.profile_conventional,
        "conventional_mode": args.conventional_mode,
        "mechanism": (
            "GPU0 copies only raw IQ and immediately runs conventional. GPU1 performs "
            "NeuralRx channel estimation, TensorRT, LDPC, and CRC, returning a small result."
        ),
        "raw_copy_gpu_ms": summary([x["raw_copy_gpu_ms"] for x in records]),
        "conventional_gpu_ms": summary([x["conventional_gpu_ms"] for x in records]),
        "conventional_done_ms": summary([x["conventional_done_ms"] for x in records]),
        "remote_neural_gpu_ms": summary([x["remote_neural_gpu_ms"] for x in records]),
        "parallel_pair_wall_ms": summary(pair),
        "deadline_counts": {
            "parallel_pair_wall_le_deadline": sum(x <= args.deadline_ms for x in pair),
            "parallel_pair_wall_gt_deadline": sum(x > args.deadline_ms for x in pair),
        },
        "correctness": {
            "neural_correct": sum(x["neural_correct"] for x in records),
            "conventional_correct": sum(x["conventional_correct"] for x in records),
        },
        "records": records,
    }
    atomic_json(args.output, result)
    print(json.dumps({k: result[k] for k in (
        "raw_copy_gpu_ms", "parallel_pair_wall_ms", "deadline_counts", "correctness"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
