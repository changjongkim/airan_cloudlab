#!/usr/bin/env python3
"""Speculatively overlap local conventional with remote-P2P NeuralRx."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp

from dual_receiver_phy import PairedDualReceiver
from isca_v2.cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from softwall_phy import summary


FWD_ELEMENTS = 2 * (1 * 3276 * 12 * 4) + 2 * (1 * 4914 * 1 * 4)
LLR_ELEMENTS = 2 * 1 * 3276 * 12


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--ipc-dir", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--deadline-ms", type=float, default=4.5)
    parser.add_argument("--endpoint-timeout-ms", type=float, default=100.0)
    parser.add_argument("--seed", type=int, default=20358000)
    args = parser.parse_args()

    cp.cuda.runtime.setDevice(0)
    forward = cp.empty(FWD_ELEMENTS, dtype=cp.float32)
    backward = cp.empty(LLR_ELEMENTS, dtype=cp.float32)
    owner = CudaIpcOwner(args.tag, forward, backward, directory=args.ipc_dir)
    receiver = PairedDualReceiver(
        args.engine, seed=args.seed, enable_local_neural=False
    )
    sequence = 0
    warmup_correct = 0
    records = []
    try:
        owner.wait_ready(120.0)
        for _ in range(args.warmup):
            sequence += 1
            receiver.prepare_neural_ipc(forward)
            owner.publish_forward(sequence)
            conventional = receiver.run_conventional()
            owner.wait_backward(sequence, 2.0)
            neural = receiver.complete_neural_ipc(backward)
            warmup_correct += int(conventional[1] and neural[1])
        if warmup_correct != args.warmup:
            raise RuntimeError(f"warmup correctness {warmup_correct}/{args.warmup}")

        for iteration in range(args.iterations):
            release_ns = time.perf_counter_ns()
            sequence += 1
            front_ms = receiver.prepare_neural_ipc(forward)
            owner.publish_forward(sequence)
            conventional = receiver.run_conventional()
            conventional_done_ns = time.perf_counter_ns()
            owner.wait_backward(sequence, args.endpoint_timeout_ms / 1000.0)
            neural = receiver.complete_neural_ipc(backward)
            completed_ns = time.perf_counter_ns()
            records.append({
                "iteration": iteration,
                "sequence": sequence,
                "front_gpu_ms": front_ms,
                "conventional_gpu_ms": conventional[0],
                "conventional_correct": bool(conventional[1]),
                "conventional_done_ms": (conventional_done_ns - release_ns) / 1e6,
                "neural_post_gpu_ms": neural[0],
                "neural_correct": bool(neural[1]),
                "parallel_pair_wall_ms": (completed_ns - release_ns) / 1e6,
            })
    finally:
        try:
            owner.publish_forward(TERMINATE_SEQ)
            time.sleep(0.2)
        finally:
            owner.close()

    pair = [x["parallel_pair_wall_ms"] for x in records]
    result = {
        "schema": "softwall-c163-p2p-speculative-controller-v1",
        "analysis_role": (
            "Optimistic two-GPU speculative diagnostic against testMAC's threshold; "
            "not a production qualification or WCET result."
        ),
        "host": platform.node(),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "deadline_ms": args.deadline_ms,
        "warmup": args.warmup,
        "iterations": args.iterations,
        "warmup_correct": warmup_correct,
        "mechanism": (
            "GPU0 prepares NeuralRx inputs, GPU1 performs P2P TensorRT while GPU0 "
            "executes conventional, then GPU0 completes NeuralRx postprocessing."
        ),
        "front_gpu_ms": summary([x["front_gpu_ms"] for x in records]),
        "conventional_gpu_ms": summary([x["conventional_gpu_ms"] for x in records]),
        "conventional_done_ms": summary([x["conventional_done_ms"] for x in records]),
        "neural_post_gpu_ms": summary([x["neural_post_gpu_ms"] for x in records]),
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
        "parallel_pair_wall_ms", "deadline_counts", "correctness"
    )}, indent=2), flush=True)


if __name__ == "__main__":
    main()
