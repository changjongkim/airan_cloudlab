#!/usr/bin/env python3
"""L1 process for the same-device CUDA-IPC direct-NRx placement gate."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuda_ipc_channel import CudaIpcOwner, TERMINATE_SEQ
from p2p_overlap_bench import FWD_BYTES, FWD_ELEMS, L1Ops, LLR_ELEMS


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)), "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)), "p99_9": float(np.percentile(array, 99.9)),
        "min": float(array.min()), "max": float(array.max()),
    }


def wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if (args.arrival_rate is None) != (args.duration is None):
        parser.error("arrival-rate and duration must be provided together")
    iterations = (
        int(round(args.arrival_rate * args.duration))
        if args.arrival_rate is not None else args.iterations
    )
    with cp.cuda.Device(0):
        forward = cp.empty(FWD_ELEMS, dtype=cp.float32)
        backward = cp.empty(LLR_ELEMS, dtype=cp.float32)
    owner = CudaIpcOwner(args.tag, forward, backward)
    l1 = L1Ops(0)
    try:
        owner.wait_ready(120.0)
        records = []
        total = args.warmup + iterations
        wall_start = None
        for sequence in range(1, total + 1):
            if sequence == args.warmup + 1:
                wall_start = time.perf_counter_ns()
            target_ns = None
            if sequence > args.warmup and args.arrival_rate is not None:
                measured_index = sequence - args.warmup - 1
                target_ns = wall_start + round(
                    measured_index * 1e9 / args.arrival_rate
                )
                wait_until(target_ns)
            started = time.perf_counter_ns()
            front_ms = l1.front(forward)
            owner.publish_forward(sequence)
            owner.wait_backward(sequence, 600.0)
            back_ms = l1.back(backward)
            if sequence > args.warmup:
                record = {
                    "sequence": sequence - args.warmup,
                    "l1_front_ms": front_ms, "l1_back_ms": back_ms,
                    "l1_active_ms": front_ms + back_ms,
                    "e2e_ms": (time.perf_counter_ns() - started) / 1e6,
                }
                if target_ns is not None:
                    record["sojourn_ms"] = (
                        time.perf_counter_ns() - target_ns
                    ) / 1e6
                records.append(record)
        wall_s = (time.perf_counter_ns() - wall_start) / 1e9
        result = {
            "schema": "cuda-ipc-placement-v1", "variant": args.variant,
            "trial": args.trial, "iterations": iterations,
            "arrival_rate_slots_per_s": args.arrival_rate,
            "duration_target_s": args.duration,
            "fwd_bytes": FWD_BYTES, "bwd_bytes": backward.nbytes,
            "timing_boundary": "l1_front_start_to_ldpc_crc_complete",
            "payload": "CUDA IPC GPU memory; CPU 8-byte doorbell only",
            "throughput_slots_per_s": iterations / wall_s,
            "metrics": {
                key: stats([x[key] for x in records])
                for key in records[0] if key != "sequence"
            },
            "raw": records, "pass": True,
        }
        output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
        temporary = output.with_suffix(output.suffix + ".tmp")
        temporary.write_text(json.dumps(result), encoding="utf-8"); temporary.replace(output)
        print(
            f"[CUDA-IPC-L1] PASS variant={args.variant} "
            f"mean={result['metrics']['e2e_ms']['mean']:.3f}ms",
            flush=True,
        )
    finally:
        try:
            owner.publish_forward(TERMINATE_SEQ)
        finally:
            time.sleep(0.1)
            owner.close()


if __name__ == "__main__":
    main()
