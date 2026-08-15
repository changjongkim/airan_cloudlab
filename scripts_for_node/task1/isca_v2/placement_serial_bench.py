#!/usr/bin/env python3
"""Same-boundary in-process MIG-local and cross-MIG P2P benchmark."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import cupy as cp
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from nrx_trt_direct import DirectNrx
from p2p_overlap_bench import (
    BWD_BYTES,
    CE_ELEMS,
    CE_SHAPE,
    FWD_BYTES,
    FWD_ELEMS,
    L1Ops,
    LLR_ELEMS,
    LLR_SHAPE,
    P2PCopier,
    RX_ELEMS,
    RX_SHAPE,
    enable_peer_access,
    flat_section,
)


def stats(values: list[float]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "n": int(array.size), "mean": float(array.mean()),
        "p50": float(np.percentile(array, 50)),
        "p95": float(np.percentile(array, 95)),
        "p99": float(np.percentile(array, 99)),
        "p99_9": float(np.percentile(array, 99.9)),
        "min": float(array.min()), "max": float(array.max()),
    }


def wait_until(target_ns: int) -> None:
    while True:
        remaining = target_ns - time.perf_counter_ns()
        if remaining <= 0:
            return
        if remaining > 200_000:
            time.sleep((remaining - 100_000) / 1e9)


def bind_nrx(runtime: DirectNrx, forward: cp.ndarray, backward: cp.ndarray) -> None:
    offset = 0
    for name, count, shape in (
        ("rx_slot_real", RX_ELEMS, RX_SHAPE),
        ("rx_slot_imag", RX_ELEMS, RX_SHAPE),
        ("h_hat_real", CE_ELEMS, CE_SHAPE),
        ("h_hat_imag", CE_ELEMS, CE_SHAPE),
    ):
        runtime.bind_tensor(name, flat_section(forward, offset, count, shape))
        offset += count
    runtime.bind_tensor(
        "output_1", backward.reshape((1,) + LLR_SHAPE, order="C")
    )
    runtime.capture_graph()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("standalone", "mig_local", "p2p"), required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--arrival-rate", type=float)
    parser.add_argument("--duration", type=float)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--trial", type=int, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or args.warmup < 0 or args.trial <= 0:
        parser.error("iterations/trial positive and warmup non-negative")
    if (args.arrival_rate is None) != (args.duration is None):
        parser.error("arrival-rate and duration must be provided together")
    if args.arrival_rate is not None and (
        args.arrival_rate <= 0 or args.duration <= 0
    ):
        parser.error("arrival-rate and duration must be positive")
    expected_devices = 2 if args.mode == "p2p" else 1
    if cp.cuda.runtime.getDeviceCount() != expected_devices:
        raise RuntimeError(f"mode {args.mode} requires {expected_devices} visible devices")

    l1 = L1Ops(0)
    with cp.cuda.Device(0):
        l1_fwd = cp.empty(FWD_ELEMS, dtype=cp.float32)
        l1_bwd = cp.empty(LLR_ELEMS, dtype=cp.float32)
    runtime = None
    fwd_copier = None
    bwd_copier = None
    if args.mode != "standalone":
        nrx_device = 1 if args.mode == "p2p" else 0
        if args.mode == "p2p":
            enable_peer_access(0, 1)
            with cp.cuda.Device(1):
                nrx_fwd = cp.empty(FWD_ELEMS, dtype=cp.float32)
                nrx_bwd = cp.empty(LLR_ELEMS, dtype=cp.float32)
            fwd_copier = P2PCopier(0, 1)
            bwd_copier = P2PCopier(1, 0)
        else:
            nrx_fwd, nrx_bwd = l1_fwd, l1_bwd
        with cp.cuda.Device(nrx_device):
            runtime = DirectNrx(args.engine)
            bind_nrx(runtime, nrx_fwd, nrx_bwd)
            for _ in range(args.warmup):
                runtime.launch(use_graph=True)
            runtime.stream.synchronize()
    fixed_llr = cp.zeros_like(l1_bwd)

    def one_slot() -> dict:
        started = time.perf_counter_ns()
        front_ms = l1.front(l1_fwd)
        fwd_us = bwd_us = nrx_ms = 0.0
        if runtime is not None:
            if fwd_copier is not None:
                fwd_us = fwd_copier.copy(nrx_fwd, l1_fwd)
            with cp.cuda.Device(1 if args.mode == "p2p" else 0), runtime.stream:
                begin = cp.cuda.Event(); end = cp.cuda.Event()
                begin.record(); runtime.launch(use_graph=True); end.record(); end.synchronize()
                nrx_ms = float(cp.cuda.get_elapsed_time(begin, end))
            if bwd_copier is not None:
                bwd_us = bwd_copier.copy(l1_bwd, nrx_bwd)
            llrs = l1_bwd
        else:
            llrs = fixed_llr
        back_ms = l1.back(llrs)
        return {
            "l1_front_ms": front_ms, "nrx_ms": nrx_ms,
            "fwd_us": fwd_us, "bwd_us": bwd_us,
            "transport_us": fwd_us + bwd_us, "l1_back_ms": back_ms,
            "l1_active_ms": front_ms + back_ms,
            "e2e_ms": (time.perf_counter_ns() - started) / 1e6,
        }

    for _ in range(args.warmup):
        one_slot()
    iterations = (
        int(round(args.arrival_rate * args.duration))
        if args.arrival_rate is not None else args.iterations
    )
    records = []
    wall_start = time.perf_counter_ns()
    for index in range(iterations):
        target_ns = None
        if args.arrival_rate is not None:
            target_ns = wall_start + round(index * 1e9 / args.arrival_rate)
            wait_until(target_ns)
        record = one_slot()
        completed_ns = time.perf_counter_ns()
        if target_ns is not None:
            record["sojourn_ms"] = (completed_ns - target_ns) / 1e6
        records.append(record)
    wall_s = (time.perf_counter_ns() - wall_start) / 1e9
    keys = records[0].keys()
    result = {
        "schema": "placement-serial-direct-v1",
        "mode": args.mode, "trial": args.trial,
        "iterations": iterations, "warmup": args.warmup,
        "arrival_rate_slots_per_s": args.arrival_rate,
        "duration_target_s": args.duration,
        "fwd_bytes": FWD_BYTES, "bwd_bytes": BWD_BYTES,
        "timing_boundary": "l1_front_start_to_ldpc_crc_complete",
        "nrx_binding": "caller_owned_io_cuda_graph" if runtime else "none",
        "throughput_slots_per_s": iterations / wall_s,
        "metrics": {key: stats([x[key] for x in records]) for key in keys},
        "raw": records, "pass": True,
    }
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(result), encoding="utf-8"); temporary.replace(output)
    print(
        f"[PLACEMENT-SERIAL] PASS mode={args.mode} "
        f"mean={result['metrics']['e2e_ms']['mean']:.3f}ms "
        f"p99={result['metrics']['e2e_ms']['p99']:.3f}ms",
        flush=True,
    )


if __name__ == "__main__":
    main()
