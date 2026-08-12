#!/usr/bin/env python3
"""Bidirectional same-GPU MIG P2P payload integrity and latency gate."""

import argparse
import json
import time

import cupy as cp
import numpy as np

from p2p_overlap_bench import BWD_BYTES, FWD_BYTES, enable_peer_access


def run_direction(source_device, destination_device, size, iterations, warmup):
    if size % 4:
        raise ValueError("payload size must be divisible by four")
    count = size // 4
    with cp.cuda.Device(source_device):
        source = cp.empty(count, dtype=cp.uint32)
        copy_stream = cp.cuda.Stream(non_blocking=True)
    with cp.cuda.Device(destination_device):
        destination = cp.empty(count, dtype=cp.uint32)

    def copy_once(sequence, verify):
        with cp.cuda.Device(source_device):
            source[:] = cp.arange(count, dtype=cp.uint32) + np.uint32(sequence)
            cp.cuda.get_current_stream().synchronize()
            started_ns = time.perf_counter_ns()
            cp.cuda.runtime.memcpyPeerAsync(
                destination.data.ptr,
                destination_device,
                source.data.ptr,
                source_device,
                size,
                copy_stream.ptr,
            )
            copy_stream.synchronize()
            elapsed_us = (time.perf_counter_ns() - started_ns) / 1e3
        if verify:
            with cp.cuda.Device(destination_device):
                first = int(destination[0].item())
                last = int(destination[-1].item())
                checksum = int(cp.sum(destination, dtype=cp.uint64).item())
            expected_first = sequence
            expected_last = count - 1 + sequence
            expected_checksum = count * (count - 1) // 2 + count * sequence
            if (first, last, checksum) != (
                expected_first,
                expected_last,
                expected_checksum,
            ):
                raise RuntimeError(
                    f"integrity mismatch seq={sequence}: "
                    f"got={(first, last, checksum)} "
                    f"expected={(expected_first, expected_last, expected_checksum)}"
                )
            return {
                "sequence": sequence,
                "latency_us": elapsed_us,
                "first": first,
                "last": last,
                "checksum": checksum,
                "verified": True,
            }
        return None

    for sequence in range(1, warmup + 1):
        copy_once(sequence, verify=False)
    records = [
        copy_once(sequence, verify=True)
        for sequence in range(1, iterations + 1)
    ]
    values = np.asarray([record["latency_us"] for record in records])
    return {
        "source_device": source_device,
        "destination_device": destination_device,
        "payload_bytes": size,
        "iterations": iterations,
        "warmup": warmup,
        "latency_us": {
            "mean": float(values.mean()),
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "p99": float(np.percentile(values, 99)),
            "min": float(values.min()),
            "max": float(values.max()),
        },
        "records": records,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", default="p2p_copy_gate.json")
    args = parser.parse_args()
    if cp.cuda.runtime.getDeviceCount() != 2:
        raise RuntimeError("exactly two visible MIG CUDA devices are required")
    if args.iterations <= 0 or args.warmup < 0:
        parser.error("iterations must be positive and warmup non-negative")

    can_access = {
        "0_to_1": int(cp.cuda.runtime.deviceCanAccessPeer(0, 1)),
        "1_to_0": int(cp.cuda.runtime.deviceCanAccessPeer(1, 0)),
    }
    if set(can_access.values()) != {1}:
        raise RuntimeError(f"MIG P2P access unavailable: {can_access}")
    enable_peer_access(0, 1)

    result = {
        "visible_cuda_devices": 2,
        "can_access_peer": can_access,
        "directions": [
            run_direction(0, 1, FWD_BYTES, args.iterations, args.warmup),
            run_direction(1, 0, BWD_BYTES, args.iterations, args.warmup),
        ],
    }
    with open(args.output, "w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2)
    for direction in result["directions"]:
        stats = direction["latency_us"]
        print(
            f"[P2P-GATE] {direction['source_device']}→{direction['destination_device']} "
            f"bytes={direction['payload_bytes']} mean={stats['mean']:.2f}us "
            f"p99={stats['p99']:.2f}us verified={args.iterations}/{args.iterations}",
            flush=True,
        )
    print(f"[P2P-GATE] OK output={args.output}", flush=True)


if __name__ == "__main__":
    main()
