#!/usr/bin/env python3
"""Two-MIG GPUDirect RDMA payload and integrity test.

Start the consumer first, then the producer with the same tag, payload size,
and iteration count.  Reported producer latency includes the completed GPU
payload RDMA WRITE followed by the completed 8-byte sequence RDMA WRITE.  The
/dev/shm validation ACK is outside that timed region and prevents the producer
from overwriting its payload before the consumer verifies it.
"""

import argparse
import os
import sys
import time
import traceback

import cupy as cp
import numpy as np

from gdr_rdma_channel import GdrRdmaEndpoint


DEFAULT_PAYLOAD_SIZE = 64 * 1024
DEFAULT_ITERATIONS = 10


def positive_int(value):
    parsed = int(value, 0)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def expected_integrity(word_count, sequence):
    words = np.arange(word_count, dtype=np.uint32)
    words ^= np.uint32(sequence)
    return int(words[0]), int(words.astype(np.uint64).sum(dtype=np.uint64))


def run_producer(endpoint, iterations):
    indices = cp.arange(endpoint.word_count, dtype=cp.uint32)
    print(
        f"[prod] endpoint ready session={endpoint.session_id} "
        f"payload={endpoint.payload_size}",
        flush=True,
    )
    for sequence in range(1, iterations + 1):
        endpoint.gpu_array[:] = indices ^ cp.uint32(sequence)
        # NVIDIA's GPUDirect ordering contract requires CUDA work producing the
        # source allocation to complete before the NIC reads it.
        cp.cuda.get_current_stream().synchronize()
        cp.cuda.runtime.deviceSynchronize()

        started_ns = time.perf_counter_ns()
        endpoint.write_payload()
        endpoint.write_sequence(sequence)
        elapsed_us = (time.perf_counter_ns() - started_ns) / 1e3
        print(
            f"[prod] seq={sequence} payload+seq took {elapsed_us:.2f} us",
            flush=True,
        )
        endpoint.wait_for_ack(sequence)
    print("[prod] done", flush=True)


def run_consumer(endpoint, iterations):
    print(
        f"[cons] endpoint ready session={endpoint.session_id} "
        f"payload={endpoint.payload_size}",
        flush=True,
    )
    for sequence in range(1, iterations + 1):
        endpoint.wait_for_sequence(sequence)
        # Marker WRITE follows payload completion on the same RC QP.  Observe
        # it on CPU, establish a CUDA synchronization point, then launch CuPy
        # reads/checksum work against the GPU allocation.
        cp.cuda.runtime.deviceSynchronize()
        first_word = int(endpoint.gpu_array[0].item())
        checksum = int(cp.sum(endpoint.gpu_array, dtype=cp.uint64).item())
        expected_first, expected_checksum = expected_integrity(
            endpoint.word_count, sequence
        )
        if first_word != expected_first or checksum != expected_checksum:
            raise RuntimeError(
                f"integrity mismatch seq={sequence}: "
                f"first={first_word}/{expected_first} "
                f"checksum={checksum}/{expected_checksum}"
            )
        print(
            f"[cons] got seq={sequence} first_word={first_word} "
            f"checksum={checksum} verified=1",
            flush=True,
        )
        endpoint.publish_ack(sequence)
    print("[cons] done", flush=True)


def main():
    env_size = positive_int(
        os.environ.get("GDR_PAYLOAD_SIZE", str(DEFAULT_PAYLOAD_SIZE))
    )
    env_iterations = positive_int(
        os.environ.get("GDR_ITERATIONS", str(DEFAULT_ITERATIONS))
    )
    parser = argparse.ArgumentParser(
        description="RC GPUDirect RDMA test between two selected MIG containers"
    )
    parser.add_argument("role", choices=("prod", "cons"))
    parser.add_argument(
        "payload_size",
        nargs="?",
        default=env_size,
        type=positive_int,
        help=f"payload bytes (default/env: {env_size})",
    )
    parser.add_argument(
        "--iterations",
        type=positive_int,
        default=env_iterations,
        help=f"number of writes (default/env: {env_iterations})",
    )
    parser.add_argument(
        "--tag",
        default=os.environ.get("GDR_RDMA_TAG", "test"),
        help="shared /dev/shm rendezvous tag",
    )
    args = parser.parse_args()

    endpoint = None
    cleanup_errors = []
    try:
        endpoint = GdrRdmaEndpoint(args.payload_size, args.tag, args.role)
        if args.role == "prod":
            run_producer(endpoint, args.iterations)
        else:
            run_consumer(endpoint, args.iterations)
    except BaseException as exc:
        traceback.print_exc()
        print(
            f"GDR_RDMA_TEST_FAIL: {type(exc).__name__}: {exc}",
            file=sys.stderr,
            flush=True,
        )
        return 1
    finally:
        if endpoint is not None:
            cleanup_errors = endpoint.close()

    if cleanup_errors:
        print(
            "GDR_RDMA_TEST_FAIL: cleanup failed: " + "; ".join(cleanup_errors),
            file=sys.stderr,
            flush=True,
        )
        return 1
    print(f"GDR_RDMA_TEST_OK role={args.role}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
