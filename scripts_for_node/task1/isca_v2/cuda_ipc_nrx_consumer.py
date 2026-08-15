#!/usr/bin/env python3
"""Direct-bound NRx process for the same-device CUDA-IPC placement gate."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cupy as cp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cuda_ipc_channel import CudaIpcPeer, TERMINATE_SEQ
from nrx_trt_direct import DirectNrx
from p2p_overlap_bench import CE_ELEMS, CE_SHAPE, LLR_SHAPE, RX_ELEMS, RX_SHAPE, flat_section


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True)
    parser.add_argument("--engine", required=True)
    args = parser.parse_args()
    peer = CudaIpcPeer(args.tag, 120.0)
    try:
        forward = peer.forward.view(cp.float32)
        backward = peer.backward.view(cp.float32)
        runtime = DirectNrx(args.engine)
        offset = 0
        for name, count, shape in (
            ("rx_slot_real", RX_ELEMS, RX_SHAPE),
            ("rx_slot_imag", RX_ELEMS, RX_SHAPE),
            ("h_hat_real", CE_ELEMS, CE_SHAPE),
            ("h_hat_imag", CE_ELEMS, CE_SHAPE),
        ):
            runtime.bind_tensor(name, flat_section(forward, offset, count, shape))
            offset += count
        runtime.bind_tensor("output_1", backward.reshape((1,) + LLR_SHAPE, order="C"))
        runtime.capture_graph()
        for _ in range(20): runtime.launch(use_graph=True)
        runtime.stream.synchronize()
        peer.mark_ready()
        print("[CUDA-IPC-NRX] ready", flush=True)
        last = 0
        while True:
            sequence = peer.read_forward()
            if sequence == TERMINATE_SEQ:
                print("[CUDA-IPC-NRX] shutdown", flush=True)
                break
            if sequence <= last:
                time.sleep(0)
                continue
            if sequence != last + 1:
                raise RuntimeError(f"sequence skip expected={last+1} observed={sequence}")
            runtime.launch(use_graph=True)
            runtime.stream.synchronize()
            peer.publish_backward(sequence)
            last = sequence
    finally:
        peer.close()


if __name__ == "__main__":
    main()
