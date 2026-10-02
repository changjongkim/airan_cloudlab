#!/usr/bin/env python3
"""Frequency-stacked NeuralRx batching: S TBs side by side in one inference.

The public model is convolutional over subcarriers and its subcarrier axis is
dynamic, so S full-band TBs can be concatenated along frequency (with their LS
pilot estimates concatenated the same way).  We check that every TB still
decodes as it does alone, and time the TensorRT call against S=1.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np
import tensorrt as trt

from cell_ring import CellRing
from slot_radio import NrxPath

DATA = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/run_state/backstop_slot/dataset_v1")
SC, PILOTS = 3276, 4914


class Engine:
    def __init__(self, path: str, stream: cp.cuda.Stream) -> None:
        runtime = trt.Runtime(trt.Logger(trt.Logger.ERROR))
        self.engine = runtime.deserialize_cuda_engine(open(path, "rb").read())
        self.context = self.engine.create_execution_context()
        self.stream = stream
        self.tensors = {}
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            shape = tuple(self.engine.get_tensor_shape(name))
            dtype = cp.dtype(trt.nptype(self.engine.get_tensor_dtype(name)))
            self.tensors[name] = cp.zeros(shape, dtype=dtype)
            self.context.set_tensor_address(name, int(self.tensors[name].data.ptr))

    def run(self) -> None:
        self.context.execute_async_v3(int(self.stream.ptr))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engines", nargs="+", required=True, help="S=path")
    parser.add_argument("--tbs", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cp.cuda.Device(0).use()
    stream = cp.cuda.Stream(non_blocking=True)
    ring = CellRing(DATA, "weak_rank1", args.tbs, 0)
    base = NrxPath("/softwall_runtime/engines/neural_rx_fp16_full.trt", ring.profile, ring.tb_bytes, stream)
    start = ring.profile.start_sym
    rel = np.asarray(ring.profile.dmrs_positions) - start

    # Per-TB LS estimates and reference single-TB outcomes.
    windows, estimates, single_ok = [], [], []
    for i in range(args.tbs):
        view, slot = ring.views[i], ring.slots[i]
        ok, _ = base.run(view, slot)
        single_ok.append(bool(ok))
        channel = base.estimator.estimate(rx_slot=view, slot=slot, pusch_configs=base.configs)
        with stream:
            est = cp.asarray(channel[0])
            est = cp.transpose(est, (0, 3, 1, 2)).reshape(est.shape[0] * est.shape[3], est.shape[1], est.shape[2])
            estimates.append(est.copy())
            windows.append(view[:, start:start + 12, :].copy())
    stream.synchronize()

    def decode(llrs: cp.ndarray, index: int) -> bool:
        with stream:
            coded = base.derate.derate_match(input_llrs=[llrs], pusch_configs=base.configs)
            blocks = base.decoder.decode(input_llrs=coded, pusch_configs=base.configs)
            tbs, crcs = base.crc.check_crc(input_bits=blocks, pusch_configs=base.configs)
        crc = cp.asnumpy(crcs[0]) if isinstance(crcs[0], cp.ndarray) else np.asarray(crcs[0])
        return int(crc.reshape(-1)[0]) == 0

    result = {"single_pass": sum(single_ok), "tbs": args.tbs, "stacks": {}}
    for spec in args.engines:
        s, path = spec.split("=")
        s = int(s)
        eng = Engine(path, stream)
        t = eng.tensors
        with stream:
            t["active_dmrs_ports"].fill(1)
            t["dmrs_ofdm_pos"][:] = cp.asarray(rel[None, :], dtype=cp.int32)
            t["dmrs_subcarrier_pos"][:] = cp.asarray([[0, 2, 4, 6, 8, 10]], dtype=cp.int32)
        stacked_ok = []
        for first in range(0, args.tbs - s + 1, s):
            with stream:
                for j in range(s):
                    w, e = windows[first + j], estimates[first + j]
                    t["rx_slot_real"][0, j * SC:(j + 1) * SC] = w.real
                    t["rx_slot_imag"][0, j * SC:(j + 1) * SC] = w.imag
                    t["h_hat_real"][0, j * PILOTS:(j + 1) * PILOTS] = e.real
                    t["h_hat_imag"][0, j * PILOTS:(j + 1) * PILOTS] = e.imag
            eng.run()
            out = t["output_1"][0]            # (bits, layers, subcarriers, symbols)
            for j in range(s):
                llrs = cp.take(out[:, :, j * SC:(j + 1) * SC, :], base.data_symbols, axis=3)
                stacked_ok.append(decode(cp.ascontiguousarray(llrs), first + j))
        agree = sum(a == b for a, b in zip(stacked_ok, single_ok))
        # TensorRT time alone
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        times = []
        for _ in range(200):
            e0.record(stream); eng.run(); e1.record(stream); e1.synchronize()
            times.append(cp.cuda.get_elapsed_time(e0, e1))
        result["stacks"][str(s)] = {
            "pass": sum(stacked_ok), "decoded": len(stacked_ok),
            "agree_with_single": agree,
            "single_pass_same_tbs": sum(single_ok[:len(stacked_ok)]),
            "trt_ms_p50": float(np.percentile(times, 50)),
            "trt_ms_per_tb": float(np.percentile(times, 50)) / s,
        }
        print(s, json.dumps(result["stacks"][str(s)]), flush=True)
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
