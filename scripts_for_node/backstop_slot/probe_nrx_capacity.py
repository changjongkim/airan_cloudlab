#!/usr/bin/env python3
"""NeuralRx capacity probe.

``profile``: where one NeuralRx TB spends its time (host call vs GPU work).
``worker``: one of K concurrent NeuralRx workers on a GPU; all workers start at
a shared time and decode TBs back to back for a fixed duration, so the sum of
their counts is the GPU's NeuralRx throughput with K lanes.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import NrxPath

DATA = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/run_state/backstop_slot/dataset_v1")
ENGINE = "/softwall_runtime/engines/neural_rx_fp16_full.trt"


def profile(args) -> dict:
    ring = CellRing(DATA, "weak_rank1", 64, 0)
    stream = cp.cuda.Stream(non_blocking=True)
    nrx = NrxPath(ENGINE, ring.profile, ring.tb_bytes, stream)
    for i in range(20):
        nrx.run(ring.views[i % 64], ring.slots[i % 64])
    stages = {k: [] for k in ("ce", "trt", "derate", "decode", "crc", "total", "gpu_total")}
    start_sym = ring.profile.start_sym
    for i in range(args.count):
        view, slot = ring.views[i % 64], ring.slots[i % 64]
        e0, e1 = cp.cuda.Event(), cp.cuda.Event()
        t0 = time.perf_counter_ns()
        e0.record(stream)
        channel = nrx.estimator.estimate(rx_slot=view, slot=slot, pusch_configs=nrx.configs)
        stream.synchronize(); t1 = time.perf_counter_ns()
        with stream:
            window = view[None, :, start_sym:start_sym + 12, :]
            est = cp.asarray(channel[0])
            est = cp.transpose(est, (0, 3, 1, 2)).reshape(est.shape[0] * est.shape[3], est.shape[1], est.shape[2])[None, ...]
            cp.copyto(nrx.engine.inputs["rx_slot_real"], window.real)
            cp.copyto(nrx.engine.inputs["rx_slot_imag"], window.imag)
            cp.copyto(nrx.engine.inputs["h_hat_real"], est.real)
            cp.copyto(nrx.engine.inputs["h_hat_imag"], est.imag)
            outputs = nrx.engine.launch(use_graph=True)
            llrs = cp.take(outputs["output_1"][0, ...], nrx.data_symbols, axis=3)
        stream.synchronize(); t2 = time.perf_counter_ns()
        with stream:
            coded = nrx.derate.derate_match(input_llrs=[llrs], pusch_configs=nrx.configs)
        stream.synchronize(); t3 = time.perf_counter_ns()
        with stream:
            blocks = nrx.decoder.decode(input_llrs=coded, pusch_configs=nrx.configs)
        stream.synchronize(); t4 = time.perf_counter_ns()
        with stream:
            tbs, crcs = nrx.crc.check_crc(input_bits=blocks, pusch_configs=nrx.configs)
        e1.record(stream); stream.synchronize(); t5 = time.perf_counter_ns()
        for k, a, b in (("ce", t0, t1), ("trt", t1, t2), ("derate", t2, t3), ("decode", t3, t4), ("crc", t4, t5), ("total", t0, t5)):
            stages[k].append((b - a) / 1e6)
        stages["gpu_total"].append(cp.cuda.get_elapsed_time(e0, e1))
    # GPU-only time with no host synchronization inside the chain
    e0, e1 = cp.cuda.Event(), cp.cuda.Event()
    walls = []
    for i in range(args.count):
        t0 = time.perf_counter_ns()
        nrx.run(ring.views[i % 64], ring.slots[i % 64])
        walls.append((time.perf_counter_ns() - t0) / 1e6)
    return {
        "stage_ms_p50": {k: float(np.percentile(v, 50)) for k, v in stages.items()},
        "stage_ms_p99": {k: float(np.percentile(v, 99)) for k, v in stages.items()},
        "run_wall_ms_p50": float(np.percentile(walls, 50)),
    }


def worker(args) -> dict:
    ring = CellRing(DATA, "weak_rank1", 64, 17 * args.index)
    stream = cp.cuda.Stream(non_blocking=True)
    nrx = NrxPath(ENGINE, ring.profile, ring.tb_bytes, stream)
    for i in range(20):
        nrx.run(ring.views[i % 64], ring.slots[i % 64])
    ready = Path(args.sync) / f"ready_{os.getpid()}"
    ready.write_text("1")
    go = Path(args.sync) / "go"
    while not go.is_file():
        time.sleep(0.001)
    start_ns = int(go.read_text())
    while time.monotonic_ns() < start_ns:
        pass
    end_ns = start_ns + int(args.seconds * 1e9)
    lat, count, i = [], 0, 0
    while True:
        t0 = time.monotonic_ns()
        if t0 >= end_ns:
            break
        nrx.run(ring.views[i % 64], ring.slots[i % 64])
        lat.append((time.monotonic_ns() - t0) / 1e6)
        count += 1
        i += 1
    return {"index": args.index, "count": count, "seconds": args.seconds,
            "latency_ms_p50": float(np.percentile(lat, 50)),
            "latency_ms_p99": float(np.percentile(lat, 99))}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("profile", "worker"))
    parser.add_argument("--count", type=int, default=300)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--sync", default="")
    parser.add_argument("--seconds", type=float, default=5.0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cp.cuda.Device(0).use()
    result = profile(args) if args.mode == "profile" else worker(args)
    args.output.write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
