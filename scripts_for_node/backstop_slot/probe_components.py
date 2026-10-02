#!/usr/bin/env python3
"""Isolated decode success and latency of each receiver path on one GPU."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import ConvPath, NrxPath
from summarize_slot import dist


def timed(path, ring, repeats):
    wall, passes, payload_ok = [], [], []
    for repeat in range(repeats):
        for index in range(ring.ring):
            begin = time.perf_counter_ns()
            ok, payload = path.run(ring.views[index], ring.slots[index])
            wall.append((time.perf_counter_ns() - begin) / 1e6)
            if repeat == 0:
                passes.append(bool(ok))
                payload_ok.append(bool(ok and np.array_equal(
                    payload[:ring.tb_bytes], ring.blocks[index])))
    return wall, passes, payload_ok


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engines", nargs="+", required=True)
    parser.add_argument("--ring", type=int, default=256)
    parser.add_argument("--repeats", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cp.cuda.Device(0).use()
    stream = cp.cuda.Stream(non_blocking=True)
    result = {"schema": "backstop-slot-components-v1", "host": platform.node(),
              "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
              "gpu": cp.cuda.runtime.getDeviceProperties(0)["name"].decode()}
    for profile in ("weak_rank1", "strong_rank2"):
        ring = CellRing(args.dataset, profile, args.ring, 0)
        conv = ConvPath(ring.profile, ring.tb_bytes, stream)
        timed(conv, ring, 1)
        wall, passes, payload = timed(conv, ring, args.repeats)
        entry = {
            "tb_bytes": ring.tb_bytes,
            "conv_crc_pass": sum(passes), "conv_payload_ok": sum(payload),
            "trials": len(passes), "conv_wall_ms": dist(wall),
            "conv_pass_per_slot": passes,
        }
        if ring.profile.neural_eligible:
            for engine in args.engines:
                nrx = NrxPath(engine, ring.profile, ring.tb_bytes, stream)
                timed(nrx, ring, 1)
                wall, n_passes, n_payload = timed(nrx, ring, args.repeats)
                name = Path(engine).stem
                entry[name] = {
                    "nrx_crc_pass": sum(n_passes), "nrx_payload_ok": sum(n_payload),
                    "nrx_wall_ms": dist(wall),
                    "nrx_only": sum((not c) and n for c, n in zip(passes, n_passes)),
                    "conv_only": sum(c and (not n) for c, n in zip(passes, n_passes)),
                    "both": sum(c and n for c, n in zip(passes, n_passes)),
                    "neither": sum((not c) and (not n) for c, n in zip(passes, n_passes)),
                }
                del nrx
        entry["esno_db"] = ring.esno_db
        entry["gpu_mem_used_gib"] = (
            cp.cuda.runtime.memGetInfo()[1] - cp.cuda.runtime.memGetInfo()[0]
        ) / 2**30
        result[profile] = entry
        del conv
        ring.close()
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    brief = {p: {k: v for k, v in result[p].items()
                 if k not in ("conv_pass_per_slot", "esno_db")}
             for p in ("weak_rank1", "strong_rank2")}
    print(json.dumps(brief, indent=1), flush=True)


if __name__ == "__main__":
    main()
