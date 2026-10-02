#!/usr/bin/env python3
"""Probe: decode several cells in one cuPHY PUSCH pipeline call (a cell group).

The slot-scale runtime uses one process and one pipeline per cell.  Aerial's cuPHY can take
a group of cells in a single setup/run; this measures the wall time of one call for N cells
on one GPU and checks the CRC results against the single-cell pipeline on the same slots.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from cellgroup_radio import CellGroupRx
from slot_radio import pusch_configs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--counts", default="1,2,4,6,8,10,12,16,24,32")
    parser.add_argument("--weak-every", type=int, default=2, help="cell i is weak when i %% this == 0")
    parser.add_argument("--ring", type=int, default=32)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    cp.cuda.Device(args.gpu).use()
    counts = [int(c) for c in args.counts.split(",")]
    most = max(counts)
    rings = [CellRing(args.dataset, "weak_rank1" if i % args.weak_every == 0 else "strong_rank2",
                      args.ring, 97 * i) for i in range(most)]
    configs = [pusch_configs(r.profile, r.tb_bytes)[0] for r in rings]
    stream = cp.cuda.Stream(non_blocking=True)

    # Reference: every (cell, ring index) decoded alone with a one-cell pipeline.
    single = CellGroupRx(1, stream)
    reference, single_ms = {}, []
    for i, ring in enumerate(rings):
        single.harq = {}
        for j in range(args.ring):
            start = time.perf_counter_ns()
            crcs, cb_fail = single.run([ring.views[j]], [ring.slots[j]], [configs[i]])
            single_ms.append((time.perf_counter_ns() - start) / 1e6)
            reference[(i, j)] = (crcs[0], cb_fail[0])

    rows = []
    for n in counts:
        rx = CellGroupRx(n, stream)
        times, mismatch, passes, total = [], 0, 0, 0
        for it in range(args.iterations + 20):
            j = it % args.ring
            start = time.perf_counter_ns()
            crcs, cb_fail = rx.run([rings[i].views[j] for i in range(n)],
                                   [rings[i].slots[j] for i in range(n)], configs[:n])
            elapsed = (time.perf_counter_ns() - start) / 1e6
            if it < 20:
                continue
            times.append(elapsed)
            for i in range(n):
                ok = crcs[i]
                total += 1
                passes += ok
                mismatch += (ok, cb_fail[i]) != reference[(i, j)]
        t = np.asarray(times)
        row = {"cells": n, "weak_cells": sum(1 for i in range(n) if i % args.weak_every == 0),
               "p50_ms": float(np.percentile(t, 50)), "p99_ms": float(np.percentile(t, 99)),
               "max_ms": float(t.max()), "per_cell_ms": float(np.percentile(t, 50)) / n,
               "crc_pass": passes, "tbs": total, "mismatch_vs_single_cell": mismatch}
        rows.append(row)
        print(json.dumps(row), flush=True)
        del rx
    result = {"gpu": cp.cuda.runtime.getDeviceProperties(args.gpu)["name"].decode(),
              "single_cell_ms_p50": float(np.percentile(single_ms, 50)), "rows": rows}
    args.output.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
