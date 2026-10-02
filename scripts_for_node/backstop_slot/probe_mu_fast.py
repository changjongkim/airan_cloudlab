#!/usr/bin/env python3
"""Fast MU-MIMO neural path against the reference path on every slot: same per-UE results?
And the time of each, alone on one GPU."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import nrx_path, result_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    count = json.loads((args.dataset / "nv_mu2_meta.json").read_text())["count"]
    ring = CellRing(args.dataset, "nv_mu2", count, 0)
    stream = cp.cuda.Stream(non_blocking=True)
    report = {"slots": count}
    masks = {}
    for mode in ("reference", "fast"):
        path = nrx_path({"engine": args.engine, "nrx_ldpc_iterations": args.iterations, "nrx_path_mode": mode},
                        ring.profile, ring.tb_bytes, stream)
        wall, good = [], []
        for attempt in range(3):
            for i in range(count):
                start = time.perf_counter()
                ok, payload = path.run(ring.views[i], ring.slots[i])
                if attempt:
                    wall.append((time.perf_counter() - start) * 1e3)
                if attempt == 2:
                    good.append(result_masks(path, ok, payload, ring.blocks[i], ring.tb_bytes))
        masks[mode] = np.asarray(good)
        report[mode] = {"tbs_decoded": int(sum(bin(g).count("1") for _, g in good)),
                        "ms_p50": float(np.percentile(wall, 50)), "ms_p99": float(np.percentile(wall, 99)),
                        "ms_max": float(np.max(wall))}
    report["slots_with_same_result"] = int((masks["reference"] == masks["fast"]).all(axis=1).sum())
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
