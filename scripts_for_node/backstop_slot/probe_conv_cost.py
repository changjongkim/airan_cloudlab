#!/usr/bin/env python3
"""Cost of more LDPC iterations in the conventional receiver, per profile (one GPU, alone)."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import cupy as cp
import numpy as np

from cell_ring import CellRing
from slot_radio import conv_path, result_masks


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--count", type=int, default=256)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    total = json.loads((args.dataset / f"{args.profile}_meta.json").read_text())["count"]
    count = min(args.count, total)
    ring = CellRing(args.dataset, args.profile, count, 0)
    stream = cp.cuda.Stream(non_blocking=True)
    report = {"profile": args.profile, "slots": count, "tb_bytes": ring.tb_bytes, "ues": ring.profile.num_ue}
    for iterations in (10, 20, 40):
        path = conv_path(ring.profile, ring.tb_bytes, stream, iterations)
        wall, good = [], 0
        for attempt in range(3):
            for i in range(count):
                start = time.perf_counter()
                ok, payload = path.run(ring.views[i], ring.slots[i])
                if attempt:
                    wall.append((time.perf_counter() - start) * 1e3)
                if attempt == 2:
                    good += bin(result_masks(path, ok, payload, ring.blocks[i], ring.tb_bytes)[1]).count("1")
        report[f"it{iterations}"] = {"decoded": good, "ms_p50": float(np.percentile(wall, 50)),
                                     "ms_p99": float(np.percentile(wall, 99))}
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    main()
