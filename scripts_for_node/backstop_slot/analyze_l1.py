#!/usr/bin/env python3
"""L1 misses by what else ran on the GPU while the conventional receiver decoded.

usage: analyze_l1.py TAG CELLS JOB[,JOB] POLICY ...        (POLICY n = the run without AI)
Per TB: in the 2.5 ms after its samples arrived, did a neural receiver run on the GPU of its
cell (more than 30% of the time), did an AI piece run there?  Per policy (all seeds of the
jobs): share of the TBs in each case and the share of them whose conventional result missed
the L1 deadline.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
STEP = 100_000


def grid(spans, epoch: int, n: int) -> np.ndarray:
    g = np.zeros(n + 1, dtype=np.int32)
    for a, b in spans:
        i, j = max(0, (a - epoch) // STEP), min(n, (b - epoch) // STEP)
        if j > i:
            g[i] += 1
            g[j] -= 1
    return np.cumsum(g[:-1]) > 0


def one(path: Path) -> dict:
    config = json.loads(path.read_text())["config"]
    work = Path(str(path)[:-5] + "_work")
    deadline = float(config["deadline_ms"]) * 1e6
    skip, gpus = int(config.get("skip_periods", 20)), int(config["num_gpus"])
    period = int(float(config["period_ms"]) * 1e6)
    first = json.loads((work / "conv0.json").read_text())["records"][0]
    epoch = first[2] - first[0] * period
    n = (int(config["periods"]) + 4) * period // STEP
    nrx, ai = {}, {}
    for g in range(gpus):
        nrx[g] = grid([(r[2], r[3]) for r in json.loads((work / f"lane{g}.json").read_text())["records"]], epoch, n)
        pieces = work / f"ai{g}.json"
        ai[g] = grid([(p[0], p[1]) for p in json.loads(pieces.read_text())["pieces"]], epoch, n) if pieces.exists() \
            else np.zeros(n, dtype=bool)
    out: dict[str, list[int]] = {}
    for cell in config["cells"]:
        g, ues = int(cell["gpu"]), int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{int(cell['cell'])}.json").read_text())["records"]:
            if r[0] < skip:
                continue
            i = (r[2] - epoch) // STEP
            with_nrx, with_ai = nrx[g][i:i + 25].mean() > 0.3, ai[g][i:i + 25].mean() > 0.3
            key = "nrx+ai" if with_nrx and with_ai else "nrx" if with_nrx else "ai" if with_ai else "neither"
            row = out.setdefault(key, [0, 0])
            row[0] += ues
            row[1] += ues * int(r[4] - r[2] > deadline)
    return out


def main() -> None:
    tag, cells, jobs = sys.argv[1], sys.argv[2], sys.argv[3].split(",")
    print("| policy | runs | L1 late, all TBs | neither: share of TBs, late | AI only | NeuralRx only | NeuralRx and AI |")
    print("|---|---|---|---|---|---|---|")
    for policy in sys.argv[4:]:
        total: dict[str, list[int]] = {}
        runs = 0
        for job in jobs:
            name = f"{tag}[0-9]n_c{cells}_*_j{job}.json" if policy == "n" else f"{tag}[0-9]r*[0-9]{policy}_c{cells}_*_j{job}.json"
            for path in sorted(RAW.glob(name)):
                runs += 1
                for key, (tbs, late) in one(path).items():
                    row = total.setdefault(key, [0, 0])
                    row[0] += tbs
                    row[1] += late
        if not runs:
            continue
        tbs = sum(v[0] for v in total.values())
        late = sum(v[1] for v in total.values())
        cell = lambda key: (f"{100 * total[key][0] / tbs:.0f}%, {100 * total[key][1] / max(1, total[key][0]):.3f}%"
                            if key in total else "–")
        print(f"| {policy} | {runs} | {100 * late / tbs:.3f}% | {cell('neither')} | {cell('ai')} | {cell('nrx')} | {cell('nrx+ai')} |")


if __name__ == "__main__":
    main()
