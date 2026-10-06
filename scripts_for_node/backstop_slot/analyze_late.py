#!/usr/bin/env python3
"""TBs past the L1 deadline: where the time went.

usage: analyze_late.py JOB[,JOB] TAG CELLS POLICY ...      (POLICY n = the run without AI)

L1 latency of a TB = wait (samples arrive -> the conventional receiver starts on them) + decode (start ->
result).  Per policy (all seeds, the run of the first job listed per seed): the late TBs, how many of them
waited more than 0.5 ms before the decode started (the receiver process was still busy with its previous slot
or was not scheduled: a cause on the host side or a late previous slot), their decode time, the number of
slots with a late TB, and what ran on the GPU of the cell during the decode (a neural receiver, an AI piece).
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")


def overlap(spans: np.ndarray, a: int, b: int) -> float:
    """Share of [a, b] covered by the spans (sorted by start, not overlapping one another much)."""
    if not len(spans):
        return 0.0
    lo = np.searchsorted(spans[:, 1], a, side="right")
    hi = np.searchsorted(spans[:, 0], b, side="left")
    part = spans[lo:hi]
    if not len(part):
        return 0.0
    return float((np.minimum(part[:, 1], b) - np.maximum(part[:, 0], a)).clip(min=0).sum()) / max(1, b - a)


def one(path: Path) -> dict:
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    period = int(float(config["period_ms"]) * 1e6)
    skip, gpus = int(config.get("skip_periods", 20)), int(config["num_gpus"])
    skip = max(skip, int(float(os.environ.get("WARMUP_S", "0")) * 1e9 / period))      # WARMUP_S=5: leave out the first 5 s
    deadline = float(config["deadline_ms"]) * 1e6
    nrx, ai = {}, {}
    for g in range(gpus):
        nrx[g] = np.array(sorted((r[2], r[3]) for r in json.loads((work / f"lane{g}.json").read_text())["records"]), dtype=np.int64).reshape(-1, 2)
        pieces = []
        for file in sorted(work.glob(f"ai{g}.json")) + sorted(work.glob(f"ai{g}s*.json")):
            pieces += [(p[0], p[1]) for p in json.loads(file.read_text())["pieces"]]
        ai[g] = np.array(sorted(pieces), dtype=np.int64).reshape(-1, 2)
    out = {"tbs": 0, "late": 0, "wait_late": 0, "slots": set(), "wait": [], "decode": [], "with_nrx": 0, "with_ai": 0,
           "all_wait_p999": [], "all_decode_p999": []}
    waits, decodes = [], []
    for cell in config["cells"]:
        g, ues = int(cell["gpu"]), int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{int(cell['cell'])}.json").read_text())["records"]:
            if r[0] < skip:
                continue
            out["tbs"] += ues
            waits.append((r[3] - r[2]) / 1e6)
            decodes.append((r[4] - r[3]) / 1e6)
            if r[4] - r[2] > deadline:
                out["late"] += ues
                out["wait_late"] += ues * int(r[3] - r[2] > 500_000)
                out["slots"].add((g, r[0]))
                out["wait"].append((r[3] - r[2]) / 1e6)
                out["decode"].append((r[4] - r[3]) / 1e6)
                out["with_nrx"] += ues * int(overlap(nrx[g], r[3], r[4]) > 0.3)
                out["with_ai"] += ues * int(overlap(ai[g], r[3], r[4]) > 0.3)
    out["wait_p999"], out["decode_p999"] = float(np.percentile(waits, 99.9)), float(np.percentile(decodes, 99.9))
    out["decode_p50"] = float(np.percentile(decodes, 50))
    return out


def main() -> None:
    jobs, tag, cells, policies = sys.argv[1].split(","), sys.argv[2], sys.argv[3], sys.argv[4:]
    rate = os.environ.get("RATE", "")
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(?:n|r{rate or '[0-9]+'}([a-z][a-z0-9]*))_c{cells}_")
    paths: dict[str, dict[int, Path]] = {}
    for job in jobs:
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if m:
                paths.setdefault(m.group(2) or "n", {}).setdefault(int(m.group(1)), path)
    print("| policy | seeds | late TBs / TBs | slots with a late TB | late TBs that waited > 0.5 ms | wait of the late TBs p50 (ms) | "
          "decode of the late TBs p50 (ms) | late TBs decoded next to a NeuralRx | next to AI | decode, all TBs p50 / p99.9 (ms) | wait, all TBs p99.9 (ms) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for policy in ["n"] + [p for p in policies if p != "n"]:
        rows = [one(path) for _, path in sorted(paths.get(policy, {}).items())]
        if not rows:
            continue
        late, tbs = sum(r["late"] for r in rows), sum(r["tbs"] for r in rows)
        wait = [v for r in rows for v in r["wait"]]
        decode = [v for r in rows for v in r["decode"]]
        share = lambda key: f"{100.0 * sum(r[key] for r in rows) / late:.0f}%" if late else "–"
        print(f"| {policy} | {len(rows)} | {late} / {tbs} ({100.0 * late / tbs:.4f}%) | {sum(len(r['slots']) for r in rows)} | {share('wait_late')} | "
              f"{np.median(wait) if wait else float('nan'):.2f} | {np.median(decode) if decode else float('nan'):.2f} | {share('with_nrx')} | {share('with_ai')} | "
              f"{np.mean([r['decode_p50'] for r in rows]):.2f} / {np.mean([r['decode_p999'] for r in rows]):.2f} | {np.mean([r['wait_p999'] for r in rows]):.2f} |")


if __name__ == "__main__":
    main()
