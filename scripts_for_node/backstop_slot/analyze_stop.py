#!/usr/bin/env python3
"""How long AI keeps running after the rule says it must stop.

usage: analyze_stop.py JOB TAG CELLS POLICY[:RESERVE] ...
Per policy (all seeds of the tag): on a 50 us grid, the time in which a GPU runs a neural
receiver and the rule forbids AI there (fewer than RESERVE neural receivers free; RESERVE =
number of neural receivers means "never next to one"), and the share of that time in which an
AI piece is still on that GPU.  Also the wall time of the AI pieces, the stop latency recorded
by the controller (stop word -> piece ended), and the neural receiver run length.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
STEP = 50_000


def one(path: Path, reserve: int) -> dict:
    result = json.loads(path.read_text())
    config = result["config"]
    work = Path(str(path)[:-5] + "_work")
    gpus = int(config["num_gpus"])
    period_ns = int(float(config["period_ms"]) * 1e6)
    first = json.loads((work / "conv0.json").read_text())["records"][0]
    epoch = first[2] - first[0] * period_ns
    span = int(config["periods"]) * period_ns
    n = span // STEP
    on = np.zeros((gpus, n), dtype=bool)
    running = np.zeros((gpus, n), dtype=bool)
    runs, walls = [], []
    for g in range(gpus):
        for r in json.loads((work / f"lane{g}.json").read_text())["records"]:
            a, b = (r[2] - epoch) // STEP, (r[3] - epoch) // STEP
            on[g, max(0, a):max(0, b)] = True
            runs.append((r[3] - r[2]) / 1e6)
        for p in json.loads((work / f"ai{g}.json").read_text())["pieces"]:
            a, b = (p[0] - epoch) // STEP, (p[1] - epoch) // STEP
            running[g, max(0, a):max(0, b)] = True
            walls.append((p[1] - p[0]) / 1e6)
    free = gpus - on.sum(axis=0)
    forbidden = on & (free < reserve)[None, :]
    return {"forbidden": int(forbidden.sum()), "leak": int((running & forbidden).sum()),
            "nrx_time": int(on.sum()), "ai_next_to_nrx": int((running & on).sum()),
            "runs": runs, "walls": walls, "stop": (result.get("controller") or {}).get("stop_latency_ms")}


def main() -> None:
    job, tag, cells = sys.argv[1:4]
    print("| policy | seeds | NRx time in which AI must be off | AI still on in that time | piece wall time p50 / p90 / p99 (ms) | stop latency p50 / p90 / p99 (ms) | NRx run p50 (ms) |")
    print("|---|---|---|---|---|---|---|")
    for item in sys.argv[4:]:
        policy, _, reserve = item.partition(":")
        rows = []
        for path in sorted(RAW.glob(f"{tag}[0-9]r32{policy}_c{cells}_*_j{job}.json")):
            if re.match(rf"^{re.escape(tag)}\dr32{re.escape(policy)}_c", path.name):
                rows.append(one(path, int(reserve or 3)))
        if not rows:
            continue
        forbidden, leak = sum(r["forbidden"] for r in rows), sum(r["leak"] for r in rows)
        nrx_time = sum(r["nrx_time"] for r in rows)
        walls = np.concatenate([r["walls"] for r in rows])
        runs = np.concatenate([r["runs"] for r in rows])
        stops = [r["stop"] for r in rows if r["stop"]]
        stop = (f"{np.mean([s['p50'] for s in stops]):.2f} / {np.mean([s['p90'] for s in stops]):.2f} / "
                f"{np.mean([s['p99'] for s in stops]):.2f}") if stops else "–"
        print(f"| {policy} | {len(rows)} | {100 * forbidden / max(1, nrx_time):.0f}% of NRx time | {100 * leak / max(1, forbidden):.0f}% | "
              f"{np.percentile(walls, 50):.1f} / {np.percentile(walls, 90):.1f} / {np.percentile(walls, 99):.1f} | {stop} | {np.median(runs):.2f} |")


if __name__ == "__main__":
    main()
