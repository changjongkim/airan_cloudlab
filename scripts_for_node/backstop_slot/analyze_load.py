#!/usr/bin/env python3
"""Conventional completion time by the number of active cells on the GPU in that period.

usage: analyze_load.py RESULT.json   (a run with partial load)
Prints, per load n, the count and p50/p99/p99.9/max of the conventional completion time and
the share of TBs past the L1 deadline: the table behind the per-period delay budget.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from activity import activity_mask, gpu_load  # noqa: E402


def main() -> None:
    path = Path(sys.argv[1])
    d = json.loads(path.read_text())
    cfg = d["config"]
    work = Path(str(path).replace(".json", "_work"))
    mask = activity_mask(cfg)
    load = gpu_load(cfg, mask)
    skip = int(cfg.get("skip_periods", 20))
    deadline = float(cfg["deadline_ms"])
    by_load: dict[int, list[float]] = {}
    for cell in cfg["cells"]:
        conv = json.loads((work / f"conv{cell['cell']}.json").read_text())
        g = int(cell["gpu"])
        for r in conv["records"]:
            if r[0] >= skip:
                by_load.setdefault(int(load[g, r[0]]), []).append((r[4] - r[2]) / 1e6)
    table = {}
    print("| active cells on the GPU | TBs | conv p50 | p99 | p99.9 | max | past L1 |")
    print("|---|---|---|---|---|---|---|")
    for n in sorted(by_load):
        t = np.asarray(by_load[n])
        table[n] = {"tbs": int(t.size), "p50": float(np.percentile(t, 50)), "p99": float(np.percentile(t, 99)),
                    "p999": float(np.percentile(t, 99.9)), "max": float(t.max()),
                    "late": int((t > deadline).sum())}
        e = table[n]
        print(f"| {n} | {e['tbs']} | {e['p50']:.2f} | {e['p99']:.2f} | {e['p999']:.2f} | {e['max']:.2f} | {e['late']} |")
    Path(str(path).replace(".json", "_byload.json")).write_text(json.dumps(table, indent=2))


if __name__ == "__main__":
    main()
