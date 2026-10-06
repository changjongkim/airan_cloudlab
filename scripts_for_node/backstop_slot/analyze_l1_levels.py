#!/usr/bin/env python3
"""What the neural receiver and the AI each add to the layer-1 latency.

usage: analyze_l1_levels.py OUT.json JOB[,JOB] TAG:CELLS[:LABEL] ...     (runs of la_cl.sh / la_cl2.sh)

The single-user cells of a closed-loop run have the same workload under every policy (their MCS is fixed), so
their layer-1 latency (arrival of the slot's samples to the conventional result) compares three levels:
  x     conventional receiver alone: no neural receiver and no AI
  n     neural receiver for the failed TBs of the two-user cells, no AI
  ...   the same with AI under each policy
Printed per condition: median, 99th and 99.9th percentile, and the room left to the layer-1 deadline at the
99.9th percentile.  Periods before SKIP (default 2000 = 5 s) are left out; the count of TBs past the deadline
is not printed, because runs of 20 s still carry the misses of the start of a run (README 6.0.2).

For runs at a fixed MCS (v16_long_x.sh, v16_long.sh) every cell has the same workload at every level:
ALL_CELLS=1 takes all cells, and with SKIP=8000 (the first 20 s of a run of 100 s) LATE=1 adds the 99.99th
percentile and the TBs past the deadline.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import numpy as np

from loss_trace import RAW

SKIP = int(os.environ.get("SKIP", "2000"))
ALL_CELLS = bool(os.environ.get("ALL_CELLS"))
LATE = bool(os.environ.get("LATE"))
DEADLINE_MS = 4.0
ORDER = ("x", "n", "wm", "wd", "s10", "p30", "p70", "p100")
NAMES = {"x": "Conventional receiver alone", "n": "+ neural receiver, no AI", "wm": "+ AI, rule", "wd": "+ AI, rule",
         "s10": "+ AI, fixed 10%", "p30": "+ AI, 30% share with low priority", "p70": "+ AI, 70% share with low priority",
         "p100": "+ AI, low priority alone"}


def latencies(jobs: list[str], tag: str, cells: str) -> dict[str, np.ndarray]:
    pattern = re.compile(rf"^{re.escape(tag)}(\d)(x|n|r32([a-z0-9]+))_c{cells}_")
    out: dict[str, list] = {}
    for job in jobs:
        for path in sorted(RAW.glob(f"{tag}[0-9]*_c{cells}_*_j{job}.json")):
            m = pattern.match(path.name)
            if not m:
                continue
            config = json.loads(path.read_text())["config"]
            work = Path(str(path)[:-5] + "_work")
            for cell in config["cells"]:
                if cell.get("nrx_gpu") is not None and not ALL_CELLS:
                    continue
                rows = json.loads((work / f"conv{cell['cell']}.json").read_text())["records"]
                r = np.asarray([row[:5] for row in rows], dtype=np.int64)
                r = r[r[:, 0] >= SKIP]
                ues = int(cell.get("num_ue", 2)) if cell.get("nrx_gpu") is not None else 1      # a slot of a two-user cell carries two TBs
                out.setdefault(m.group(3) or m.group(2), []).append(np.repeat((r[:, 4] - r[:, 2]) / 1e6, ues))
    return {policy: np.concatenate(v) for policy, v in out.items()}


def main() -> None:
    out_path, jobs = Path(sys.argv[1]), sys.argv[2].split(",")
    result = {}
    for spec in sys.argv[3:]:
        tag, cells, *label = spec.split(":")
        by = latencies(jobs, tag, cells)
        print(f"## {label[0] if label else tag}  ({'all' if ALL_CELLS else 'single-user'} cells, periods from {SKIP})\n")
        print("| level | TBs | median | 99th | 99.9th | room to the deadline at the 99.9th |" + (" 99.99th | TBs past the deadline |" if LATE else ""))
        print("|---|---|---|---|---|---|" + ("---|---|" if LATE else ""))
        rows = {}
        for policy in ORDER:
            if policy not in by:
                continue
            v = by[policy]
            p50, p99, p999 = (float(np.percentile(v, q)) for q in (50, 99, 99.9))
            rows[policy] = {"tbs": int(len(v)), "p50_ms": p50, "p99_ms": p99, "p999_ms": p999, "room_ms": DEADLINE_MS - p999}
            extra = ""
            if LATE:
                rows[policy].update(p9999_ms=float(np.percentile(v, 99.99)), late=int((v > DEADLINE_MS).sum()))
                extra = f" {rows[policy]['p9999_ms']:.2f} | {rows[policy]['late']} |"
            print(f"| {NAMES[policy]} | {len(v)} | {p50:.2f} | {p99:.2f} | {p999:.2f} | {DEADLINE_MS - p999:.2f} ms |{extra}")
        if "x" in rows and "n" in rows:
            added = rows["n"]["p999_ms"] - rows["x"]["p999_ms"]
            print(f"\nthe neural receiver adds {added:.2f} ms at the 99.9th percentile "
                  f"({100 * added / rows['x']['room_ms']:.0f}% of the room of the conventional receiver alone); "
                  + ", ".join(f"{q} adds {rows[q]['p999_ms'] - rows['n']['p999_ms']:+.2f}" for q in ORDER[2:] if q in rows) + "\n")
        result[tag] = rows
    out_path.write_text(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
