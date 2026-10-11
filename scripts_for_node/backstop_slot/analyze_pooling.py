#!/usr/bin/env python3
"""Recoveries and server size: candidates that find no free neural receiver with one two-user cell per GPU.

usage: analyze_pooling.py OUT.json
Measured: one, two, three and four GPUs with four cells each, one of them a two-user cell (tags sa, sc, ob, fa).
Computed: the event simulation of predict_deadline_sweep.py for one to six GPUs, with the candidates of the
two-user cells that a server of that size would have (cells 0-5; their candidates are known from earlier runs)
and the c and S of each policy from the four-GPU runs.  Nothing was run with more than four GPUs.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from loss_trace import RAW, trace
from predict_deadline_sweep import simulate
from predict_new_conditions import NAMES, candidate_periods, timing

POLICIES = ("n", "wd", "p30", "p70")
# measured: GPUs -> (tag, cells, {policy: (code, jobs)})
MEASURED = {1: ("sa", 4, {"n": ("n", ("59313958", "59318975", "59321430")), "wd": ("wm", ("59318975", "59321430")),
                          "p30": ("p30", ("59313958", "59321430")), "p70": ("p70", ("59313958", "59321430"))}),
            2: ("sc", 8, {"n": ("n", ("59313958", "59318975")), "wd": ("wm", ("59318975",)),
                          "p30": ("p30", ("59313958",)), "p70": ("p70", ("59313958",))}),
            3: ("ob", 12, {p: (p, ("59632171",)) for p in POLICIES}),
            4: ("fa", 16, {"n": ("n", ("59313958", "59210955")), "wd": ("wm", ("59313958",)),
                           "p30": ("p30", ("59210955", "59313958")), "p70": ("p70", ("59210955", "59313958"))})}
CELLS = (0, 1, 2, 3, 4, 5)      # two-user cells of a server with one per GPU; cells 4 and 5 are known for seeds 1 and 2


def measured(gpus: int, policy: str) -> dict | None:
    tag, cells, table = MEASURED[gpus]
    code, jobs = table[policy]
    part = "n" if code == "n" else f"r32{code}"
    seen, lost = set(), []
    for job in jobs:
        for path in sorted(RAW.glob(f"{tag}[0-9]{part}_c{cells}_*_j{job}.json")):
            seed = path.name[len(tag)]
            if seed in seen:
                continue
            seen.add(seed)
            t = trace(path.stem)
            lost.append(100.0 * float((t["start_ms"] < 0).mean()))
    return {"lost_pct": float(np.mean(lost)), "runs": len(lost)} if lost else None


def computed(gpus: int, policy: str, rng) -> float:
    known, length = timing(policy, "main")
    seeds = (1, 2, 3, 4, 5) if gpus <= 4 else (1, 2)
    values = []
    for seed in seeds:
        parts = [candidate_periods(cell, seed) for cell in CELLS[:gpus]]
        periods = min(p[1] for p in parts)
        counts = np.bincount(np.concatenate([p[0][p[0] < periods] for p in parts]), minlength=periods)[:periods]
        values += [simulate(counts, known, length, 11.5, rng, lanes=gpus)["lost"] for _ in range(4 if gpus <= 4 else 10)]
    return 100.0 * float(np.mean(values))


def main() -> None:
    rng = np.random.default_rng(0)
    out = {}
    print("| GPUs (one two-user cell each) | " + " | ".join(NAMES[p] for p in POLICIES) + " |")
    print("|---|" + "---|" * len(POLICIES))
    for gpus in range(1, 7):
        cells = []
        out[gpus] = {}
        for policy in POLICIES:
            m = measured(gpus, policy) if gpus in MEASURED else None
            c = computed(gpus, policy, rng)
            out[gpus][policy] = {"computed_pct": c, "measured": m}
            cells.append(f"{c:.1f}%" + (f" / {m['lost_pct']:.1f}% ({m['runs']})" if m else ""))
        print(f"| {gpus} | " + " | ".join(cells) + " |")
    print("\ncell: candidates without a neural receiver, computed / measured (runs)")
    Path(sys.argv[1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
