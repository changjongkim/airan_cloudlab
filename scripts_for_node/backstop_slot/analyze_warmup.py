#!/usr/bin/env python3
"""When in a run the TBs past the L1 deadline occur.

usage: analyze_warmup.py JOB[,JOB] TAG CELLS POLICY ...      (POLICY n = the run without AI)

Per policy (all seeds): the late TBs in the first 5 s, in 5-10 s, 10-20 s, and after 20 s of the run, with the
TBs of each span.  The processes of a run start cold (CUDA modules load on first use, the AI worker meets every
unit size for the first time), and a long run separates this start from the steady state.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

RAW = Path("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/raw")
EDGES = (5.0, 10.0, 20.0)


def one(path: Path) -> list[list[int]]:
    config = json.loads(path.read_text())["config"]
    work = Path(str(path)[:-5] + "_work")
    period_s = float(config["period_ms"]) / 1e3
    skip, deadline = int(config.get("skip_periods", 20)), float(config["deadline_ms"]) * 1e6
    spans = [[0, 0] for _ in range(len(EDGES) + 1)]
    for cell in config["cells"]:
        ues = int(cell.get("num_ue", 1))
        for r in json.loads((work / f"conv{int(cell['cell'])}.json").read_text())["records"]:
            if r[0] < skip:
                continue
            i = sum(r[0] * period_s >= e for e in EDGES)
            spans[i][0] += ues
            spans[i][1] += ues * int(r[4] - r[2] > deadline)
    return spans


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
    names = ["first 5 s", "5-10 s", "10-20 s", "after 20 s"]
    print("| policy | seeds | " + " | ".join(f"late TBs / TBs, {n}" for n in names) + " |")
    print("|---|---|" + "---|" * len(names))
    out = {}
    for policy in ["n"] + [p for p in policies if p != "n"]:
        rows = [one(path) for _, path in sorted(paths.get(policy, {}).items())]
        if not rows:
            continue
        total = [[sum(r[i][0] for r in rows), sum(r[i][1] for r in rows)] for i in range(len(names))]
        out[policy] = {"seeds": len(rows), "spans": {n: {"tbs": t, "late": l} for n, (t, l) in zip(names, total)}}
        print(f"| {policy} | {len(rows)} | " + " | ".join(f"{l} / {t} ({100.0 * l / t:.4f}%)" if t else "–" for t, l in total) + " |")
    target = RAW.parent / f"warmup_{tag}_c{cells}_j{'_'.join(jobs)}.json"
    target.write_text(json.dumps(out, indent=2))
    print(target)


if __name__ == "__main__":
    main()
