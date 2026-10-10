#!/usr/bin/env python3
"""Candidates not recovered in time against the decision deadline D and the time limit B of a method.

Event model on the measured candidates of runs without AI.  A candidate is counted as not
recovered if it found no neural receiver by its latest start L = D - B, or if its run ended
after D.  For each method and deadline the time limit with the smallest total is reported:
a method whose runs are longer needs a larger limit, which moves its latest start earlier.
Output: results/backstop_slot/time_limit_choice.json
"""
import json
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/scripts_for_node/backstop_slot")
from loss_trace import trace, RAW
from loss_model import Setting, simulate


def names(tag, job, n=5):
    return sorted(p.stem for p in RAW.glob(f"{tag}[0-9]n_c16_*_none_j{job}.json"))[:n]


CASES = {"full load": names("fa", "59313958"), "8 two-user cells, target 1%": names("lec", "59482496", 3)}
# rule, run length next to AI, conventional result later by, stop tail (medians measured at full load)
METHODS = {"no AI": ("none", 7.97, 0.0, 0.0), "Antiphase": ("never", 7.97, 0.3, 0.4),
           "Priority (30% cap)": ("always", 7.04, 0.13, 0.0), "Priority-70": ("always", 7.62, 0.29, 0.0),
           "Priority-max": ("always", 7.97, 0.39, 0.0)}
STARTS = [round(x, 1) for x in np.arange(2.4, 6.01, 0.2)]
DEADLINES = (11.5, 12.0, 12.5)


def one(job):
    name, label, latest = job
    rule, s1, shift, tail = METHODS[label]
    t = trace(name)
    alone = float(np.median(t["run_ms"][t["start_ms"] >= 0]))
    s = Setting(lanes=int(t["lanes"]), latest_start_ms=latest, run_alone_ms=alone, run_ai_ms=s1 + (alone - 6.29),
                known_shift_ms=shift, stop_latency_ms=tail)
    r = simulate(t["period"], t["known_ms"], rule, s, periods=int(t["periods"]))
    return name, label, latest, r["candidates"], r["lost"], [float((r["end_offset_ms"] > d).sum()) for d in DEADLINES]


def main():
    out = {}
    for case, runs in CASES.items():
        jobs = [(n, m, L) for n in runs for m in METHODS for L in STARTS]
        with Pool(16) as pool:
            rows = pool.map(one, jobs)
        print("\n" + case)
        for di, deadline in enumerate(DEADLINES):
            print(f" decision deadline {deadline} ms after the samples arrive")
            for label in METHODS:
                best = None
                for latest in STARTS:
                    sel = [r for r in rows if r[1] == label and r[2] == latest]
                    cand = sum(r[3] for r in sel)
                    blocked = 100 * sum(r[4] for r in sel) / cand
                    late = 100 * sum(r[5][di] for r in sel) / cand
                    entry = (blocked + late, round(deadline - latest, 1), latest, blocked, late)
                    out.setdefault(case, {}).setdefault(str(deadline), {}).setdefault(label, []).append(
                        {"limit_ms": entry[1], "latest_start_ms": latest, "no_nrx_pct": round(blocked, 2), "late_pct": round(late, 2)})
                    if best is None or entry[0] < best[0] - 1e-9:
                        best = entry
                fixed = next(e for e in out[case][str(deadline)][label] if abs(e["limit_ms"] - 7.6) < 0.05) \
                    if any(abs(e["limit_ms"] - 7.6) < 0.05 for e in out[case][str(deadline)][label]) else None
                extra = f"; with the 7.6 ms limit: {fixed['no_nrx_pct'] + fixed['late_pct']:.1f}% ({fixed['no_nrx_pct']:.1f} + {fixed['late_pct']:.1f} late)" if fixed else ""
                print(f"  {label:20s} best limit {best[1]:.1f} ms (L {best[2]:.1f}): {best[0]:.1f}% not recovered "
                      f"({best[3]:.1f} no receiver + {best[4]:.1f} late){extra}")
    json.dump(out, open("/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/backstop_slot/time_limit_choice.json", "w"), indent=1)


if __name__ == "__main__":
    main()
