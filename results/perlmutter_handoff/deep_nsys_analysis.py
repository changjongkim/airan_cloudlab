#!/usr/bin/env python3
"""Deep nsys SQLite analysis (matches MIG doc §12/§16/§17): GPU busy vs idle-gap
decomposition + per-call memcpy/memset duration distributions + inter-op gaps.
Run inside container (sqlite3 stdlib). Compares default time-slice vs MPS."""
import sqlite3, glob, os, sys
import numpy as np

BASE = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig"

def tbl_exists(c, t):
    return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?", (t,)).fetchone() is not None

def load(db):
    c = sqlite3.connect(db)
    ops = []  # (start, end, kind)
    if tbl_exists(c, "CUPTI_ACTIVITY_KIND_KERNEL"):
        for s, e in c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_KERNEL"): ops.append((s, e, "K"))
    memcpy = []
    if tbl_exists(c, "CUPTI_ACTIVITY_KIND_MEMCPY"):
        for s, e in c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY"):
            ops.append((s, e, "C")); memcpy.append(e - s)
    memset = []
    if tbl_exists(c, "CUPTI_ACTIVITY_KIND_MEMSET"):
        for s, e in c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMSET"):
            ops.append((s, e, "S")); memset.append(e - s)
    c.close()
    return ops, np.array(memcpy), np.array(memset)

def analyze(db):
    ops, memcpy, memset = load(db)
    if not ops: return None
    ops.sort()
    starts = np.array([o[0] for o in ops]); ends = np.array([o[1] for o in ops])
    busy = (ends - starts).sum()                       # sum of GPU-op durations
    wall = ends.max() - starts.min()                   # timeline span
    # inter-op gaps (idle between consecutive GPU ops on the L1 timeline)
    gaps = []
    cur_end = ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s > cur_end: gaps.append(s - cur_end)
        cur_end = max(cur_end, e)
    gaps = np.array(gaps) if gaps else np.array([0])
    return {
        "n_ops": len(ops), "wall_ms": wall/1e6, "busy_ms": busy/1e6,
        "gap_frac": 1 - busy/wall if wall else 0,
        "gap_med_us": np.percentile(gaps,50)/1e3, "gap_p99_us": np.percentile(gaps,99)/1e3,
        "memcpy_med_us": (np.percentile(memcpy,50)/1e3) if len(memcpy) else 0,
        "memcpy_p99_us": (np.percentile(memcpy,99)/1e3) if len(memcpy) else 0,
        "memset_med_us": (np.percentile(memset,50)/1e3) if len(memset) else 0,
        "memset_p99_us": (np.percentile(memset,99)/1e3) if len(memset) else 0,
    }

conds = sys.argv[1:] or ["alone","neuralrx","sat_hbm"]
print(f"{'cond':10s} {'regime':8s} {'gap_frac':>8s} {'gap_med':>8s} {'gap_p99':>9s} {'mcpy_med':>9s} {'mcpy_p99':>9s} {'mset_med':>9s} {'mset_p99':>9s}")
print("-"*92)
for cond in conds:
    for regime, db in [("default", f"{BASE}/nsys_nomig/nsys_{cond}.sqlite"),
                       ("MPS",     f"{BASE}/MPS_nsys/MPSnsys_{cond}.sqlite")]:
        if not os.path.exists(db): print(f"{cond:10s} {regime:8s} (no db)"); continue
        r = analyze(db)
        if not r: print(f"{cond:10s} {regime:8s} (empty)"); continue
        print(f"{cond:10s} {regime:8s} {r['gap_frac']*100:7.1f}% {r['gap_med_us']:7.1f} {r['gap_p99_us']:8.1f} "
              f"{r['memcpy_med_us']:8.2f} {r['memcpy_p99_us']:8.2f} {r['memset_med_us']:8.2f} {r['memset_p99_us']:8.2f}")
    print()
