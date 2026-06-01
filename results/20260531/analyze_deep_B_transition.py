#!/usr/bin/env python3
"""
Deep-dive B: AI ON->OFF->ON transition.
For each scenario, compute time-windowed p99 gap before/during/after AI ON.
Fit recovery time constant.
"""
import os, sys, sqlite3, glob, json
from pathlib import Path
import numpy as np

IN = Path(os.environ.get("IN_DIR", "/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_B"))
OUT = Path(os.environ.get("OUT_DIR", "/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_B_analysis"))
OUT.mkdir(parents=True, exist_ok=True)


def load_l1(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    cur.execute("SELECT start, end FROM CUPTI_ACTIVITY_KIND_KERNEL ORDER BY start")
    rows = cur.fetchall()
    con.close()
    return rows


def gaps_with_t(rows):
    """Return (t_mid_sec, gap_us) list, t relative to first kernel."""
    if len(rows) < 2:
        return []
    t0 = rows[0][0]
    g = []
    for i in range(len(rows) - 1):
        gap = rows[i + 1][0] - rows[i][1]
        if gap <= 0:
            continue
        tmid = ((rows[i + 1][0] + rows[i][1]) / 2 - t0) / 1e9
        g.append((tmid, gap / 1000.0))
    return g


def windowed_p99(gaps, t_start, t_end):
    vals = [g for t, g in gaps if t_start <= t < t_end]
    if not vals:
        return None, 0
    return float(np.percentile(vals, 99)), len(vals)


def fit_recovery(gaps, t_off):
    """After AI OFF (t_off), measure p99 in 2-sec rolling windows, fit decay."""
    points = []
    for w_start in np.arange(t_off, t_off + 20, 1.0):
        p99, n = windowed_p99(gaps, w_start, w_start + 2.0)
        if p99 is not None:
            points.append((w_start - t_off, p99, n))
    return points


def analyze_one(stem):
    l1_db = IN / f"{stem}_l1.sqlite"
    if not l1_db.exists():
        return None
    phases_csv = IN / f"{stem}_phases.csv"
    phases = {}
    if phases_csv.exists():
        with open(phases_csv) as f:
            next(f)
            for line in f:
                p, t = line.strip().split(",")
                phases[p] = float(t)
    rows = load_l1(l1_db)
    gaps = gaps_with_t(rows)
    if not gaps:
        return None

    # Window p99 every 2 sec
    tmax = gaps[-1][0]
    print(f"  L1 length {tmax:.1f}s, phases: {phases}")
    out_csv = OUT / f"{stem}_p99_timeline.csv"
    with open(out_csv, "w") as f:
        f.write("t_sec,p99_gap_us,n_gaps\n")
        for w_start in np.arange(0, tmax, 2.0):
            p99, n = windowed_p99(gaps, w_start, w_start + 2.0)
            if p99 is not None:
                f.write(f"{w_start:.1f},{p99:.1f},{n}\n")
    print(f"  wrote {out_csv.name}")

    # Per-phase summary
    phase_csv = OUT / f"{stem}_per_phase.csv"
    with open(phase_csv, "w") as f:
        f.write("phase,t_start,t_end,p99_us,p50_us,n_gaps\n")
        phase_keys = list(phases.keys())
        for i, p in enumerate(phase_keys):
            t_s = phases[p]
            t_e = phases[phase_keys[i + 1]] if i + 1 < len(phase_keys) else tmax
            p99, n = windowed_p99(gaps, t_s, t_e)
            vals = [g for t, g in gaps if t_s <= t < t_e]
            p50 = float(np.median(vals)) if vals else 0
            f.write(f"{p},{t_s:.1f},{t_e:.1f},{p99 or 0:.1f},{p50:.1f},{n}\n")
    print(f"  wrote {phase_csv.name}")

    # Recovery fit: after phase3 (AI off)
    if "phase3_ai_off" in phases:
        rec = fit_recovery(gaps, phases["phase3_ai_off"])
        rec_csv = OUT / f"{stem}_recovery.csv"
        with open(rec_csv, "w") as f:
            f.write("t_since_ai_off,p99_us,n\n")
            for dt, p, n in rec:
                f.write(f"{dt:.1f},{p:.1f},{n}\n")
        print(f"  wrote {rec_csv.name}: {len(rec)} recovery points")
    return stem


def main():
    if not IN.exists():
        print(f"no input {IN}")
        return
    stems = set(f.stem.replace("_l1", "") for f in IN.glob("*_l1.sqlite"))
    for s in sorted(stems):
        print(f"--- {s} ---")
        analyze_one(s)
    print("DONE")


if __name__ == "__main__":
    main()
