#!/usr/bin/env python3
"""
Deep-dive A: Dual-concurrent nsys analysis.
For each scenario (S35/S34/M5c/M8a), correlate L1 burst gaps with AI-side events
on wall-clock timeline. Output per-scenario CSV: top L1 p999 burst gaps + nearest AI kernel/memcpy.

Inputs: <stem>_l1.sqlite + <stem>_ai.sqlite (or multiple _ai*.sqlite for multi-AI).
Outputs: CSV in nsys_deep_A_analysis/.
"""
import os, sys, sqlite3, glob, json, re
from pathlib import Path

IN = Path(os.environ.get("IN_DIR", "/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_A"))
OUT = Path(os.environ.get("OUT_DIR", "/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_A_analysis"))
OUT.mkdir(parents=True, exist_ok=True)

# nsys-rep -> .sqlite assumed already exported (nsys export --type sqlite)
# These queries assume the standard nsys sqlite schema.

L1_KERNEL_QUERY = """
SELECT k.start AS start_ns, k.end AS end_ns,
       (k.end - k.start) AS dur_ns, sn.value AS name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
LEFT JOIN StringIds sn ON sn.id = k.shortName
ORDER BY k.start
"""

AI_KERNEL_QUERY = """
SELECT k.start AS start_ns, k.end AS end_ns,
       (k.end - k.start) AS dur_ns, sn.value AS name
FROM CUPTI_ACTIVITY_KIND_KERNEL k
LEFT JOIN StringIds sn ON sn.id = k.shortName
ORDER BY k.start
"""

AI_MEMCPY_QUERY = """
SELECT m.start AS start_ns, m.end AS end_ns, (m.end - m.start) AS dur_ns,
       m.copyKind, m.bytes
FROM CUPTI_ACTIVITY_KIND_MEMCPY m
ORDER BY m.start
"""


def load_session_start_ns(db):
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        cur.execute("SELECT utcEpochNs FROM TARGET_INFO_SESSION_START_TIME LIMIT 1")
        r = cur.fetchone()
        v = int(r[0]) if r else 0
    except sqlite3.OperationalError:
        v = 0
    con.close()
    return v


def load_kernels(db):
    """Return (wall_clock_start_ns, wall_clock_end_ns, dur_ns, name) with utcEpochNs added."""
    base = load_session_start_ns(db)
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        cur.execute(L1_KERNEL_QUERY)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return [(base + r[0], base + r[1], r[2], r[3]) for r in rows]


def load_memcpy(db):
    base = load_session_start_ns(db)
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        cur.execute(AI_MEMCPY_QUERY)
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return [(base + r[0], base + r[1], r[2], r[3], r[4]) for r in rows]


def compute_gaps(kernels):
    """Inter-kernel gap from kernel[i].end to kernel[i+1].start."""
    gaps = []  # (gap_ns, prev_end_ns, next_start_ns, prev_name, next_name)
    for i in range(len(kernels) - 1):
        g = kernels[i + 1][0] - kernels[i][1]
        if g > 0:
            gaps.append((g, kernels[i][1], kernels[i + 1][0], kernels[i][3], kernels[i + 1][3]))
    return gaps


def find_near(events, t_ns, window_ns):
    """Return events whose start..end overlaps [t-window, t+window]."""
    lo = t_ns - window_ns
    hi = t_ns + window_ns
    out = []
    for ev in events:
        s, e = ev[0], ev[1]
        if e >= lo and s <= hi:
            out.append(ev)
    return out


def analyze_scenario(stem):
    l1_db = IN / f"{stem}_l1.sqlite"
    if not l1_db.exists():
        print(f"  skip {stem}: no L1 sqlite ({l1_db})")
        return None
    ai_dbs = sorted(glob.glob(str(IN / f"{stem.replace('_3gL1','')}*_ai.sqlite"))) + sorted(glob.glob(str(IN / f"{stem}_ai*.sqlite")))
    # fallback: any _ai.sqlite that shares prefix
    if not ai_dbs:
        ai_dbs = sorted(glob.glob(str(IN / f"{stem}*ai*.sqlite")))
    print(f"  L1 db: {l1_db.name}, AI dbs: {len(ai_dbs)}")

    l1_kernels = load_kernels(l1_db)
    if not l1_kernels:
        print(f"  no L1 kernels")
        return None
    gaps = compute_gaps(l1_kernels)
    gaps.sort(key=lambda x: -x[0])
    top_n = max(20, int(len(gaps) * 0.001))
    top_gaps = gaps[:top_n]

    # Wall-clock offset: L1 sqlite and AI sqlite use independent CUPTI clocks.
    # Use approximate offset from session start (process_start) if available.
    # If both captured on same host with NSYS_LAUNCH_TIME equivalent, treat starts as aligned w/ delay.

    ai_kernels_all = []
    ai_memcpy_all = []
    for aidb in ai_dbs:
        ak = load_kernels(Path(aidb))
        am = load_memcpy(Path(aidb))
        ai_kernels_all.append((Path(aidb).stem, ak, am))

    # Match each top gap to AI events in [gap_start - 1ms, gap_end + 1ms]
    rows = []
    WIN = 1_000_000  # 1 ms
    for g, prev_end, next_start, pname, nname in top_gaps:
        gap_mid = (prev_end + next_start) // 2
        ai_hits = []
        for aname, ak, am in ai_kernels_all:
            nk = find_near(ak, gap_mid, WIN)
            nm = find_near(am, gap_mid, WIN)
            if nk:
                ai_hits.append((aname, "kernel", len(nk), nk[0][3] if nk else ""))
            if nm:
                # sum bytes in window
                total_bytes = sum(x[4] for x in nm)
                ai_hits.append((aname, "memcpy", len(nm), f"{total_bytes/1024:.0f}KB"))
        ai_summary = ";".join(f"{a}/{t}/{c}/{n}" for a, t, c, n in ai_hits) or "none"
        rows.append((g, prev_end, next_start, pname, nname, ai_summary))

    out_csv = OUT / f"{stem}_top_gaps_vs_ai.csv"
    with open(out_csv, "w") as f:
        f.write("gap_ns,prev_end_ns,next_start_ns,prev_kernel,next_kernel,ai_events_within_1ms\n")
        for r in rows:
            f.write(",".join(str(x) for x in r) + "\n")
    print(f"  wrote {out_csv.name}: {len(rows)} top gaps")

    # Summary: hit rate (AI events present within 1ms of L1 burst)
    hits = sum(1 for r in rows if r[5] != "none")
    print(f"  hit rate: {hits}/{len(rows)} = {100*hits/len(rows):.1f}%")
    return {"scenario": stem, "top_gaps": len(rows), "hit_rate": hits / max(1, len(rows))}


def main():
    if not IN.exists():
        print(f"No input dir {IN}")
        sys.exit(0)
    stems = set()
    for f in IN.glob("*_l1.sqlite"):
        stems.add(f.stem.replace("_l1", ""))
    if not stems:
        print(f"No *_l1.sqlite in {IN}; run nsys export first")
        sys.exit(0)
    summary = []
    for s in sorted(stems):
        print(f"--- {s} ---")
        r = analyze_scenario(s)
        if r:
            summary.append(r)
    with open(OUT / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE")


if __name__ == "__main__":
    main()
