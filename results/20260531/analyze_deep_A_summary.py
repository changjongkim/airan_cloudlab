#!/usr/bin/env python3
"""Summary table for deep-dive A: per-scenario L1 burst stats + AI overlap breakdown."""
import os, sqlite3, glob, json, re
from collections import Counter
from pathlib import Path
import numpy as np

IN = Path("/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_A")
OUT = Path("/Users/changjongkim/New_research/cloudlab_results/results/20260531/nsys_deep_A_analysis")
OUT.mkdir(parents=True, exist_ok=True)


def load_session_ns(db):
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
    base = load_session_ns(db)
    con = sqlite3.connect(db)
    cur = con.cursor()
    try:
        cur.execute("""SELECT k.start, k.end, (k.end-k.start) AS dur, sn.value FROM CUPTI_ACTIVITY_KIND_KERNEL k
                       LEFT JOIN StringIds sn ON sn.id=k.shortName ORDER BY k.start""")
        rows = cur.fetchall()
    except sqlite3.OperationalError:
        rows = []
    con.close()
    return [(base + r[0], base + r[1], r[2], r[3]) for r in rows]


def gaps_overall(kernels):
    return [(kernels[i+1][0] - kernels[i][1]) for i in range(len(kernels)-1) if kernels[i+1][0] > kernels[i][1]]


def summarize(stem):
    l1 = IN / f"{stem}_l1.sqlite"
    if not l1.exists():
        return None
    l1_kern = load_kernels(l1)
    if not l1_kern:
        return None
    gaps_ns = gaps_overall(l1_kern)
    if len(gaps_ns) < 10:
        return None
    g = np.array(gaps_ns) / 1000  # us

    # Match AI captures by scenario id (A1/A2/A3/A4 prefix)
    sid = stem.split("_")[0]  # A1, A2, A3, A4
    ai_files = sorted([p for p in IN.glob(f"{sid}_*_ai.sqlite")
                       if p.stat().st_size > 1_000_000 and "_l1" not in p.stem])

    ai_data = []
    for af in ai_files:
        ak = load_kernels(af)
        ai_data.append((af.stem.replace("_ai", ""), ak))

    # For TOP 1% L1 gaps, count which L1 kernel pair they occur between
    # and which AI workload had activity in the ±0.5ms window
    sorted_gaps = sorted(zip(gaps_ns, range(len(gaps_ns))), reverse=True)
    top_pct = max(20, int(len(gaps_ns) * 0.01))
    top = sorted_gaps[:top_pct]

    # Build window matching
    pair_counter = Counter()
    ai_counter = Counter()
    for gap_ns, i in top:
        prev_name = l1_kern[i][3] or "?"
        next_name = l1_kern[i+1][3] or "?"
        # short names
        prev_short = re.sub(r"<.*>", "", str(prev_name))[:40]
        next_short = re.sub(r"<.*>", "", str(next_name))[:40]
        pair_counter[(prev_short, next_short)] += 1
        gap_mid = (l1_kern[i][1] + l1_kern[i+1][0]) // 2
        WIN = 500_000  # 0.5 ms
        for name, ak in ai_data:
            for ks, ke, kd, kn in ak:
                if ke >= gap_mid - WIN and ks <= gap_mid + WIN:
                    ai_counter[(name, str(kn)[:50])] += 1
                    break

    return {
        "scenario": stem,
        "n_l1_kernels": len(l1_kern),
        "n_gaps": len(gaps_ns),
        "med_gap_us": float(np.median(g)),
        "p95_gap_us": float(np.percentile(g, 95)),
        "p99_gap_us": float(np.percentile(g, 99)),
        "p999_gap_us": float(np.percentile(g, 99.9)),
        "max_gap_us": float(g.max()),
        "top1_pct_count": top_pct,
        "top1_total_idle_pct_share": 100 * sum(x[0] for x in top) / sum(gaps_ns),
        "top_l1_pairs": pair_counter.most_common(5),
        "top_ai_events": ai_counter.most_common(8),
        "ai_workloads": [n for n, _ in ai_data],
    }


def main():
    stems = sorted(set(p.stem.replace("_l1", "") for p in IN.glob("*_l1.sqlite")))
    rows = []
    for s in stems:
        print(f"--- {s} ---")
        r = summarize(s)
        if r:
            rows.append(r)
            print(f"  L1 kernels={r['n_l1_kernels']:,}, p99={r['p99_gap_us']:.0f}us, p999={r['p999_gap_us']:.0f}us, max={r['max_gap_us']:.0f}us")
            print(f"  top L1 pairs (between which gap occurs):")
            for (p, n), c in r["top_l1_pairs"]:
                print(f"    {c:3d}x {p} → {n}")
            print(f"  top AI events near L1 burst:")
            for (ai, ev), c in r["top_ai_events"][:6]:
                print(f"    {c:3d}x [{ai}] {ev}")
            print()
    with open(OUT / "summary_concurrent.json", "w") as f:
        json.dump(rows, f, default=str, indent=2)
    # Build paper-ready table
    with open(OUT / "paper_table.csv", "w") as f:
        f.write("scenario,n_l1_kernels,med_gap_us,p99_gap_us,p999_gap_us,max_gap_ms,top1_pct_idle_share,ai_workloads\n")
        for r in rows:
            f.write(f"{r['scenario']},{r['n_l1_kernels']},{r['med_gap_us']:.0f},{r['p99_gap_us']:.0f},"
                    f"{r['p999_gap_us']:.0f},{r['max_gap_us']/1000:.1f},{r['top1_total_idle_pct_share']:.1f}%,"
                    f"{'+'.join(r['ai_workloads'])}\n")
    print(f"wrote {OUT}/paper_table.csv")


if __name__ == "__main__":
    main()
