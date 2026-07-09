#!/usr/bin/env python3
"""Per-call cudaFree distribution analysis from nsys SQLite databases."""
import sqlite3
import os
import json

FILES = [
    ("TS_alone_t1",   "/data/mps_ts_v2/TS_alone_t1"),
    ("TS_alone_t2",   "/data/mps_ts_v2/TS_alone_t2"),
    ("TS_alone_t3",   "/data/mps_ts_v2/TS_alone_t3"),
    ("TS_coloc_t1",   "/data/mps_ts_v2/TS_coloc_t1"),
    ("TS_coloc_t2",   "/data/mps_ts_v2/TS_coloc_t2"),
    ("TS_coloc_t3",   "/data/mps_ts_v2/TS_coloc_t3"),
    ("MPS_alone_t1",  "/data/mps_only/MPS_alone_t1"),
    ("MPS_alone_t2",  "/data/mps_only/MPS_alone_t2"),
    ("MPS_alone_t3",  "/data/mps_only/MPS_alone_t3"),
    ("MPS_coloc_t1",  "/data/mps_only/MPS_coloc_t1"),
    ("MPS_coloc_t2",  "/data/mps_only/MPS_coloc_t2"),
    ("MPS_coloc_t3",  "/data/mps_only/MPS_coloc_t3"),
    ("MPS_hbm",       "/data/mps_hbm/MPS_hbm"),
]

def analyze(name, base):
    sqlite_path = base + ".sqlite"
    if not os.path.exists(sqlite_path):
        os.system(f"nsys export --type sqlite --output {sqlite_path} {base}.nsys-rep >/dev/null 2>&1")
    if not os.path.exists(sqlite_path):
        return {"name": name, "error": "no sqlite"}

    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()

    # Find cudaFree call durations
    try:
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaFree%v____' AND value NOT LIKE '%Host%' AND value NOT LIKE '%Async%' AND value NOT LIKE '%Mip%'")
        rows = cur.fetchall()
        if not rows:
            con.close()
            return {"name": name, "error": "no cudaFree StringId"}
        sids = [r[0] for r in rows]
        placeholders = ",".join("?"*len(sids))
        cur.execute(f"SELECT (end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({placeholders}) ORDER BY (end - start)", sids)
        durations = [r[0] for r in cur.fetchall()]
    except Exception as e:
        con.close()
        return {"name": name, "error": str(e)}
    con.close()

    if not durations:
        return {"name": name, "error": "empty"}

    n = len(durations)
    def p(pct):
        return durations[min(n-1, int(n*pct/100))]

    under_1ms  = sum(1 for d in durations if d <  1_000_000)
    under_10ms = sum(1 for d in durations if d < 10_000_000)
    return {
        "name": name,
        "n": n,
        "total_ms": sum(durations)/1e6,
        "min_us": durations[0]/1000,
        "p25_us": p(25)/1000,
        "p50_us": p(50)/1000,
        "p75_us": p(75)/1000,
        "p95_us": p(95)/1000,
        "p99_us": p(99)/1000,
        "max_us": durations[-1]/1000,
        "pct_fast_lt_1ms": 100*under_1ms/n,
        "pct_mid_1_to_10ms": 100*(under_10ms - under_1ms)/n,
        "pct_slow_gt_10ms": 100*(n - under_10ms)/n,
    }

if __name__ == "__main__":
    results = []
    for name, base in FILES:
        r = analyze(name, base)
        results.append(r)
        if "error" in r:
            print(f"[{name}] ERROR: {r['error']}", flush=True)
            continue
        print(f"\n=== {name} ===")
        print(f"  n={r['n']} total={r['total_ms']:.0f} ms")
        print(f"  min={r['min_us']:>8.1f} p25={r['p25_us']:>8.1f} p50={r['p50_us']:>8.1f} p75={r['p75_us']:>8.1f} p95={r['p95_us']:>8.1f} p99={r['p99_us']:>8.1f} max={r['max_us']:>10.1f}  (µs)")
        print(f"  bimodal: <1ms {r['pct_fast_lt_1ms']:.1f}% | 1-10ms {r['pct_mid_1_to_10ms']:.1f}% | >10ms {r['pct_slow_gt_10ms']:.1f}%")

    with open("/tmp/cudaFree_analysis.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nSaved: /tmp/cudaFree_analysis.json")
