#!/usr/bin/env python3
"""Aggregate cudaFree stats from matrix_mps_ts experiment."""
import sqlite3
import os
import json
import glob
import statistics

MATRIX_DIR = "/data"
OUT_JSON = "/tmp/matrix_summary.json"

def cudaFree_stats(sqlite_path):
    if not os.path.exists(sqlite_path):
        return None
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    try:
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaFree%v____' AND value NOT LIKE '%Host%' AND value NOT LIKE '%Async%' AND value NOT LIKE '%Mip%'")
        rows = cur.fetchall()
        if not rows:
            return None
        sids = [r[0] for r in rows]
        placeholders = ",".join("?"*len(sids))
        cur.execute(f"SELECT (end - start) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({placeholders})", sids)
        durations = [r[0] for r in cur.fetchall()]
    finally:
        con.close()

    if not durations:
        return None
    durations.sort()
    n = len(durations)
    def p(pct):
        return durations[min(n-1, int(n*pct/100))]

    total_ms = sum(durations) / 1e6
    return {
        "n": n,
        "total_ms": total_ms,
        "mean_us": sum(durations) / n / 1000,
        "p50_us": p(50) / 1000,
        "p95_us": p(95) / 1000,
        "p99_us": p(99) / 1000,
        "max_us": durations[-1] / 1000,
        "pct_lt_1ms":  100 * sum(1 for d in durations if d < 1_000_000) / n,
        "pct_1_to_10ms": 100 * sum(1 for d in durations if 1_000_000 <= d < 10_000_000) / n,
        "pct_gt_10ms": 100 * sum(1 for d in durations if d >= 10_000_000) / n,
    }

def ensure_sqlite(nsys_rep):
    sqlite = nsys_rep.replace(".nsys-rep", ".sqlite")
    if not os.path.exists(sqlite):
        os.system(f"nsys export --type sqlite --output {sqlite} {nsys_rep} >/dev/null 2>&1")
    return sqlite

def condition_key(fname):
    """Parse filename like 'TS_c20_nrx_t1.nsys-rep' → ('TS', 20, 'nrx')."""
    base = os.path.basename(fname).replace(".nsys-rep", "")
    parts = base.split("_")
    mode = parts[0]
    cells = int(parts[1][1:]) if parts[1].startswith("c") else None
    workload = parts[2] if len(parts) > 3 else "alone"
    return (mode, cells, workload)

def main():
    files = sorted(glob.glob(f"{MATRIX_DIR}/*.nsys-rep"))
    print(f"Found {len(files)} nsys files")

    results = {}
    for f in files:
        cond = condition_key(f)
        key = f"{cond[0]}_c{cond[1]}_{cond[2]}"
        sqlite = ensure_sqlite(f)
        stats = cudaFree_stats(sqlite)
        if stats is None:
            print(f"  [{os.path.basename(f)}] EMPTY (skipped)")
            continue
        results.setdefault(key, []).append(stats)

    # aggregate per condition (mean of 3 trials)
    summary = {}
    for key, trials in results.items():
        if not trials:
            continue
        summary[key] = {
            "n_trials": len(trials),
            "cudaFree_total_ms_mean": statistics.mean(t["total_ms"] for t in trials),
            "cudaFree_total_ms_std": statistics.stdev([t["total_ms"] for t in trials]) if len(trials) > 1 else 0,
            "cudaFree_calls_mean": statistics.mean(t["n"] for t in trials),
            "cudaFree_p50_us": statistics.mean(t["p50_us"] for t in trials),
            "cudaFree_p95_us": statistics.mean(t["p95_us"] for t in trials),
            "pct_fast_lt_1ms": statistics.mean(t["pct_lt_1ms"] for t in trials),
            "pct_slow_1_10ms": statistics.mean(t["pct_1_to_10ms"] for t in trials),
            "pct_cat_gt_10ms": statistics.mean(t["pct_gt_10ms"] for t in trials),
        }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # print table sorted by mode, cells, workload
    print("\n=== Matrix summary ===")
    print(f"{'condition':<28} {'trials':<6} {'total_ms':>10} {'stdev':>8} {'n_calls':>8} {'p50_us':>10} {'p95_us':>10} {'%fast':>7} {'%slow':>7} {'%cat':>7}")
    def sort_key(k):
        mode, c_str, wl = k.split("_", 2)
        cells = int(c_str[1:])
        wl_order = {"alone": 0, "nrx": 1, "chanpred": 2, "hbm": 3, "resnet": 4}.get(wl, 99)
        mode_order = {"TS": 0, "MPS": 1}.get(mode, 99)
        return (mode_order, cells, wl_order)

    for key in sorted(summary.keys(), key=sort_key):
        s = summary[key]
        print(f"{key:<28} {s['n_trials']:<6} {s['cudaFree_total_ms_mean']:>10.0f} {s['cudaFree_total_ms_std']:>8.0f} "
              f"{s['cudaFree_calls_mean']:>8.0f} {s['cudaFree_p50_us']:>10.0f} {s['cudaFree_p95_us']:>10.0f} "
              f"{s['pct_fast_lt_1ms']:>7.1f} {s['pct_slow_1_10ms']:>7.1f} {s['pct_cat_gt_10ms']:>7.1f}")

    print(f"\nSaved: {OUT_JSON}")

if __name__ == "__main__":
    main()
