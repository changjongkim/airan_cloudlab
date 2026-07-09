#!/usr/bin/env python3
"""Aggregate cudaFree stats from all today's nsys captures (local Mac analysis).

Reads .sqlite files (already exported from .nsys-rep on the node).
Outputs summary JSON + human-readable table.
"""
import sqlite3
import os
import json
import glob
import statistics
from collections import defaultdict

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260708"
SUBDIRS = [
    "matrix_v2",
    "round3_mig_mps",
    "mps_ts_v2",
    "mps_only",
    "mps_hbm",
    "mps_verify",
]
OUT_JSON = os.path.join(BASE, "analysis", "cudaFree_all_conditions.json")

def cudaFree_stats(sqlite_path):
    if not os.path.exists(sqlite_path):
        return None
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    try:
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaFree%v____' "
                    "AND value NOT LIKE '%Host%' AND value NOT LIKE '%Async%' AND value NOT LIKE '%Mip%'")
        rows = cur.fetchall()
        if not rows:
            return None
        sids = [r[0] for r in rows]
        placeholders = ",".join("?"*len(sids))
        cur.execute(f"SELECT (end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({placeholders})", sids)
        durations = [r[0] for r in cur.fetchall()]

        # Also get total host CUDA time (all runtime + driver APIs) + memcpyAsync + malloc
        cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        total_runtime = cur.fetchone()[0] or 0

        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMemcpyAsync%v____'")
        r = cur.fetchall()
        memcpy_ns = 0
        if r:
            sids2 = [x[0] for x in r]
            ph2 = ",".join("?"*len(sids2))
            cur.execute(f"SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({ph2})", sids2)
            memcpy_ns = cur.fetchone()[0] or 0

        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMalloc%v____' AND value NOT LIKE '%FromPool%'")
        r = cur.fetchall()
        malloc_ns = 0
        if r:
            sids3 = [x[0] for x in r]
            ph3 = ",".join("?"*len(sids3))
            cur.execute(f"SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({ph3})", sids3)
            malloc_ns = cur.fetchone()[0] or 0
    finally:
        con.close()

    if not durations:
        return None
    durations.sort()
    n = len(durations)
    def p(pct):
        return durations[min(n-1, int(n*pct/100))]
    return {
        "n": n,
        "cudaFree_total_ms": sum(durations)/1e6,
        "cudaFree_mean_us": sum(durations)/n/1000,
        "cudaFree_p50_us": p(50)/1000,
        "cudaFree_p95_us": p(95)/1000,
        "cudaFree_p99_us": p(99)/1000,
        "cudaFree_max_us": durations[-1]/1000,
        "total_host_runtime_ms": total_runtime/1e6,
        "memcpyAsync_total_ms": memcpy_ns/1e6,
        "malloc_total_ms": malloc_ns/1e6,
        "pct_lt_1ms": 100*sum(1 for d in durations if d < 1_000_000)/n,
        "pct_1_to_10ms": 100*sum(1 for d in durations if 1_000_000 <= d < 10_000_000)/n,
        "pct_gt_10ms": 100*sum(1 for d in durations if d >= 10_000_000)/n,
    }

def condition_key(fname):
    """Normalize condition label from filename."""
    base = os.path.basename(fname).replace(".sqlite", "")
    # Strip trailing _t1/_t2/_t3
    for suffix in ["_t1", "_t2", "_t3"]:
        if base.endswith(suffix):
            base = base[:-3]
            break
    return base

def main():
    results = defaultdict(list)
    for sub in SUBDIRS:
        sub_dir = os.path.join(BASE, sub)
        if not os.path.isdir(sub_dir):
            continue
        for sqlite_path in sorted(glob.glob(f"{sub_dir}/*.sqlite")):
            stats = cudaFree_stats(sqlite_path)
            if stats is None:
                continue
            key = f"{sub}/{condition_key(sqlite_path)}"
            results[key].append(stats)

    summary = {}
    for key, trials in results.items():
        if not trials:
            continue
        summary[key] = {
            "n_trials": len(trials),
            "cudaFree_total_ms_mean": statistics.mean(t["cudaFree_total_ms"] for t in trials),
            "cudaFree_total_ms_std": statistics.stdev([t["cudaFree_total_ms"] for t in trials]) if len(trials) > 1 else 0,
            "cudaFree_calls_mean": statistics.mean(t["n"] for t in trials),
            "cudaFree_p50_us": statistics.mean(t["cudaFree_p50_us"] for t in trials),
            "cudaFree_p95_us": statistics.mean(t["cudaFree_p95_us"] for t in trials),
            "total_host_runtime_ms_mean": statistics.mean(t["total_host_runtime_ms"] for t in trials),
            "memcpyAsync_total_ms_mean": statistics.mean(t["memcpyAsync_total_ms"] for t in trials),
            "malloc_total_ms_mean": statistics.mean(t["malloc_total_ms"] for t in trials),
            "pct_fast_lt_1ms": statistics.mean(t["pct_lt_1ms"] for t in trials),
            "pct_slow_1_10ms": statistics.mean(t["pct_1_to_10ms"] for t in trials),
            "pct_cat_gt_10ms": statistics.mean(t["pct_gt_10ms"] for t in trials),
        }

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)

    # Human-readable table
    def sort_key(k):
        parts = k.split("/", 1)
        return parts

    print(f"{'Condition':<50} {'trials':<6} {'cudaFree_ms':>12} {'stdev':>8} {'#calls':>8} {'p50_us':>10} {'p95_us':>10} {'total_host':>12} {'%fast':>7} {'%slow':>7} {'%cat':>7}")
    print("-"*160)
    for key in sorted(summary.keys(), key=sort_key):
        s = summary[key]
        print(f"{key:<50} {s['n_trials']:<6} {s['cudaFree_total_ms_mean']:>12.0f} {s['cudaFree_total_ms_std']:>8.0f} "
              f"{s['cudaFree_calls_mean']:>8.0f} {s['cudaFree_p50_us']:>10.1f} {s['cudaFree_p95_us']:>10.1f} "
              f"{s['total_host_runtime_ms_mean']:>12.0f} "
              f"{s['pct_fast_lt_1ms']:>7.1f} {s['pct_slow_1_10ms']:>7.1f} {s['pct_cat_gt_10ms']:>7.1f}")

    print(f"\nSaved: {OUT_JSON}")
    print(f"Total conditions: {len(summary)}")

if __name__ == "__main__":
    main()
