#!/usr/bin/env python3
"""Chain 9 shim matrix analysis: cudaFree total per shim × mode × condition."""
import sqlite3, glob, os, json, statistics
from collections import defaultdict

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain9"

def stats(sqlite_path):
    con = sqlite3.connect(sqlite_path)
    cur = con.cursor()
    try:
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaFree%v____' AND value NOT LIKE '%Host%' AND value NOT LIKE '%Async%' AND value NOT LIKE '%Mip%'")
        cf_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaFreeAsync%v____'")
        cfa_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMemcpyAsync%v____'")
        mca_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMalloc%v____' AND value NOT LIKE '%FromPool%'")
        cma_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMallocFromPoolAsync%v____'")
        mfpa_sids = [r[0] for r in cur.fetchall()]

        def sumns(sids):
            if not sids: return 0, 0
            ph = ",".join("?"*len(sids))
            cur.execute(f"SELECT SUM(end-start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({ph})", sids)
            r = cur.fetchone()
            return (r[0] or 0), (r[1] or 0)

        cf_ns, cf_n = sumns(cf_sids)
        cfa_ns, cfa_n = sumns(cfa_sids)
        mca_ns, mca_n = sumns(mca_sids)
        cma_ns, cma_n = sumns(cma_sids)
        mfpa_ns, mfpa_n = sumns(mfpa_sids)

        cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        total_ns = cur.fetchone()[0] or 0
    except Exception as e:
        con.close(); return None
    con.close()
    return {
        "cudaFree_ms": cf_ns/1e6, "cudaFree_n": cf_n,
        "cudaFreeAsync_ms": cfa_ns/1e6, "cudaFreeAsync_n": cfa_n,
        "memcpyAsync_ms": mca_ns/1e6, "memcpyAsync_n": mca_n,
        "cudaMalloc_ms": cma_ns/1e6, "cudaMalloc_n": cma_n,
        "cudaMallocFromPool_ms": mfpa_ns/1e6, "cudaMallocFromPool_n": mfpa_n,
        "total_host_ms": total_ns/1e6,
    }

def parse_name(fname):
    base = os.path.basename(fname).replace(".sqlite","")
    for suf in ["_t1","_t2","_t3"]:
        if base.endswith(suf):
            base = base[:-3]; break
    return base

results = defaultdict(list)
for sq in sorted(glob.glob(f"{BASE}/*.sqlite")):
    s = stats(sq)
    if s is None or s["total_host_ms"] == 0:
        continue
    key = parse_name(sq)
    results[key].append(s)

summary = {}
for k, trials in results.items():
    if not trials: continue
    def avg(field): return statistics.mean(t[field] for t in trials)
    summary[k] = {
        "n_trials": len(trials),
        "cudaFree_ms": avg("cudaFree_ms"), "cudaFree_n": avg("cudaFree_n"),
        "cudaFreeAsync_ms": avg("cudaFreeAsync_ms"), "cudaFreeAsync_n": avg("cudaFreeAsync_n"),
        "memcpyAsync_ms": avg("memcpyAsync_ms"), "memcpyAsync_n": avg("memcpyAsync_n"),
        "cudaMalloc_ms": avg("cudaMalloc_ms"),
        "cudaMallocFromPool_ms": avg("cudaMallocFromPool_ms"),
        "total_host_ms": avg("total_host_ms"),
    }

# Table print (sorted by shim, mode, condition)
def sortk(k):
    p = k.split("_")
    shim = p[0]
    mode = p[1]
    cond = p[3]
    shim_order = {"baseline":0,"cudaFreeAsync":1,"cudaMemPool":2,"defer":3,"arena":4}.get(shim,99)
    mode_order = {"TS":0,"MPS":1,"MIG4g":2}.get(mode,99)
    cond_order = {"alone":0,"nrx":1}.get(cond,99)
    return (mode_order, shim_order, cond_order)

print(f"{'condition':<40} {'trials':>4} {'cudaFree':>10} {'#calls':>7} {'FreeAsync':>10} {'#call':>6} {'memcpy':>10} {'malloc':>10} {'pool':>10} {'total':>10}")
print("-" * 140)
for k in sorted(summary.keys(), key=sortk):
    s = summary[k]
    print(f"{k:<40} {s['n_trials']:>4} {s['cudaFree_ms']:>10.0f} {s['cudaFree_n']:>7.0f} "
          f"{s['cudaFreeAsync_ms']:>10.0f} {s['cudaFreeAsync_n']:>6.0f} "
          f"{s['memcpyAsync_ms']:>10.0f} {s['cudaMalloc_ms']:>10.0f} "
          f"{s['cudaMallocFromPool_ms']:>10.0f} {s['total_host_ms']:>10.0f}")

out = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain9_summary.json"
with open(out,"w") as f: json.dump(summary, f, indent=2)
print(f"\nSaved: {out}")
