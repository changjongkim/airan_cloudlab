#!/usr/bin/env python3
"""Chain 10 analysis: non-CUDA-API approach comparison.
Extracts cudaFree total, cudaMemcpyAsync total, and total host CUDA time
for each condition (approach × mode × alone/nrx × trial).
"""
import sqlite3, glob, os, json, statistics
from collections import defaultdict

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain10"

def stats(sq):
    con = sqlite3.connect(sq)
    cur = con.cursor()
    try:
        def sids_like(pat):
            cur.execute(f"SELECT id FROM StringIds WHERE value LIKE '{pat}%v____' AND value NOT LIKE '%Host%' AND value NOT LIKE '%Mip%'")
            return [r[0] for r in cur.fetchall()]
        cf = sids_like('cudaFree')
        cfa = [i for i in sids_like('cudaFreeAsync') if i]
        cf = [i for i in cf if i not in cfa]  # exclude async from Free set
        # Actually let me redo: cudaFree pattern includes cudaFreeAsync
        cur.execute("SELECT id FROM StringIds WHERE value = 'cudaFree_v3020' OR value LIKE 'cudaFree_v____'")
        cf_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaFreeAsync%'")
        cfa_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMemcpyAsync%'")
        mca_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaMalloc_v____'")
        cma_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaLaunchKernel%v____'")
        clk_sids = [r[0] for r in cur.fetchall()]
        cur.execute("SELECT id FROM StringIds WHERE value LIKE 'cudaGraphLaunch%'")
        cgl_sids = [r[0] for r in cur.fetchall()]

        def sumns(sids):
            if not sids: return 0, 0
            ph = ",".join("?"*len(sids))
            cur.execute(f"SELECT SUM(end-start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({ph})", sids)
            r = cur.fetchone()
            return (r[0] or 0), (r[1] or 0)

        cf_ns, cf_n   = sumns(cf_sids)
        cfa_ns, cfa_n = sumns(cfa_sids)
        mca_ns, mca_n = sumns(mca_sids)
        cma_ns, cma_n = sumns(cma_sids)
        clk_ns, clk_n = sumns(clk_sids)
        cgl_ns, cgl_n = sumns(cgl_sids)

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
        "cudaLaunchKernel_ms": clk_ns/1e6, "cudaLaunchKernel_n": clk_n,
        "cudaGraphLaunch_ms": cgl_ns/1e6, "cudaGraphLaunch_n": cgl_n,
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
    def avg(f): return statistics.mean(t[f] for t in trials)
    summary[k] = {
        "n_trials": len(trials),
        "cudaFree_ms": avg("cudaFree_ms"), "cudaFree_n": avg("cudaFree_n"),
        "cudaFreeAsync_ms": avg("cudaFreeAsync_ms"), "cudaFreeAsync_n": avg("cudaFreeAsync_n"),
        "memcpyAsync_ms": avg("memcpyAsync_ms"), "memcpyAsync_n": avg("memcpyAsync_n"),
        "cudaMalloc_ms": avg("cudaMalloc_ms"),
        "cudaLaunchKernel_ms": avg("cudaLaunchKernel_ms"), "cudaLaunchKernel_n": avg("cudaLaunchKernel_n"),
        "cudaGraphLaunch_ms": avg("cudaGraphLaunch_ms"), "cudaGraphLaunch_n": avg("cudaGraphLaunch_n"),
        "total_host_ms": avg("total_host_ms"),
    }

def sk(k):
    p = k.split("_")
    approach = p[0]
    mode_or_cond = p[1] if len(p) > 1 else ""
    approach_order = {"baseL1":0,"graphL1":1,"mpsP30":2,"mpsP50":3,"mpsP70":4}.get(approach,99)
    mode_order = {"TS":0,"MPS":1,"alone":2,"nrx":3}.get(mode_or_cond,99)
    return (approach_order, mode_order, k)

print(f"{'condition':<32} {'n':>2} {'cudaFree':>10} {'#calls':>7} {'FreeAsync':>10} {'memcpy':>10} {'malloc':>10} {'clkKrl':>10} {'#klaunches':>10} {'graphLnch':>10} {'total':>10}")
print("-" * 150)
for k in sorted(summary.keys(), key=sk):
    s = summary[k]
    print(f"{k:<32} {s['n_trials']:>2} {s['cudaFree_ms']:>10.0f} {s['cudaFree_n']:>7.0f} "
          f"{s['cudaFreeAsync_ms']:>10.0f} {s['memcpyAsync_ms']:>10.0f} {s['cudaMalloc_ms']:>10.0f} "
          f"{s['cudaLaunchKernel_ms']:>10.0f} {s['cudaLaunchKernel_n']:>10.0f} "
          f"{s['cudaGraphLaunch_ms']:>10.0f} {s['total_host_ms']:>10.0f}")

out = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain10_summary.json"
with open(out,"w") as f: json.dump(summary, f, indent=2)
print(f"\nSaved: {out}")
