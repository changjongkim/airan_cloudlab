#!/usr/bin/env python3
"""Chain 11 analysis: megakernel validation across TS/MPS100/MPS30 modes.
MIG same-part / cross-part failed setup — analyzed separately when re-run.
"""
import sqlite3, glob, os, json, statistics
from collections import defaultdict

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain11"

def stats(sq):
    con = sqlite3.connect(sq)
    cur = con.cursor()
    try:
        def like(pat):
            cur.execute(f"SELECT id FROM StringIds WHERE value LIKE '{pat}'")
            return [r[0] for r in cur.fetchall()]

        cf   = like('cudaFree_v____')
        cfa  = like('cudaFreeAsync%')
        mca  = like('cudaMemcpyAsync%')
        cma  = like('cudaMalloc_v____')
        clk  = like('cudaLaunchKernel%v____')

        def sumns(sids):
            if not sids: return 0, 0
            ph = ",".join("?"*len(sids))
            cur.execute(f"SELECT SUM(end-start), COUNT(*) FROM CUPTI_ACTIVITY_KIND_RUNTIME WHERE nameId IN ({ph})", sids)
            r = cur.fetchone()
            return (r[0] or 0), (r[1] or 0)

        cf_ns, cf_n   = sumns(cf)
        cfa_ns, _     = sumns(cfa)
        mca_ns, _     = sumns(mca)
        cma_ns, _     = sumns(cma)
        clk_ns, clk_n = sumns(clk)

        cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        total_ns = cur.fetchone()[0] or 0
    except Exception as e:
        con.close(); return None
    con.close()
    return {
        "cudaFree_ms": cf_ns/1e6, "cudaFree_n": cf_n,
        "cudaFreeAsync_ms": cfa_ns/1e6,
        "memcpyAsync_ms": mca_ns/1e6,
        "cudaMalloc_ms": cma_ns/1e6,
        "launchKernel_n": clk_n,
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
    if s is None or s["total_host_ms"] == 0: continue
    key = parse_name(sq)
    results[key].append(s)

summary = {}
for k, trials in results.items():
    def avg(f): return statistics.mean(t[f] for t in trials)
    summary[k] = {
        "n": len(trials),
        "cudaFree_ms": avg("cudaFree_ms"),
        "cudaFree_n": avg("cudaFree_n"),
        "memcpyAsync_ms": avg("memcpyAsync_ms"),
        "cudaMalloc_ms": avg("cudaMalloc_ms"),
        "launchKernel_n": avg("launchKernel_n"),
        "total_host_ms": avg("total_host_ms"),
    }

def sk(k):
    p = k.split("_")
    wl_order = {"baseL1":0,"persistBase":1,"persistMega":2}.get(p[0], 99)
    mode_order = {"TS":0,"MPS100":1,"MPS30":2,"MIGsamepart":3,"MIGcrosspart":4}.get(p[1], 99)
    cond_order = {"alone":0,"nrx":1}.get(p[2] if len(p)>2 else "", 99)
    return (wl_order, mode_order, cond_order, k)

print(f"{'condition':<35} {'n':>2} {'cudaFree':>10} {'#free':>7} {'memcpy':>10} {'malloc':>10} {'#launch':>8} {'TOTAL host':>12}")
print("-" * 110)
for k in sorted(summary.keys(), key=sk):
    s = summary[k]
    print(f"{k:<35} {s['n']:>2} {s['cudaFree_ms']:>10.0f} {s['cudaFree_n']:>7.0f} "
          f"{s['memcpyAsync_ms']:>10.0f} {s['cudaMalloc_ms']:>10.0f} "
          f"{s['launchKernel_n']:>8.0f} {s['total_host_ms']:>12.0f}")

out = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain11_summary.json"
with open(out,"w") as f: json.dump(summary, f, indent=2)
print(f"\nSaved: {out}")
