#!/usr/bin/env python3
"""Chain 12 analysis — combined A/B/C approach comparison."""
import sqlite3, glob, os, json, statistics
from collections import defaultdict

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain12"

def stats(sq):
    con = sqlite3.connect(sq); cur = con.cursor()
    try:
        def like(pat):
            cur.execute(f"SELECT id FROM StringIds WHERE value LIKE '{pat}'")
            return [r[0] for r in cur.fetchall()]

        cf   = like('cudaFree_v____')
        cfa  = like('cudaFreeAsync%')
        mca  = like('cudaMemcpyAsync%')
        cma  = like('cudaMalloc_v____')
        clk  = like('cuLaunchKernel')
        cdlk = like('cudaLaunchKernel_v____')

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
        cdlk_ns, cdlk_n = sumns(cdlk)
        total_launches = clk_n + cdlk_n

        cur.execute("SELECT SUM(end-start) FROM CUPTI_ACTIVITY_KIND_RUNTIME")
        total_ns = cur.fetchone()[0] or 0

        # GPU kernel activity
        try:
            cur.execute("SELECT COUNT(*), SUM(end-start) FROM CUPTI_ACTIVITY_KIND_KERNEL")
            r = cur.fetchone()
            gpu_kern_ns = r[1] or 0
        except Exception:
            gpu_kern_ns = 0
    except Exception:
        con.close(); return None
    con.close()
    return {
        "cudaFree_ms": cf_ns/1e6, "cudaFree_n": cf_n,
        "cudaFreeAsync_ms": cfa_ns/1e6,
        "memcpyAsync_ms": mca_ns/1e6,
        "cudaMalloc_ms": cma_ns/1e6,
        "launches_n": total_launches,
        "total_host_ms": total_ns/1e6,
        "gpu_kernel_ms": gpu_kern_ns/1e6,
    }

def parse(fname):
    base = os.path.basename(fname).replace(".sqlite","")
    for suf in ["_t1","_t2","_t3"]:
        if base.endswith(suf): base = base[:-3]; break
    return base

r = defaultdict(list)
for sq in sorted(glob.glob(f"{BASE}/*.sqlite")):
    s = stats(sq)
    if s is None or s["total_host_ms"] == 0: continue
    r[parse(sq)].append(s)

summary = {}
for k, ts in r.items():
    def avg(f): return statistics.mean(x[f] for x in ts)
    summary[k] = {"n": len(ts),
                  "cudaFree_ms": avg("cudaFree_ms"),
                  "cudaFree_n": avg("cudaFree_n"),
                  "memcpyAsync_ms": avg("memcpyAsync_ms"),
                  "cudaMalloc_ms": avg("cudaMalloc_ms"),
                  "launches_n": avg("launches_n"),
                  "total_host_ms": avg("total_host_ms"),
                  "gpu_kernel_ms": avg("gpu_kernel_ms")}

def sk(k):
    p = k.split("_")
    wl = p[0]
    if wl == "baseL1" and len(p) > 1 and p[1] == "arena": wl = "baseL1_arena"; mode = p[2]; cond = p[3]
    else: mode = p[1]; cond = p[2]
    wl_o = {"baseL1":0,"baseL1_arena":1,"multiBase":2,"multiMega":3,"persistMega":4}.get(wl,99)
    m_o  = {"TS":0,"MPS100":1}.get(mode,99)
    c_o  = {"alone":0,"nrx":1}.get(cond,99)
    return (m_o, wl_o, c_o, k)

print(f"{'condition':<36} {'n':>2} {'cudaFree':>10} {'#free':>7} {'memcpy':>10} {'malloc':>10} {'#lnch':>8} {'GPUkern':>10} {'TOTAL':>10}")
print("-" * 120)
for k in sorted(summary.keys(), key=sk):
    s = summary[k]
    print(f"{k:<36} {s['n']:>2} {s['cudaFree_ms']:>10.0f} {s['cudaFree_n']:>7.0f} "
          f"{s['memcpyAsync_ms']:>10.0f} {s['cudaMalloc_ms']:>10.0f} "
          f"{s['launches_n']:>8.0f} {s['gpu_kernel_ms']:>10.0f} {s['total_host_ms']:>10.0f}")

out = "/Users/changjongkim/New_research/cloudlab_results/results/20260715/chain12_summary.json"
with open(out,"w") as f: json.dump(summary, f, indent=2)
print(f"\nSaved: {out}")
