#!/usr/bin/env python3
"""CUDA API (host runtime) breakdown from nsys sqlite — which host-side call
dominates and inflates under contention. default vs MPS. (MIG doc §17.3 style)"""
import sqlite3, os, sys
BASE="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig"
APIS=["cudaFree","cudaMemcpyAsync","cudaMalloc","cudaStreamSynchronize","cudaEventSynchronize","cudaLaunchKernel"]
def api_stats(db):
    if not os.path.exists(db): return None
    c=sqlite3.connect(db); out={}
    q="""SELECT s.value api, COUNT(*) n, SUM(r.end-r.start)/1e6 tot_ms, AVG(r.end-r.start)/1e3 avg_us
         FROM CUPTI_ACTIVITY_KIND_RUNTIME r JOIN StringIds s ON r.nameId=s.id GROUP BY s.value"""
    for api,n,tot,avg in c.execute(q):
        base=api.split("_v")[0]
        for k in APIS:
            if base==k: out[k]=(n,tot,avg)
    c.close(); return out
conds=sys.argv[1:] or ["alone","neuralrx","sat_hbm"]
for cond in conds:
    print(f"\n===== {cond} =====")
    print(f"{'API':24s} {'default tot/avg':>22s}   {'MPS tot/avg':>22s}   {'avg ratio':>9s}")
    d=api_stats(f"{BASE}/nsys_nomig/nsys_{cond}.sqlite")
    m=api_stats(f"{BASE}/MPS_nsys/MPSnsys_{cond}.sqlite")
    if not d or not m: print("  (missing db)"); continue
    for k in APIS:
        dn,dt,da=d.get(k,(0,0,0)); mn,mt,ma=m.get(k,(0,0,0))
        r=(ma/da) if da else 0
        print(f"{k:24s} {dt:9.1f}ms/{da:8.1f}us   {mt:9.1f}ms/{ma:8.1f}us   {r:8.2f}x")
