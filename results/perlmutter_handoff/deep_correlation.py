#!/usr/bin/env python3
"""#3 correlationId/temporal linking: prove the GPU idle-gap == host blocked in
cudaFree. For each GPU gap interval, measure how much of it is covered by each
host CUDA API call (by name). The dominant-overlap API is the gap's cause."""
import sqlite3, os, sys
import numpy as np
BASE="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig"

def gpu_gaps(con):
    ops=[]
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_MEMSET"):
        for s,e in con.execute(f"SELECT start,end FROM {t}"): ops.append((s,e))
    ops.sort()
    gaps=[]; cur=ops[0][1]
    for s,e in ops[1:]:
        if s>cur: gaps.append((cur,s))
        cur=max(cur,e)
    return gaps  # list of (start,end) idle intervals

def overlap_by_api(con, gaps, top=6):
    # host API intervals by name
    rows=con.execute("""SELECT s.value, r.start, r.end FROM CUPTI_ACTIVITY_KIND_RUNTIME r
                        JOIN StringIds s ON r.nameId=s.id ORDER BY r.start""").fetchall()
    gaps=sorted(gaps)
    gstart=np.array([g[0] for g in gaps]); gend=np.array([g[1] for g in gaps])
    from collections import defaultdict
    ov=defaultdict(float)
    import bisect
    for name,a_s,a_e in rows:
        base=name.split("_v")[0]
        # find gaps overlapping [a_s,a_e]
        i=bisect.bisect_right(gstart,a_e)
        j=bisect.bisect_left(gend,a_s)
        for k in range(j,i):
            lo=max(a_s,gstart[k]); hi=min(a_e,gend[k])
            if hi>lo: ov[base]+=(hi-lo)
    return ov

conds=sys.argv[1:] or ["neuralrx","sat_hbm","resnet"]
for cond in conds:
    for reg,db in [("default",f"{BASE}/nsys_nomig/nsys_{cond}.sqlite"),
                   ("MPS",f"{BASE}/MPS_nsys/MPSnsys_{cond}.sqlite")]:
        if not os.path.exists(db): continue
        con=sqlite3.connect(db); gaps=gpu_gaps(con)
        total_gap=sum(e-s for s,e in gaps)/1e6
        ov=overlap_by_api(con,gaps); con.close()
        print(f"\n== {cond} [{reg}] — total GPU-gap={total_gap:.0f}ms; host API covering the gaps: ==")
        for name,t in sorted(ov.items(),key=lambda x:-x[1])[:6]:
            print(f"   {name:28s} covers {t/1e6:8.0f}ms of gap  ({100*t/1e6/total_gap:5.1f}%)")
