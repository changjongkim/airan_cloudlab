#!/usr/bin/env python3
"""Critical nsys SQLite analyses matching MIG doc §13/§15/§16.1:
  A. 60KB (61440 B) memcpy per-call duration distribution -> BIMODAL queue signature
  B. per-kernel POST-gap -> which kernel boundary the queue wait sits at (§13)
  C. memcpy direction (copyKind) breakdown (§15)
Compares default vs MPS for alone/neuralrx/sat_hbm."""
import sqlite3, os, sys
import numpy as np
BASE="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig"
COPYKIND={1:"H2D",2:"D2H",8:"D2D",10:"H2H"}  # CUPTI enum

def dbpath(cond,regime):
    return (f"{BASE}/nsys_nomig/nsys_{cond}.sqlite" if regime=="default"
            else f"{BASE}/MPS_nsys/MPSnsys_{cond}.sqlite")

def memcpy_60k(db):
    c=sqlite3.connect(db)
    d=np.array([e-s for s,e in c.execute("SELECT start,end FROM CUPTI_ACTIVITY_KIND_MEMCPY WHERE bytes=61440")])
    c.close(); return d/1e3  # us

def memcpy_by_dir(db):
    c=sqlite3.connect(db); out={}
    for ck,n,avg in c.execute("SELECT copyKind,COUNT(*),AVG(end-start)/1e3 FROM CUPTI_ACTIVITY_KIND_MEMCPY GROUP BY copyKind"):
        out[COPYKIND.get(ck,ck)]=(n,avg)
    c.close(); return out

def post_gap_per_kernel(db, topn=6):
    c=sqlite3.connect(db)
    rows=list(c.execute("""SELECT k.start,k.end,COALESCE(sd.value,ss.value) name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k LEFT JOIN StringIds sd ON k.demangledName=sd.id
        LEFT JOIN StringIds ss ON k.shortName=ss.id ORDER BY k.start"""))
    c.close()
    from collections import defaultdict
    gaps=defaultdict(list); tot=defaultdict(float); cnt=defaultdict(int)
    for i in range(len(rows)-1):
        s,e,name=rows[i]; ns=rows[i+1][0]; name=(name or "?").split("<")[0][:40]
        tot[name]+=(e-s); cnt[name]+=1
        if ns>e: gaps[name].append(ns-e)
    top=sorted(tot, key=lambda k:-tot[k])[:topn]
    return [(k, cnt[k], tot[k]/1e6,
             np.percentile(gaps[k],50)/1e3 if gaps[k] else 0,
             np.percentile(gaps[k],99)/1e3 if gaps[k] else 0) for k in top]

conds=sys.argv[1:] or ["alone","neuralrx","sat_hbm"]
print("="*70); print("A. 60KB memcpy per-call duration (us) — BIMODAL queue signature (§16.1)")
print("="*70)
print(f"{'cond':10s} {'regime':8s} {'n':>5s} {'p10':>6s} {'p50':>6s} {'p90':>7s} {'p99':>8s} {'%>8us':>7s} {'%>15us':>7s}")
for cond in conds:
    for reg in ("default","MPS"):
        db=dbpath(cond,reg)
        if not os.path.exists(db): continue
        d=memcpy_60k(db)
        if len(d)==0: print(f"{cond:10s} {reg:8s}  (no 60KB)"); continue
        slow8=100*np.mean(d>8); slow15=100*np.mean(d>15)
        print(f"{cond:10s} {reg:8s} {len(d):5d} {np.percentile(d,10):6.2f} {np.percentile(d,50):6.2f} "
              f"{np.percentile(d,90):7.2f} {np.percentile(d,99):8.2f} {slow8:6.1f}% {slow15:6.1f}%")
    print()

print("="*70); print("B. memcpy direction breakdown (count, avg us) (§15)")
print("="*70)
for cond in conds:
    for reg in ("default","MPS"):
        db=dbpath(cond,reg)
        if not os.path.exists(db): continue
        dd=memcpy_by_dir(db)
        s=" ".join(f"{k}:{v[0]}@{v[1]:.1f}us" for k,v in sorted(dd.items(), key=lambda x: str(x[0])))
        print(f"  {cond:10s} {reg:8s}  {s}")
    print()

print("="*70); print("C. per-kernel POST-gap — where the queue wait sits (§13)")
print("="*70)
for cond in conds:
    print(f"--- {cond} ---")
    print(f"  {'kernel':42s} {'regime':8s} {'n':>5s} {'busy_ms':>7s} {'gap_med':>8s} {'gap_p99':>9s}")
    for reg in ("default","MPS"):
        db=dbpath(cond,reg)
        if not os.path.exists(db): continue
        for name,n,ms,gm,gp in post_gap_per_kernel(db):
            print(f"  {name:42s} {reg:8s} {n:5d} {ms:7.1f} {gm:7.1f}u {gp:8.1f}u")
    print()
