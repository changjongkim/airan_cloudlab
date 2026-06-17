#!/usr/bin/env python3
"""figF12 (gap p99 default vs MPS) + figF13 (gap fraction) from deep nsys sqlite."""
import sqlite3, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.size":11,"savefig.dpi":145,"savefig.bbox":"tight",
                     "axes.grid":True,"grid.alpha":0.3,"font.family":["DejaVu Sans","sans-serif"]})
BASE="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/perlmutter_nomig"
OUT="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff/figures"
C_DEF="#d7301f"; C_MPS="#1a9850"
def te(c,t): return c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",(t,)).fetchone() is not None
def analyze(db):
    if not os.path.exists(db): return None
    c=sqlite3.connect(db); ops=[]
    for t in ("CUPTI_ACTIVITY_KIND_KERNEL","CUPTI_ACTIVITY_KIND_MEMCPY","CUPTI_ACTIVITY_KIND_MEMSET"):
        if te(c,t):
            for s,e in c.execute(f"SELECT start,end FROM {t}"): ops.append((s,e))
    c.close()
    if not ops: return None
    ops.sort(); starts=np.array([o[0] for o in ops]); ends=np.array([o[1] for o in ops])
    busy=(ends-starts).sum(); wall=ends.max()-starts.min()
    gaps=[]; cur=ends[0]
    for s,e in zip(starts[1:],ends[1:]):
        if s>cur: gaps.append(s-cur)
        cur=max(cur,e)
    gaps=np.array(gaps) if gaps else np.array([0])
    return {"gap_frac":1-busy/wall if wall else 0,"gap_p99_us":np.percentile(gaps,99)/1e3}

# order: compute AIs then memory AIs
COMPUTE=["alone","qwen","xapp","chanpred","forecaster","neuralrx","resnet","3AI","2AI","resnet_fore"]
MEMORY=["sat_compute","sat_hbm","2sat","3sat"]
order=COMPUTE+MEMORY
res={}
for c in order:
    res[c]={"default":analyze(f"{BASE}/nsys_nomig/nsys_{c}.sqlite"),
            "MPS":analyze(f"{BASE}/MPS_nsys/MPSnsys_{c}.sqlite")}

# ---- figF12: gap p99 default vs MPS (log) ----
fig,ax=plt.subplots(figsize=(14,6.5)); x=np.arange(len(order)); w=0.4
dv=[res[c]["default"]["gap_p99_us"] if res[c]["default"] else np.nan for c in order]
mv=[res[c]["MPS"]["gap_p99_us"] if res[c]["MPS"] else np.nan for c in order]
ax.bar(x-w/2,dv,w,color=C_DEF,label="no-MIG default (time-slice)")
ax.bar(x+w/2,mv,w,color=C_MPS,label="no-MIG MPS")
ax.set_yscale("log"); ax.set_ylabel("worst inter-kernel gap  p99 (us, log)")
ax.axhline(251,color="gray",ls="--",lw=1,label="alone baseline (~251us)")
ax.axvline(len(COMPUTE)-0.5,color="black",ls=":",lw=1.5)
ax.text(len(COMPUTE)/2-0.5,3e5,"COMPUTE-bound AI",ha="center",fontsize=10,color="#1a9850")
ax.text(len(COMPUTE)+len(MEMORY)/2-0.5,3e5,"MEMORY-bound AI",ha="center",fontsize=10,color="#d7301f")
ax.set_title("MPS eliminates queue gaps for COMPUTE AI (recovery) but EXPLODES them for MEMORY AI (collapse)")
ax.set_xticks(x); ax.set_xticklabels(order,rotation=40,ha="right"); ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/figF12_deep_gap_p99.png"); plt.close(fig)

# ---- figF13: gap fraction ----
fig,ax=plt.subplots(figsize=(14,6))
df=[res[c]["default"]["gap_frac"]*100 if res[c]["default"] else np.nan for c in order]
mf=[res[c]["MPS"]["gap_frac"]*100 if res[c]["MPS"] else np.nan for c in order]
ax.bar(x-w/2,df,w,color=C_DEF,label="default"); ax.bar(x+w/2,mf,w,color=C_MPS,label="MPS")
ax.axvline(len(COMPUTE)-0.5,color="black",ls=":",lw=1.5)
ax.set_ylabel("L1 timeline idle-gap fraction (%)")
ax.set_title("Idle-gap fraction: MPS restores compute AI to ~baseline; memory AI stays maxed")
ax.set_xticks(x); ax.set_xticklabels(order,rotation=40,ha="right"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figF13_deep_gap_fraction.png"); plt.close(fig)
print("figF12/figF13 written.")
for c in order: print(f"  {c:12s} default gap_p99={dv[order.index(c)]:9.0f}us  MPS={mv[order.index(c)]:9.0f}us")
