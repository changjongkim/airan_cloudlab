#!/usr/bin/env python3
"""Priority 2 MPS analysis: default time-slice vs MPS vs MIG. Figures figF8/figF9."""
import json, glob, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.size":11,"savefig.dpi":145,"savefig.bbox":"tight",
                     "axes.grid":True,"grid.alpha":0.3,"font.family":["DejaVu Sans","sans-serif"]})
R="/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results"
NM=f"{R}/perlmutter_handoff/perlmutter_nomig/F_nomig/"
MP=f"{R}/perlmutter_handoff/perlmutter_nomig/F_nomig_mps/"
XP=f"{R}/20260601/F_saturation/"
OUT=f"{R}/perlmutter_handoff/figures"
C_DEF="#d7301f"; C_MPS="#1a9850"; C_MIG="#2c7fb8"
def agg(d,pat):
    ms=[]
    for f in glob.glob(d+pat):
        try: ms+=json.load(open(f))["raw_ms"]
        except: pass
    return np.array(ms)
rows=[("alone","F_0_alone","F_0_alone"),("chanpred","F_E_chanpred_b64","F_E_chanpred_b64"),
      ("qwen","F_E_qwen_small",None),("forecaster","F_E_forecaster_d384","F_E_forecaster_d512"),
      ("NeuralRx","F_E_neuralrx",None),("resnet","F_E_resnet_b64","F_E_resnet_b64"),
      ("sat_hbm","F_E_sat_hbm",None)]
disp=[];dft=[];mps=[];mig=[]
for d,nl,xl in rows:
    disp.append(d)
    dft.append(np.percentile(agg(NM,f"realL1_{nl}_run*.json"),99))
    mps.append(np.percentile(agg(MP,f"realL1_MPS_{nl}_run*.json"),99))
    mig.append(np.percentile(agg(XP,f"realL1_{xl}_run*.json"),99) if xl else np.nan)

# ---- figF8: default vs MPS vs MIG cross-part ----
fig,ax=plt.subplots(figsize=(13,6.5)); x=np.arange(len(disp)); w=0.27
ax.bar(x-w,dft,w,color=C_DEF,label="no-MIG default (time-slice)")
ax.bar(x,mps,w,color=C_MPS,label="no-MIG MPS")
ax.bar(x+w,mig,w,color=C_MIG,label="MIG cross-partition (isolated)")
ax.set_yscale("log"); ax.set_ylabel("L1 frame-time p99 (ms, log)")
ax.set_title("MPS recovers most contention to ~MIG levels — EXCEPT memory-bandwidth (sat_hbm bistable)")
ax.set_xticks(x); ax.set_xticklabels(disp)
for xi,v in zip(x-w,dft): ax.text(xi,v*1.05,f"{v:.0f}",ha="center",fontsize=8,color=C_DEF)
for xi,v in zip(x,mps):
    ax.text(xi,v*1.05,f"{v:.0f}",ha="center",fontsize=8,color=C_MPS)
for xi,v in zip(x+w,mig):
    if not np.isnan(v): ax.text(xi,v*1.05,f"{v:.0f}",ha="center",fontsize=8,color=C_MIG)
ax.annotate("sat_hbm MPS = 6985ms\n(2/5 runs ~7s, bistable)",xy=(6,6985),xytext=(4.2,2500),
            fontsize=9,color=C_MPS,arrowprops=dict(arrowstyle="->",color=C_MPS))
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/figF8_mps_vs_default_vs_mig.png"); plt.close(fig)

# ---- figF9: sat_hbm MPS bistability per-run ----
fig,ax=plt.subplots(figsize=(8,5))
runs=[]
for f in sorted(glob.glob(MP+"realL1_MPS_F_E_sat_hbm_run*.json")):
    runs.append(np.percentile(json.load(open(f))["raw_ms"],99))
ax.bar(range(1,len(runs)+1),runs,color=[C_MPS if r<500 else C_DEF for r in runs])
ax.set_yscale("log"); ax.set_xlabel("run #"); ax.set_ylabel("L1 p99 (ms, log)")
ax.axhline(426,color="gray",ls="--",lw=1,label="no-MIG default sat_hbm (426ms)")
ax.set_title("MPS + sat_hbm is BISTABLE: 2/5 runs ~7000ms, 3/5 runs ~40ms")
for i,r in enumerate(runs): ax.text(i+1,r*1.1,f"{r:.0f}",ha="center",fontsize=9)
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/figF9_mps_sat_hbm_bistable.png"); plt.close(fig)
print("figF8/figF9 written. MPS p99:",{d:round(m,1) for d,m in zip(disp,mps)})
