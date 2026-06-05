#!/usr/bin/env python3
"""
Figures for Perlmutter no-MIG vs CloudLab MIG (4-antenna, 20-cell, cells=20).
Three regimes, all from the 20260601 4-ant campaign + our no-MIG:
  - MIG cross-partition (F_saturation): L1 on 3g, generic AI on separate 2g (isolated)
  - MIG same-partition coloc (G_coloc):  L1 + NeuralRx on the SAME 3g/4g partition
  - no-MIG (ours): full A100, L1 + AI time-sliced on the same GPU
Run: shifter --image=<aerial> <venv>/bin/python build_perlmutter_figures.py
"""
import json, glob, os
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, matplotlib as mpl
mpl.rcParams.update({"font.size":11,"savefig.dpi":145,"savefig.bbox":"tight",
                     "axes.grid":True,"grid.alpha":0.3,"font.family":["DejaVu Sans","sans-serif"]})

R = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results"
HANDOFF = f"{R}/perlmutter_handoff"
NOMIG = f"{HANDOFF}/perlmutter_nomig/F_nomig"
XPART = f"{R}/20260601/F_saturation"     # MIG cross-partition (isolated)
COLOC = f"{R}/20260601/G_coloc"          # MIG same-partition coloc
OUT = f"{HANDOFF}/figures"; os.makedirs(OUT, exist_ok=True)

C_X="#2c7fb8"; C_CO="#7b3294"; C_NM="#d7301f"; C_BASE="#888888"

def p99(d, pat):
    ms=[]
    for f in glob.glob(os.path.join(d, pat)):
        try: ms += json.load(open(f))["raw_ms"]
        except: pass
    return float(np.percentile(ms,99)) if ms else None
def raw(d, pat):
    ms=[]
    for f in glob.glob(os.path.join(d, pat)):
        try: ms += json.load(open(f))["raw_ms"]
        except: pass
    return np.array(ms) if ms else None

# ---- 10 conditions that exist in BOTH no-MIG and MIG cross-partition (clean pairs) ----
PAIRS = [  # (disp, nomig label, xpart label)
    ("alone",       "F_0_alone",            "F_0_alone"),
    ("chanpred",    "F_E_chanpred_b64",     "F_E_chanpred_b64"),
    ("H2D memcpy",  "F_C_H2D_256MB_str4",   "F_C_H2D_sz256_str4"),
    ("forecaster",  "F_E_forecaster_d384",  "F_E_forecaster_d512"),
    ("GEMM",        "F_D_GEMM_4096",        "F_D_GEMM_d4096"),
    ("ResNet",      "F_E_resnet_b64",       "F_E_resnet_b64"),
    ("D2D memcpy",  "F_B_D2D_256MB_str4",   "F_B_D2D_sz256_str4"),
    ("ResNet x2",   "F_F_stack_resnet_x2",  "F_F_stack_resnet_x2"),
    ("kitchen",     "F_G_kitchen",          "F_G_kitchen_all"),
    ("chanpred x4", "F_F_stack_chanpred_x4","F_F_stack_chanpred_x4"),
]
data=[]
for disp,nm,xp in PAIRS:
    data.append((disp, p99(NOMIG,f"realL1_{nm}_run*.json"), p99(XPART,f"realL1_{xp}_run*.json")))

nm_alone = data[0][1]; xp_alone = data[0][2]
# MIG same-partition coloc canonical (L1+NeuralRx): 4g stable
coloc_nrx = p99(COLOC, "realL1_G_1b_4g_coloc_run*.json")
coloc_alone = p99(COLOC, "realL1_G_0b_4g_alone_run*.json")

# ===================== FIG 1: no-MIG vs MIG cross-partition (paired, clean) =====================
fig,ax=plt.subplots(figsize=(12,6))
labs=[d[0] for d in data]; x=np.arange(len(labs)); w=0.4
xp=[d[2] for d in data]; nm=[d[1] for d in data]
ax.bar(x-w/2,xp,w,label="MIG cross-partition (AI on separate slice)",color=C_X)
ax.bar(x+w/2,nm,w,label="no-MIG full GPU (time-slice)",color=C_NM)
ax.set_yscale("log"); ax.set_ylabel("L1 frame-time p99 (ms, log)")
ax.set_title("MIG cross-partition isolates L1; no-MIG does not (4-ant, n=5)")
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=35,ha="right")
for xi,v in zip(x-w/2,xp): ax.text(xi,v*1.05,f"{v:.0f}",ha="center",fontsize=8,color=C_X)
for xi,v in zip(x+w/2,nm): ax.text(xi,v*1.05,f"{v:.0f}",ha="center",fontsize=8,color=C_NM)
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/figF1_mig_vs_nomig_p99.png"); plt.close(fig)

# ===================== FIG 2: ratio no-MIG / MIG cross-part =====================
fig,ax=plt.subplots(figsize=(10,6))
rr=sorted([(d[0],d[1]/d[2]) for d in data if d[1] and d[2]],key=lambda t:t[1])
labs=[t[0] for t in rr]; vals=[t[1] for t in rr]
b=ax.barh(labs,vals,color=C_NM); ax.axvline(1,color="gray",ls="--",label="parity")
for bb,v in zip(b,vals): ax.text(v+0.3,bb.get_y()+bb.get_height()/2,f"{v:.1f}x",va="center",fontsize=9)
ax.set_xlabel("no-MIG p99 / MIG cross-partition p99"); ax.set_title("How much worse no-MIG is than MIG isolation")
ax.legend(); fig.tight_layout(); fig.savefig(f"{OUT}/figF2_nomig_over_mig_ratio.png"); plt.close(fig)

# ===================== FIG 3: normalized to own alone baseline =====================
fig,ax=plt.subplots(figsize=(12,6))
x=np.arange(len(data)); labs=[d[0] for d in data]
ax.plot(x,[d[2]/xp_alone for d in data],"o-",color=C_X,label="MIG cross-part (vs MIG alone)")
ax.plot(x,[d[1]/nm_alone for d in data],"s-",color=C_NM,label="no-MIG (vs no-MIG alone)")
ax.axhline(1,color="gray",ls="--",lw=0.8); ax.set_yscale("log")
ax.set_ylabel("L1 p99 / own alone baseline"); ax.set_title("Isolation: MIG stays ~flat, no-MIG climbs up to ~11x")
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=35,ha="right"); ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figF3_normalized_contention.png"); plt.close(fig)

# ===================== FIG 4: CDF =====================
fig,axes=plt.subplots(1,3,figsize=(15,5),sharey=True)
for ax,(disp,nmlab,xplab) in zip(axes,[("alone","F_0_alone","F_0_alone"),
                                       ("chanpred","F_E_chanpred_b64","F_E_chanpred_b64"),
                                       ("chanpred x4","F_F_stack_chanpred_x4","F_F_stack_chanpred_x4")]):
    for arr,c,lab in [(raw(XPART,f"realL1_{xplab}_run*.json"),C_X,"MIG cross-part"),
                      (raw(NOMIG,f"realL1_{nmlab}_run*.json"),C_NM,"no-MIG")]:
        if arr is None: continue
        s=np.sort(arr); ax.plot(s,np.arange(1,len(s)+1)/len(s),color=c,label=lab,lw=2)
    ax.set_title(f"CDF: {disp}"); ax.set_xlabel("frame time (ms)"); ax.legend(); ax.grid(alpha=0.3)
axes[0].set_ylabel("CDF"); fig.suptitle("Frame-time distribution shifts right under no-MIG")
fig.tight_layout(); fig.savefig(f"{OUT}/figF4_cdf_key_conditions.png"); plt.close(fig)

# ===================== FIG 5: NeuralRx (real coloc data) =====================
fig,ax=plt.subplots(figsize=(9,6))
nrx_nomig = p99(NOMIG,"realL1_F_E_neuralrx_run*.json")
cases=["L1 alone\n(no-MIG)","L1 alone\n(MIG 4g)","MIG same-part\ncoloc +NeuralRx","no-MIG\n+NeuralRx"]
vals=[nm_alone, coloc_alone, coloc_nrx, nrx_nomig]
cols=[C_BASE,C_X,C_CO,C_NM]
b=ax.bar(cases,vals,color=cols)
for bb,v in zip(b,vals): ax.text(bb.get_x()+bb.get_width()/2,v+6,f"{v:.0f}ms",ha="center",fontsize=10)
ax.set_ylabel("L1 frame-time p99 (ms)")
ax.set_title("NeuralRx co-tenant: no-MIG ~ MIG same-partition coloc (both bad)\n(MIG coloc = CloudLab G_1b 4g; no-MIG = Perlmutter, measured)")
fig.tight_layout(); fig.savefig(f"{OUT}/figF5_neuralrx_focus.png"); plt.close(fig)

# ===================== FIG 6: 3-regime spectrum =====================
fig,ax=plt.subplots(figsize=(12,6))
labs=[d[0] for d in data]; x=np.arange(len(labs))
ax.bar(x,[d[1] for d in data],0.6,color=C_NM,label="no-MIG (time-slice)")
ax.axhline(xp_alone,color=C_X,ls="-",lw=2,label=f"MIG cross-partition isolated (~{xp_alone:.0f}ms)")
ax.axhspan(min(xp for _,_,xp in data), max(xp for _,_,xp in data),color=C_X,alpha=0.12)
ax.axhline(coloc_nrx,color=C_CO,ls="--",lw=2,label=f"MIG same-part coloc +NeuralRx (~{coloc_nrx:.0f}ms)")
ax.set_yscale("log"); ax.set_ylabel("L1 frame-time p99 (ms, log)")
ax.set_title("Three regimes: MIG isolated (best) < MIG coloc ~ no-MIG single < no-MIG stacked (worst)")
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=35,ha="right")
for xi,d in zip(x,data): ax.text(xi,d[1]*1.05,f"{d[1]:.0f}",ha="center",fontsize=8,color=C_NM)
ax.legend(loc="upper left"); fig.tight_layout(); fig.savefig(f"{OUT}/figF6_three_regime_spectrum.png"); plt.close(fig)

print("OK. baselines: no-MIG alone", round(nm_alone,1), "| MIG xpart alone", round(xp_alone,1),
      "| MIG coloc alone", round(coloc_alone,1), "| MIG coloc+nrx", round(coloc_nrx,1),
      "| no-MIG+nrx", round(nrx_nomig,1))
for f in sorted(glob.glob(f"{OUT}/figF*.png")): print("  ",os.path.basename(f))
