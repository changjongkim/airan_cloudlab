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

# ---- full no-MIG condition list (all 14, ordered light->heavy) for the headline ----
NM_ALL = [
    ("alone","F_0_alone"),("xApp","F_E_xapp"),("qwen","F_E_qwen_small"),
    ("H2D memcpy","F_C_H2D_256MB_str4"),("chanpred","F_E_chanpred_b64"),
    ("forecaster","F_E_forecaster_d384"),("NeuralRx","F_E_neuralrx"),
    ("GEMM","F_D_GEMM_4096"),("sat_compute","F_E_sat_compute"),("ResNet","F_E_resnet_b64"),
    ("sat_hbm","F_E_sat_hbm"),("D2D memcpy","F_B_D2D_256MB_str4"),
    ("ResNet x2","F_F_stack_resnet_x2"),("kitchen","F_G_kitchen"),
    ("chanpred x4","F_F_stack_chanpred_x4"),
]
nm_all = [(d, p99(NOMIG,f"realL1_{l}_run*.json")) for d,l in NM_ALL]
coloc_lo, coloc_hi = 356.0, 371.0   # MIG same-partition coloc range (4g..2g), worst case

# ===================== FIG 1: no-MIG vs MIG WORST case (same-partition coloc) =====================
fig,ax=plt.subplots(figsize=(13,6.5))
labs=[d[0] for d in nm_all]; x=np.arange(len(labs)); vals=[d[1] for d in nm_all]
bars=ax.bar(x,vals,0.62,color=C_NM,label="no-MIG full GPU (time-slice), measured")
# MIG WORST case (same-partition coloc) — the realistic MIG scenario, prominent
ax.axhspan(coloc_lo,coloc_hi,color=C_CO,alpha=0.18)
ax.axhline(coloc_nrx,color=C_CO,ls="--",lw=2.5,label=f"MIG WORST: same-partition coloc ~{coloc_nrx:.0f}ms (+ 3g bistable)")
# MIG BEST case (cross-partition isolated) — only if perfectly isolated
ax.axhline(xp_alone,color=C_X,ls=":",lw=2,label=f"MIG best: cross-partition isolated ~{xp_alone:.0f}ms")
ax.set_yscale("log"); ax.set_ylabel("L1 frame-time p99 (ms, log)")
ax.set_title("no-MIG vs MIG: MIG's realistic (worst) case is same-partition coloc ~356ms, not the isolated 45ms")
ax.set_xticks(x); ax.set_xticklabels(labs,rotation=40,ha="right")
for xi,v in zip(x,vals):
    if v: ax.text(xi,v*1.05,f"{v:.0f}",ha="center",fontsize=8,color=C_NM)
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(f"{OUT}/figF1_mig_vs_nomig_p99.png"); plt.close(fig)

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
ax.axhline(coloc_nrx/xp_alone, color=C_CO, ls=":", lw=2,
           label=f"MIG same-part coloc (~{coloc_nrx/xp_alone:.0f}x — MIG is NOT always flat)")
ax.set_ylabel("L1 p99 / own alone baseline")
ax.set_title("MIG cross-part stays flat; but MIG coloc jumps ~6x and no-MIG up to ~11x")
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

# ===================== FIG 7: MIG is NOT flat — placement & bistability =====================
fig,(axL,axR)=plt.subplots(1,2,figsize=(15,6),gridspec_kw={"width_ratios":[1.1,1]})
# left: MIG L1 p99 by placement
mig_xpart = xp_alone  # ~45-59 representative (use chanpred cross-part instead for "under AI")
mig_xpart_ai = p99(XPART,"realL1_F_E_chanpred_b64_run*.json")
c4 = p99(COLOC,"realL1_G_1b_4g_coloc_run*.json")
c2 = p99(COLOC,"realL1_G_1c_2g_coloc_run*.json")
names=["cross-part\n+AI (3g)","coloc 4g\n+NeuralRx","coloc 2g\n+NeuralRx","coloc 3g\n+NeuralRx\n(bistable)"]
vals=[mig_xpart_ai, c4, c2, p99(COLOC,"realL1_G_1a_3g_coloc_run*.json")]
cols=[C_X,C_CO,C_CO,C_CO]
b=axL.bar(names,vals,color=cols)
for bb,v in zip(b,vals): axL.text(bb.get_x()+bb.get_width()/2,v+6,f"{v:.0f}",ha="center",fontsize=10)
axL.set_ylabel("MIG L1 frame-time p99 (ms)")
axL.set_title("MIG is NOT flat: depends entirely on placement\n(cross-part isolates; same-part coloc 6-8x worse)")
# right: 3g coloc per-run bistability
runs=[]
for f in sorted(glob.glob(f"{COLOC}/realL1_G_1a_3g_coloc_run*.json")):
    d=json.load(open(f)); runs.append(np.percentile(d["raw_ms"],99))
axR.bar(range(1,len(runs)+1),runs,color=[C_X if r<100 else C_CO for r in runs])
axR.axhline(100,color="gray",ls="--",lw=0.8)
axR.set_xlabel("run #"); axR.set_ylabel("L1 p99 (ms)")
axR.set_title("Same MIG 3g coloc config, run-to-run BISTABLE\n(7/10 runs ~360ms, 3/10 runs ~45ms)")
fig.tight_layout(); fig.savefig(f"{OUT}/figF7_mig_not_flat_bistable.png"); plt.close(fig)

print("MIG not-flat: xpart+AI", round(mig_xpart_ai,1), "coloc4g", round(c4,1), "coloc2g", round(c2,1))
print("OK. baselines: no-MIG alone", round(nm_alone,1), "| MIG xpart alone", round(xp_alone,1),
      "| MIG coloc alone", round(coloc_alone,1), "| MIG coloc+nrx", round(coloc_nrx,1),
      "| no-MIG+nrx", round(nrx_nomig,1))
for f in sorted(glob.glob(f"{OUT}/figF*.png")): print("  ",os.path.basename(f))
