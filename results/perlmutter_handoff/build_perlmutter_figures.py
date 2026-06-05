#!/usr/bin/env python3
"""
Figures for the Perlmutter no-MIG vs CloudLab MIG comparison.
Run inside the container with the venv python (matplotlib installed there):
  shifter --image=<aerial> <venv>/bin/python build_perlmutter_figures.py
"""
import json, glob, os
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.rcParams.update({
    "font.size": 11, "figure.figsize": (11, 6), "savefig.dpi": 145,
    "savefig.bbox": "tight", "axes.grid": True, "grid.alpha": 0.3,
    "font.family": ["DejaVu Sans", "sans-serif"],
})

HANDOFF = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/perlmutter_handoff"
NOMIG = f"{HANDOFF}/perlmutter_nomig/F_nomig"
MIG   = "/pscratch/sd/s/sgkim/kcj/airan_cloudlab/results/20260601/F_saturation"
OUT = f"{HANDOFF}/figures"; os.makedirs(OUT, exist_ok=True)

MIG_C   = "#2c7fb8"   # blue  = MIG (isolated)
NOMIG_C = "#d7301f"   # red   = no-MIG (time-slice)

# (display, no-MIG label, MIG label)
MAP = [
    ("alone",        "F_0_alone",             "F_0_alone"),
    ("chanpred",     "F_E_chanpred_b64",      "F_E_chanpred_b64"),
    ("H2D memcpy",   "F_C_H2D_256MB_str4",    "F_C_H2D_sz256_str4"),
    ("qwen",         "F_E_qwen_small",        None),
    ("xApp",         "F_E_xapp",              None),
    ("forecaster",   "F_E_forecaster_d384",   "F_E_forecaster_d512"),
    ("NeuralRx",     "F_E_neuralrx",          None),
    ("GEMM",         "F_D_GEMM_4096",         "F_D_GEMM_d4096"),
    ("sat_compute",  "F_E_sat_compute",       None),
    ("ResNet",       "F_E_resnet_b64",        "F_E_resnet_b64"),
    ("sat_hbm",      "F_E_sat_hbm",           None),
    ("D2D memcpy",   "F_B_D2D_256MB_str4",    "F_B_D2D_sz256_str4"),
    ("ResNet x2",    "F_F_stack_resnet_x2",   "F_F_stack_resnet_x2"),
    ("kitchen",      "F_G_kitchen",           "F_G_kitchen_all"),
    ("chanpred x4",  "F_F_stack_chanpred_x4", "F_F_stack_chanpred_x4"),
]

def load(d, cond):
    if cond is None: return None
    ms = []
    for f in glob.glob(os.path.join(d, f"realL1_{cond}_run*.json")):
        ms += json.load(open(f))["raw_ms"]
    return np.array(ms) if ms else None

rows = []
for disp, nm, mg in MAP:
    n = load(NOMIG, nm); m = load(MIG, mg)
    rows.append({
        "disp": disp,
        "nm_p50": np.percentile(n,50) if n is not None else None,
        "nm_p99": np.percentile(n,99) if n is not None else None,
        "mg_p50": np.percentile(m,50) if m is not None else None,
        "mg_p99": np.percentile(m,99) if m is not None else None,
        "n": n, "m": m,
    })

nm_alone = next(r for r in rows if r["disp"]=="alone")["nm_p99"]
mg_alone = next(r for r in rows if r["disp"]=="alone")["mg_p99"]

# ---------------------------------------------------------------- FIG 1: grouped p99 bars (log)
fig, ax = plt.subplots(figsize=(13,6))
labels = [r["disp"] for r in rows]
x = np.arange(len(labels)); w = 0.4
mg_vals = [r["mg_p99"] if r["mg_p99"] else np.nan for r in rows]
nm_vals = [r["nm_p99"] if r["nm_p99"] else np.nan for r in rows]
ax.bar(x-w/2, mg_vals, w, label="MIG cross-partition (isolated)", color=MIG_C)
ax.bar(x+w/2, nm_vals, w, label="no-MIG full GPU (time-slice)", color=NOMIG_C)
ax.set_yscale("log")
ax.set_ylabel("L1 frame-time p99 (ms, log scale)")
ax.set_title("MIG vs no-MIG: L1 p99 under co-located AI (20-cell frame, n=5)")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
ax.axhline(1.0, color="gray", lw=0.5)
for xi, v in zip(x+w/2, nm_vals):
    if not np.isnan(v): ax.text(xi, v*1.05, f"{v:.0f}", ha="center", fontsize=8, color=NOMIG_C)
for xi, v in zip(x-w/2, mg_vals):
    if not np.isnan(v): ax.text(xi, v*1.05, f"{v:.0f}", ha="center", fontsize=8, color=MIG_C)
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figF1_mig_vs_nomig_p99.png"); plt.close(fig)

# ---------------------------------------------------------------- FIG 2: no-MIG/MIG ratio
fig, ax = plt.subplots(figsize=(10,6))
rr = [(r["disp"], r["nm_p99"]/r["mg_p99"]) for r in rows if r["mg_p99"] and r["nm_p99"]]
rr.sort(key=lambda t: t[1])
labs=[t[0] for t in rr]; vals=[t[1] for t in rr]
bars=ax.barh(labs, vals, color=NOMIG_C)
ax.set_xlabel("no-MIG p99 / MIG p99  (x worse without MIG)")
ax.set_title("How much worse is no-MIG than MIG (same condition)")
ax.axvline(1.0, color="gray", ls="--", label="parity")
for b,v in zip(bars,vals): ax.text(v+0.3, b.get_y()+b.get_height()/2, f"{v:.1f}x", va="center", fontsize=9)
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figF2_nomig_over_mig_ratio.png"); plt.close(fig)

# ---------------------------------------------------------------- FIG 3: normalized to own baseline (isolation story)
fig, ax = plt.subplots(figsize=(13,6))
mg_norm = [ (r["mg_p99"]/mg_alone) if r["mg_p99"] else np.nan for r in rows]
nm_norm = [ (r["nm_p99"]/nm_alone) if r["nm_p99"] else np.nan for r in rows]
ax.plot(x, mg_norm, "o-", color=MIG_C, label="MIG (vs MIG-alone)")
ax.plot(x, nm_norm, "s-", color=NOMIG_C, label="no-MIG (vs no-MIG-alone)")
ax.axhline(1.0, color="gray", ls="--", lw=0.8)
ax.set_yscale("log")
ax.set_ylabel("L1 p99 normalized to each platform's own alone baseline")
ax.set_title("Isolation story: MIG stays ~flat, no-MIG degrades up to ~11x")
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=40, ha="right")
ax.legend()
fig.tight_layout(); fig.savefig(f"{OUT}/figF3_normalized_contention.png"); plt.close(fig)

# ---------------------------------------------------------------- FIG 4: CDF for key conditions
fig, axes = plt.subplots(1, 3, figsize=(15,5), sharey=True)
for ax, disp in zip(axes, ["alone", "chanpred", "chanpred x4"]):
    r = next(rr for rr in rows if rr["disp"]==disp)
    for arr, c, lab in [(r["m"], MIG_C, "MIG"), (r["n"], NOMIG_C, "no-MIG")]:
        if arr is None: continue
        s = np.sort(arr); cdf = np.arange(1,len(s)+1)/len(s)
        ax.plot(s, cdf, color=c, label=lab, lw=2)
    ax.set_title(f"L1 frame-time CDF: {disp}")
    ax.set_xlabel("frame time (ms)"); ax.grid(alpha=0.3); ax.legend()
axes[0].set_ylabel("CDF")
fig.suptitle("Frame-time distribution: MIG isolates, no-MIG shifts right under load")
fig.tight_layout(); fig.savefig(f"{OUT}/figF4_cdf_key_conditions.png"); plt.close(fig)

# ---------------------------------------------------------------- FIG 5: NeuralRx focus (paper headline)
fig, ax = plt.subplots(figsize=(9,6))
nrx_nomig = next(r for r in rows if r["disp"]=="NeuralRx")["nm_p99"]
# MIG reference numbers from prior CloudLab analysis (BRIEFING / visual_evidence)
cases = ["L1 alone\n(no-MIG)", "MIG x-part\n+NeuralRx", "MIG coloc\n+NeuralRx", "no-MIG\n+NeuralRx"]
vals  = [nm_alone, 196.0, 357.0, nrx_nomig]
cols  = ["gray", MIG_C, "#7b3294", NOMIG_C]
bars = ax.bar(cases, vals, color=cols)
for b,v in zip(bars,vals): ax.text(b.get_x()+b.get_width()/2, v+8, f"{v:.0f}ms", ha="center", fontsize=10)
ax.set_ylabel("L1 frame-time p99 (ms)")
ax.set_title("NeuralRx (PHY-AI) — the critical co-tenant\n(MIG x-part/coloc from CloudLab; no-MIG measured on Perlmutter)")
fig.tight_layout(); fig.savefig(f"{OUT}/figF5_neuralrx_focus.png"); plt.close(fig)

print("wrote figures to", OUT)
for f in sorted(glob.glob(f"{OUT}/figF*.png")): print("  ", os.path.basename(f))
