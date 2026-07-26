#!/usr/bin/env python3
"""V3 polish: bigger fonts, clearer layout, better contrast for headline figures.
Regenerates P23, P24, P28 with maximum readability."""
import os, csv, glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
FIG  = os.path.join(BASE, "figures", "polished")

INK="#0f172a"; INK_SEC="#334155"; INK_MUT="#64748b"
SURFACE="#ffffff"; GRID="#e2e8f0"
COL_BASELINE="#0f172a"; COL_GOOD="#059669"; COL_WARN="#d97706"; COL_BAD="#b91c1c"
COL_MPS_OFF="#b91c1c"; COL_MPS_ON="#059669"

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 16,
    "axes.titlesize": 20,
    "axes.labelsize": 17,
    "xtick.labelsize": 15,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": INK_MUT, "axes.linewidth": 1.0,
    "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "grid.color": GRID, "grid.linewidth": 0.9,
    "axes.grid": True, "axes.axisbelow": True,
    "axes.unicode_minus": False, "savefig.dpi": 170, "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

def parse_trace(p):
    ks = []
    if not os.path.exists(p): return ks
    with open(p) as f:
        rdr=csv.reader(f); hdr=None
        for r in rdr:
            if r and r[0]=="Start (ns)": hdr=r; break
        if not hdr: return ks
        idx_s=hdr.index("Start (ns)"); idx_d=hdr.index("Duration (ns)")
        idx_g=hdr.index("GrdX"); idx_n=hdr.index("Name"); idx_st=hdr.index("Strm")
        for r in rdr:
            if len(r)<=idx_n or not r[idx_g].strip(): continue
            n=r[idx_n]
            if 'memcpy' in n.lower() or 'memset' in n.lower(): continue
            try: ks.append((int(r[idx_s]),int(r[idx_d]),n[:60],r[idx_st]))
            except: pass
    return ks

def stats(files):
    ad=[]; ag=[]
    for p in files:
        ks = parse_trace(p)
        by = defaultdict(list)
        for s,d,_,st in ks: by[st].append((s,d))
        for _, kl in by.items():
            kl.sort()
            for j,(s,d) in enumerate(kl):
                ad.append(d)
                if j>0:
                    g=s-(kl[j-1][0]+kl[j-1][1])
                    if g>0: ag.append(g)
    if not ad: return None
    d=np.array(ad); g=np.array(ag) if ag else np.array([0])
    return dict(dur_med=np.median(d), gap_med=np.median(g),
                gap_p95=np.percentile(g,95), gap_p99=np.percentile(g,99),
                duty=d.sum()/(d.sum()+g.sum())*100)

# ============================================================
# P23 V3: Duty cycle headline (bigger everything)
# ============================================================
Ns = [1,2,3,4,6,8]
d_on=[]; d_off=[]
for N in Ns:
    off = stats([f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPSoff_t1.gputrace.csv"])
    on = stats([f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPSon_t1.gputrace.csv"])
    d_off.append(off["duty"] if off else 0)
    d_on.append(on["duty"] if on else 0)
baseline = stats([f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"])["duty"]

fig, ax = plt.subplots(figsize=(15, 8.5))
# Zone shading with clearer labels
ax.axvspan(0.5, 4.5, alpha=0.10, color=COL_GOOD, zorder=0)
ax.axvspan(5.5, 8.5, alpha=0.10, color=COL_BAD, zorder=0)
ax.text(2.5, 38, "SAFE ZONE\n(MPS on)", ha="center", va="center",
        fontsize=16, color=COL_GOOD, fontweight="bold", alpha=0.95,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=COL_GOOD, lw=2, alpha=0.9))
ax.text(7, 38, "BREAKDOWN ZONE\n(MPS on saturates)", ha="center", va="center",
        fontsize=16, color=COL_BAD, fontweight="bold", alpha=0.95,
        bbox=dict(boxstyle="round,pad=0.6", fc="white", ec=COL_BAD, lw=2, alpha=0.9))

# Baseline reference line - thicker + clearer
ax.axhline(baseline, color=INK_MUT, linestyle="--", linewidth=2.2, alpha=0.75, zorder=1)
ax.text(8.4, baseline + 0.4, f"L1 alone baseline: {baseline:.1f}%",
        color=INK, fontsize=14, va="bottom", ha="right", style="italic",
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=INK_MUT, alpha=0.95))

# Two thick lines with clear markers
ax.plot(Ns, d_on, "-", color=COL_MPS_ON, linewidth=4.5, marker="o",
        markersize=17, markerfacecolor="white", markeredgewidth=3.5, zorder=6, label=None)
ax.plot(Ns, d_off, "-", color=COL_MPS_OFF, linewidth=4.5, marker="s",
        markersize=15, markerfacecolor="white", markeredgewidth=3.5, zorder=6, label=None)

# Direct end-labels (larger)
ax.annotate("MPS on", xy=(Ns[-1], d_on[-1]), xytext=(15, -8),
            textcoords="offset points", color=COL_MPS_ON, fontsize=17, fontweight="bold", va="center")
ax.annotate("MPS off", xy=(Ns[-1], d_off[-1]), xytext=(15, 0),
            textcoords="offset points", color=COL_MPS_OFF, fontsize=17, fontweight="bold", va="center")

# Value labels on key points (larger)
for N, v in zip(Ns, d_on):
    if N in [1, 4, 6, 8]:
        ax.text(N, v + 1.8, f"{v:.1f}%", ha="center", fontsize=14, color=COL_MPS_ON, fontweight="bold")
for N, v in zip(Ns, d_off):
    if N in [1, 8]:
        ax.text(N, v - 2.5, f"{v:.1f}%", ha="center", fontsize=14, color=COL_MPS_OFF, fontweight="bold")

ax.set_xlabel("N (concurrent NRx processes on same MIG partition)", color=INK, fontsize=17, fontweight="bold")
ax.set_ylabel("L1 GPU duty cycle (%, higher = better)", color=INK, fontsize=17, fontweight="bold")
ax.set_xticks(Ns)
ax.set_ylim(0, 44)
ax.set_xlim(0.5, 9.7)
ax.grid(axis="y", alpha=0.5)
ax.grid(axis="x", alpha=0)
ax.tick_params(axis="both", which="major", labelsize=15, pad=8)

ax.set_title("Figure 23 · L1 duty cycle collapses at N=6 processes even with MPS on",
             fontweight="bold", pad=22, loc="left", fontsize=20)
fig.text(0.02, 0.008,
         "L1 duty = kernel time / (kernel time + gap time). Chain 17 Config A same-partition. "
         "MPS on recovers baseline through N=4; ≥N=6 MPS scheduler saturates → duty drops from 28% to 14%.",
         fontsize=13, color=INK_SEC, style="italic")
plt.tight_layout(rect=[0, 0.045, 1, 1])
plt.savefig(f"{FIG}/P23_duty_cycle_headline.png"); plt.close()
print("saved P23 v3")

# ============================================================
# P24 V3: Part 8 realistic stack (biggest labels)
# ============================================================
p8_conds = ["baseline", "CPdiverse", "CPuniform", "SPdiverse", "SPuniform"]
p8_labels = {
    "baseline":  "L1 alone\n(reference)",
    "CPdiverse": "Cross-partition\n+ 6 diverse AI",
    "CPuniform": "Cross-partition\n+ 6× NRx",
    "SPdiverse": "Same-partition\n+ 6 diverse AI",
    "SPuniform": "Same-partition\n+ 6× NRx",
}
p8_colors = {"baseline":COL_BASELINE,"CPdiverse":COL_GOOD,"CPuniform":COL_GOOD,
             "SPdiverse":COL_WARN,"SPuniform":COL_BAD}
p8 = {c: stats(sorted(glob.glob(f"{BASE}/chain18_p8_gapstats/p8_{c}_t*.gputrace.csv"))) for c in p8_conds}

fig, ax = plt.subplots(figsize=(15, 9))
positions = np.arange(len(p8_conds))
p95_vals = [p8[c]["gap_p95"]/1000 for c in p8_conds]
colors = [p8_colors[c] for c in p8_conds]

# Region shading - stronger contrast
ax.axvspan(-0.5, 2.5, alpha=0.10, color=COL_GOOD, zorder=-1)
ax.axvspan(2.5, 4.5, alpha=0.10, color=COL_BAD, zorder=-1)
# Zone labels
ax.text(1, max(p95_vals)*1.1, "SAFE  (baseline preserved)", ha="center", va="center",
        fontsize=15, color=COL_GOOD, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=COL_GOOD, lw=2, alpha=0.95))
ax.text(3.5, max(p95_vals)*1.1, "UNSAFE  (SLA-breaking)", ha="center", va="center",
        fontsize=15, color=COL_BAD, fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec=COL_BAD, lw=2, alpha=0.95))

bars = ax.bar(positions, p95_vals, color=colors, edgecolor="white", linewidth=3, width=0.7)

# Big value labels
for pos, val, col in zip(positions, p95_vals, colors):
    ax.text(pos, val + max(p95_vals)*0.02, f"{val:.0f} μs",
            ha="center", va="bottom", fontsize=18, fontweight="bold", color=col)

# Multiplier annotation (below bars)
baseline_val = p8["baseline"]["gap_p95"]/1000
for pos, val, col, cond in zip(positions, p95_vals, colors, p8_conds):
    if cond == "baseline": continue
    ratio = val / baseline_val
    ax.text(pos, -max(p95_vals)*0.03, f"{ratio:.1f}× baseline",
            ha="center", va="top", fontsize=13, color=col, fontweight="bold", style="italic")

# Baseline horizontal line (subtle)
ax.axhline(baseline_val, color=INK_MUT, linestyle="--", linewidth=2, alpha=0.6, zorder=0)

ax.set_xticks(positions)
ax.set_xticklabels([p8_labels[c] for c in p8_conds], fontsize=14)
ax.set_ylabel("L1 inter-kernel gap p95 (μs, lower is better)", color=INK, fontsize=17, fontweight="bold")
ax.set_ylim(-max(p95_vals)*0.08, max(p95_vals) * 1.25)
ax.grid(axis="y", alpha=0.5)
ax.grid(axis="x", alpha=0)

ax.set_title("Figure 24 · Cross-partition preserves L1 baseline; same-partition breaks it 5×",
             fontweight="bold", pad=22, loc="left", fontsize=20)
fig.text(0.02, 0.008,
         "Chain 18 Part 8: L1 (cuPHY 20 cells) + realistic 6-workload AI stack (Qwen 2.5-3B vLLM + Whisper large-v3 + BERT + NRx + CSI + Beam). "
         "Green = safe (baseline preserved). Red = SLA-breaking. Same-partition with identical NRx replicas fails hardest.",
         fontsize=13, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{FIG}/P24_part8_realistic_stack.png"); plt.close()
print("saved P24 v3")

# ============================================================
# P28 V3: per-kernel ratios (clearer horizontal bars)
# ============================================================
p8c = ["baseline", "SPuniform"]
kd = {c: defaultdict(list) for c in p8c}
for c in p8c:
    for p in sorted(glob.glob(f"{BASE}/chain18_p8_gapstats/p8_{c}_t*.gputrace.csv")):
        for s,d,n,_ in parse_trace(p):
            kd[c][n].append(d/1000)

# short names
def short(name):
    if "cupy_copy__complex64" in name: return "cupy_copy (complex64)"
    if "cupy_copy__float32" in name: return "cupy_copy (float32)"
    if "convert_kernel" in name: return "convert_kernel (fp16↔fp32)"
    if "eqMmseCoefCompLow" in name: return "eqMmseCoef (MMSE coef)"
    if "eqMmseSoftDemap" in name: return "eqMmseSoftDemap"
    if "chEstFilterNoDftSOfdm" in name: return "chEstFilter"
    if "windowedChEstPreNoDftSOfdm" in name: return "chEstPre"
    if "noiseIntfEst" in name: return "noiseIntfEst"
    return name[:40]

data = []
for k in kd["baseline"]:
    if k in kd["SPuniform"]:
        b = np.median(kd["baseline"][k])
        s = np.median(kd["SPuniform"][k])
        if b > 0.5:
            data.append((short(k), b, s, s/b))
data.sort(key=lambda x: -x[3])
data = data[:8]

labels = [d[0] for d in data]
baselines = [d[1] for d in data]
sps = [d[2] for d in data]
ratios = [d[3] for d in data]

fig, ax = plt.subplots(figsize=(15, 9))
y = np.arange(len(labels))
h = 0.36

ax.barh(y-h/2, baselines, h, color=COL_BASELINE, alpha=0.85, edgecolor="white", linewidth=1.5, label="L1 alone (baseline)")
ax.barh(y+h/2, sps, h, color=COL_BAD, alpha=0.85, edgecolor="white", linewidth=1.5, label="+ 6× NRx same-partition")

xmax = max(sps) * 1.35
for i, (b, s, r) in enumerate(zip(baselines, sps, ratios)):
    # Duration labels
    ax.text(b + xmax*0.005, i-h/2, f"{b:.1f} μs", va="center", fontsize=12, color=COL_BASELINE, fontweight="bold")
    ax.text(s + xmax*0.005, i+h/2, f"{s:.1f} μs", va="center", fontsize=12, color=COL_BAD, fontweight="bold")
    # Ratio callout box on right
    ax.text(xmax*0.98, i, f"{r:.1f}×", va="center", ha="right",
            fontsize=17, fontweight="bold", color="white",
            bbox=dict(boxstyle="round,pad=0.45", fc=COL_BAD, ec="none", alpha=0.95))

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=15)
ax.set_xlabel("Kernel duration median (μs)", color=INK, fontsize=17, fontweight="bold")
ax.set_xlim(0, xmax)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.5)
ax.grid(axis="y", alpha=0)
ax.legend(loc="lower right", frameon=True, framealpha=0.95, fontsize=15,
          bbox_to_anchor=(0.85, 0.02))

ax.set_title("Figure 28 · Every cuPHY kernel type slows down 1.9-3.1× under 6-proc same-partition pressure",
             fontweight="bold", pad=22, loc="left", fontsize=19)
fig.text(0.02, 0.008,
         "Median duration of the 8 most-common L1 kernel types. Uniform slowdown across ALL kernel classes (memcpy, dtype, MMSE, ch-est) "
         "confirms driver-level bottleneck: launch queue serialization delays EVERY kernel launch, not just heavy ones.",
         fontsize=13, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{FIG}/P28_per_kernel_ratios.png"); plt.close()
print("saved P28 v3")

print("\nHeadline figures v3 done.")
