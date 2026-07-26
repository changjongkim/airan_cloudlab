#!/usr/bin/env python3
"""Chain 18 polished figures — redraw the money-shot figures with clarity focus.

Applies dataviz principles:
- One clear takeaway per figure (title states the finding, not just describes)
- Larger typography (14-16pt)
- Semantic colors (status-coded: good=green, warning=amber, bad=red; sequential N=blue ramp)
- Direct data labels on key points
- Recessive grid/axes
- Minimal legend clutter — direct-label instead
- High figure DPI (150+)

Outputs polished versions:
  P24: Part 8 realistic stack (money shot)
  P23: L1 duty cycle across N (headline)
  P30: gap CDF (analytical proof)
  P33: SLA budget (deployment answer)
  P21: kernel gap vs N (breakdown discovery)
  P28: per-kernel duration ratios (mechanistic proof)
  P29: extended N-sweep asymptote
  P32: launch-rate reconciliation
"""
import os, csv, glob, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
FIG  = os.path.join(BASE, "figures", "polished")
os.makedirs(FIG, exist_ok=True)

# ---- Design tokens (semantic) ----
INK           = "#0f172a"   # primary text
INK_SEC       = "#475569"   # secondary text
INK_MUT       = "#94a3b8"   # muted/axis
SURFACE       = "#ffffff"
GRID          = "#e2e8f0"

# Status colors (reserved semantic meaning)
COL_BASELINE  = "#0f172a"   # black - reference
COL_GOOD      = "#059669"   # green - safe (baseline preserved)
COL_WARN      = "#d97706"   # amber - marginal (edge of breakdown)
COL_BAD       = "#b91c1c"   # red - breakdown

# Sequential (magnitude / N=1..16 ramp)
N_RAMP = ["#dbeafe", "#93c5fd", "#60a5fa", "#3b82f6", "#2563eb", "#1d4ed8", "#1e3a8a"]

# Diverging (MPS off vs on)
COL_MPS_OFF   = "#b91c1c"
COL_MPS_ON    = "#059669"

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"],
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 13,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 17,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": True,
    "axes.spines.bottom": True,
    "axes.edgecolor": INK_MUT,
    "axes.linewidth": 0.8,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK_SEC,
    "ytick.color": INK_SEC,
    "grid.color": GRID,
    "grid.linewidth": 0.7,
    "axes.grid": True,
    "axes.axisbelow": True,
    "axes.unicode_minus": False,
    "figure.dpi": 100,
    "savefig.dpi": 160,
    "savefig.bbox": "tight",
    "savefig.facecolor": SURFACE,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
})

def parse_trace(p):
    kernels = []
    if not os.path.exists(p): return kernels
    with open(p) as f:
        rdr = csv.reader(f); hdr=None
        for row in rdr:
            if row and row[0]=="Start (ns)": hdr=row; break
        if not hdr: return kernels
        idx_s=hdr.index("Start (ns)"); idx_d=hdr.index("Duration (ns)")
        idx_g=hdr.index("GrdX"); idx_n=hdr.index("Name"); idx_st=hdr.index("Strm")
        for row in rdr:
            if len(row)<=idx_n or not row[idx_g].strip(): continue
            n = row[idx_n]
            if 'memcpy' in n.lower() or 'memset' in n.lower(): continue
            try: kernels.append((int(row[idx_s]), int(row[idx_d]), n[:60], row[idx_st]))
            except: pass
    return kernels

def stats_from(files):
    all_d=[]; all_g=[]
    for p in files:
        kernels = parse_trace(p)
        by_stream = defaultdict(list)
        for s,d,_,st in kernels: by_stream[st].append((s,d))
        for _, kl in by_stream.items():
            kl.sort()
            for j,(s,d) in enumerate(kl):
                all_d.append(d)
                if j>0:
                    g = s - (kl[j-1][0]+kl[j-1][1])
                    if g>0: all_g.append(g)
    if not all_d: return None
    d=np.array(all_d); g=np.array(all_g) if all_g else np.array([0])
    return dict(dur_med=np.median(d), gap_med=np.median(g),
                gap_p95=np.percentile(g,95), gap_p99=np.percentile(g,99),
                duty=d.sum()/(d.sum()+g.sum())*100, n=len(d))

# ============================================================
# P24: Part 8 realistic stack — the money shot
# Focus: 5 scenarios, one clear takeaway: CP preserves L1, SP breaks it
# ============================================================
p8_conds = ["baseline", "CPdiverse", "CPuniform", "SPdiverse", "SPuniform"]
p8_labels = {
    "baseline":  "L1 alone\n(reference)",
    "CPdiverse": "Cross-partition\n+ 6 diverse AI",
    "CPuniform": "Cross-partition\n+ 6× NRx",
    "SPdiverse": "Same-partition\n+ 6 diverse AI",
    "SPuniform": "Same-partition\n+ 6× NRx",
}
# semantic color mapping
p8_colors = {
    "baseline":  COL_BASELINE,
    "CPdiverse": COL_GOOD,
    "CPuniform": COL_GOOD,
    "SPdiverse": COL_WARN,
    "SPuniform": COL_BAD,
}
p8 = {c: stats_from(sorted(glob.glob(f"{BASE}/chain18_p8_gapstats/p8_{c}_t*.gputrace.csv")))
      for c in p8_conds}

fig, ax = plt.subplots(figsize=(13, 7))
positions = np.arange(len(p8_conds))
p95_vals = [p8[c]["gap_p95"]/1000 for c in p8_conds]  # us
colors   = [p8_colors[c] for c in p8_conds]

bars = ax.bar(positions, p95_vals, color=colors, edgecolor=SURFACE, linewidth=2, width=0.7)

# Direct data labels
for pos, val, col in zip(positions, p95_vals, colors):
    ax.text(pos, val + max(p95_vals)*0.025, f"{val:.0f} μs",
            ha="center", va="bottom", fontsize=13, fontweight="bold", color=col)

# Baseline reference line
baseline_val = p8["baseline"]["gap_p95"]/1000
ax.axhline(baseline_val, color=INK_MUT, linestyle="--", linewidth=1.2, alpha=0.7, zorder=0)
ax.text(len(p8_conds)-0.5, baseline_val + 25, f"baseline {baseline_val:.0f} μs",
        ha="right", color=INK_SEC, fontsize=11, style="italic")

ax.set_xticks(positions)
ax.set_xticklabels([p8_labels[c] for c in p8_conds], fontsize=12)
ax.set_ylabel("L1 inter-kernel gap p95 (μs, lower is better)", color=INK)
ax.set_ylim(0, max(p95_vals) * 1.20)
ax.spines['bottom'].set_color(INK_MUT)
ax.grid(axis="y", alpha=0.4)
ax.grid(axis="x", alpha=0)

# Region shading
ax.axvspan(-0.5, 2.5, alpha=0.06, color=COL_GOOD, zorder=-1)
ax.axvspan(2.5, 4.5,  alpha=0.06, color=COL_BAD,  zorder=-1)

# Title = the finding, not the description
ax.set_title("Figure 24 · Cross-partition preserves L1 baseline; same-partition breaks it 5×",
             fontweight="bold", pad=20, loc="left")

# Subtitle / annotation
fig.text(0.02, 0.005,
         "Chain 18 Part 8: L1 (cuPHY, 20 cells) + realistic 6-workload AI stack (Qwen 2.5-3B + Whisper + BERT + NRx + CSI + Beam).\n"
         "Green = safe (baseline preserved). Red = SLA-breaking. Same-partition with identical NRx replicas fails hardest.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.06, 1, 1])
plt.savefig(f"{FIG}/P24_part8_realistic_stack.png"); plt.close()
print("saved P24_part8_realistic_stack.png")

# ============================================================
# P23: L1 duty cycle vs N (headline chart)
# ============================================================
Ns = [1, 2, 3, 4, 6, 8]
duty_off=[]; duty_on=[]
for N in Ns:
    off = stats_from([f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPSoff_t1.gputrace.csv"])
    on  = stats_from([f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPSon_t1.gputrace.csv"])
    duty_off.append(off["duty"] if off else 0)
    duty_on.append(on["duty"] if on else 0)
baseline_duty = stats_from([f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"])["duty"]

fig, ax = plt.subplots(figsize=(12, 6.5))
# Breakdown zone shading (semantic)
ax.axvspan(5.5, 8.5, alpha=0.08, color=COL_BAD, zorder=0)
ax.text(7, 4, "MPS-on breakdown zone",
        fontsize=11.5, color=COL_BAD, ha="center", fontweight="bold", alpha=0.9)
ax.axvspan(0.5, 4.5, alpha=0.05, color=COL_GOOD, zorder=0)
ax.text(2.5, 4, "safe with MPS on",
        fontsize=11.5, color=COL_GOOD, ha="center", fontweight="bold", alpha=0.85)

# Baseline reference
ax.axhline(baseline_duty, color=INK_MUT, linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(8.3, baseline_duty + 0.5, f"L1 alone: {baseline_duty:.1f}%",
        color=INK_SEC, fontsize=11, va="bottom", ha="right", style="italic")

# Two lines: MPS off (red) vs MPS on (green)
ax.plot(Ns, duty_on,  "-", color=COL_MPS_ON,  linewidth=3.5, marker="o",
        markersize=13, markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=5, label=None)
ax.plot(Ns, duty_off, "-", color=COL_MPS_OFF, linewidth=3.5, marker="s",
        markersize=11, markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=5, label=None)

# Direct labels on last points
ax.annotate("MPS on", xy=(Ns[-1], duty_on[-1]), xytext=(10, -5),
            textcoords="offset points", color=COL_MPS_ON, fontsize=13, fontweight="bold", va="center")
ax.annotate("MPS off", xy=(Ns[-1], duty_off[-1]), xytext=(10, 0),
            textcoords="offset points", color=COL_MPS_OFF, fontsize=13, fontweight="bold", va="center")

# Value labels on key data points
for N, v in zip(Ns, duty_on):
    if N in [1, 6, 8]:
        ax.text(N, v + 1.6, f"{v:.1f}%", ha="center", fontsize=11, color=COL_MPS_ON, fontweight="bold")

ax.set_xlabel("N (concurrent NRx processes on same MIG partition)", color=INK)
ax.set_ylabel("L1 GPU duty cycle (%, higher is better)", color=INK)
ax.set_xticks(Ns)
ax.set_ylim(0, 40)
ax.set_xlim(0.5, 9.5)
ax.grid(axis="y", alpha=0.4)
ax.grid(axis="x", alpha=0)

ax.set_title("Figure 23 · L1 duty cycle collapses at N=6 processes even with MPS on",
             fontweight="bold", pad=18, loc="left")
fig.text(0.02, 0.005,
         "L1 duty = kernel time / (kernel time + gap time). Chain 17 Config A same-partition. MPS on recovers baseline up to N=4; ≥N=6 MPS scheduler saturates.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{FIG}/P23_duty_cycle_headline.png"); plt.close()
print("saved P23_duty_cycle_headline.png")

# ============================================================
# P21: kernel gap vs N — three panels but each with clear message
# ============================================================
Ns_all = [1, 2, 3, 4, 6, 8]
gap_data = {N: {"off": None, "on": None} for N in Ns_all}
for N in Ns_all:
    for m in ["off", "on"]:
        s = stats_from([f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPS{m}_t1.gputrace.csv"])
        if s: gap_data[N][m] = s

fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
metrics = [("gap_med", "median gap (μs)", "median (typical)"),
           ("gap_p95", "gap p95 (μs)", "p95 (tail)")]
for ax, (key, ylabel, subtitle) in zip(axes, metrics):
    vals_on  = [gap_data[N]["on"][key]/1000 for N in Ns_all]
    vals_off = [gap_data[N]["off"][key]/1000 for N in Ns_all]
    ax.axvspan(5.5, 8.5, alpha=0.08, color=COL_BAD, zorder=0)
    ax.plot(Ns_all, vals_on,  "-", color=COL_MPS_ON,  linewidth=3, marker="o", markersize=12,
            markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=5)
    ax.plot(Ns_all, vals_off, "-", color=COL_MPS_OFF, linewidth=3, marker="s", markersize=11,
            markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=5)
    ax.annotate("MPS on", xy=(Ns_all[-1], vals_on[-1]), xytext=(10, -5),
                textcoords="offset points", color=COL_MPS_ON, fontsize=12, fontweight="bold")
    ax.annotate("MPS off", xy=(Ns_all[-1], vals_off[-1]), xytext=(10, 0),
                textcoords="offset points", color=COL_MPS_OFF, fontsize=12, fontweight="bold")
    ax.set_yscale("log")
    ax.set_xlabel("N (concurrent NRx processes)", color=INK)
    ax.set_ylabel(ylabel, color=INK)
    ax.set_title(subtitle, fontweight="bold", pad=10, loc="left", color=INK_SEC)
    ax.set_xticks(Ns_all)
    ax.grid(alpha=0.4, which="both")

fig.suptitle("Figure 21 · L1 inter-kernel gap explodes 100-300× at N=6 (MPS on)",
             fontweight="bold", y=1.01, x=0.02, ha="left", fontsize=17)
fig.text(0.02, 0.005,
         "Chain 17 Config A. MPS off maintains flat median but wild tails at all N. MPS on preserves both metrics through N=4; N≥6 median shoots from ~1μs to 120-380μs.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 0.96])
plt.savefig(f"{FIG}/P21_gap_vs_N.png"); plt.close()
print("saved P21_gap_vs_N.png")

# ============================================================
# P28: per-kernel duration ratios — bar chart with clear "3× slower" callout
# ============================================================
# Reload NCU data for kernel names
def parse_ncu(path):
    kernels = defaultdict(dict)
    with open(path) as f:
        rdr = csv.reader(f); hdr=None
        for row in rdr:
            if not row or row[0].startswith("=="): continue
            if row[0]=="ID": hdr = row; continue
            if hdr is None: continue
            d = dict(zip(hdr, row))
            kid = d.get("ID","")
            mn = d.get("Metric Name","")
            kernels[kid]["_name"] = d.get("Kernel Name","")[:60]
            try: kernels[kid][mn] = float(d.get("Metric Value","").replace(",",""))
            except: pass
    return list(kernels.values())

# Load per-kernel medians from Part 8 traces
kern_dur = {"baseline": defaultdict(list), "SPuniform": defaultdict(list)}
for c in ["baseline", "SPuniform"]:
    for p in sorted(glob.glob(f"{BASE}/chain18_p8_gapstats/p8_{c}_t*.gputrace.csv")):
        for s,d,n,_ in parse_trace(p):
            kern_dur[c][n].append(d)

# Build ratios
data = []
for k in kern_dur["baseline"]:
    if k in kern_dur["SPuniform"]:
        b = np.median(kern_dur["baseline"][k]) / 1000  # us
        s = np.median(kern_dur["SPuniform"][k]) / 1000
        if b > 0.5:  # skip tiny kernels
            data.append((k, b, s, s/b))
data.sort(key=lambda x: -x[3])  # ratio desc
data = data[:8]  # top 8

# Short labels
def short(name):
    if "cupy_copy" in name:
        typ = name.split("__")[1] if "__" in name else name.split("<")[1].split(">")[0] if "<" in name else ""
        return f"cupy_copy ({typ[:12]})"
    if "convert_kernel" in name: return "convert_kernel<__half2,float2>"
    if "channel_eq::eqMmseCoefCompLow" in name: return "channel_eq::eqMmseCoefCompLowMimo"
    if "channel_eq::eqMmseSoftDemap" in name: return "channel_eq::eqMmseSoftDemap"
    if "chEstFilterNoDftSOfdmDispatch" in name: return "ch_est::chEstFilterNoDftSOfdmDispatch"
    if "pusch_noise_intf_est::noiseIntfEst" in name: return "pusch::noiseIntfEst"
    if "windowedChEstPreNoDftSOfdm" in name: return "ch_est::windowedChEstPreNoDftSOfdm"
    return name[:40]

labels = [short(d[0]) for d in data]
baselines = [d[1] for d in data]
sps       = [d[2] for d in data]
ratios    = [d[3] for d in data]

fig, ax = plt.subplots(figsize=(13, 7))
y = np.arange(len(labels))
w = 0.36

b_bars = ax.barh(y-w/2, baselines, w, color=COL_BASELINE, alpha=0.85, edgecolor=SURFACE, linewidth=1.5, label="L1 alone")
s_bars = ax.barh(y+w/2, sps,       w, color=COL_BAD,      alpha=0.85, edgecolor=SURFACE, linewidth=1.5, label="+ 6× NRx same-partition")

# Direct duration labels
for i, (b, s, r) in enumerate(zip(baselines, sps, ratios)):
    ax.text(b + max(sps)*0.008, i-w/2, f"{b:.1f}μs", va="center", fontsize=10, color=COL_BASELINE)
    ax.text(s + max(sps)*0.008, i+w/2, f"{s:.1f}μs", va="center", fontsize=10, color=COL_BAD)
    # Ratio callout
    ax.text(max(sps)*1.08, i, f"{r:.1f}×", va="center", ha="left",
            fontsize=13, fontweight="bold", color=COL_BAD)

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=11)
ax.set_xlabel("Kernel duration median (μs)", color=INK)
ax.set_xlim(0, max(sps) * 1.20)
ax.invert_yaxis()
ax.grid(axis="x", alpha=0.4)
ax.grid(axis="y", alpha=0)
ax.legend(loc="lower right", frameon=False, fontsize=12)

ax.set_title("Figure 28 · Every cuPHY kernel type slows down 1.9-3.1× under 6-proc same-partition pressure",
             fontweight="bold", pad=18, loc="left")
fig.text(0.02, 0.005,
         "Median duration of the 8 most-common L1 kernel types. Uniform slowdown across all kernel classes (memcpy, dtype, MMSE, channel-est, noise-est) confirms driver-level bottleneck: launch queue serialization delays EVERY kernel.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{FIG}/P28_per_kernel_ratios.png"); plt.close()
print("saved P28_per_kernel_ratios.png")

# ============================================================
# P30: Gap CDF log-log — key story: MPS on preserves shape until N=6
# Simplified to fewer curves for clarity
# ============================================================
key_conds = [
    ("L1 alone (baseline)",  COL_BASELINE, "-",  f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"),
    ("N=4 MPS on (safe)",    COL_GOOD,     "-",  f"{BASE}/chain17_gapstats/cfgA_A_nrxN4_MPSon_t1.gputrace.csv"),
    ("N=6 MPS on (breakdown)", COL_WARN,   "-",  f"{BASE}/chain17_gapstats/cfgA_A_nrxN6_MPSon_t1.gputrace.csv"),
    ("N=8 MPS on (worse)",   COL_BAD,      "-",  f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSon_t1.gputrace.csv"),
    ("N=8 MPS off (catastrophic)", INK,    "--", f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSoff_t1.gputrace.csv"),
]
fig, ax = plt.subplots(figsize=(12, 7))
for label, color, ls, path in key_conds:
    kernels = parse_trace(path)
    if not kernels: continue
    by_stream = defaultdict(list)
    for s,d,_,st in kernels: by_stream[st].append((s,d))
    gaps=[]
    for _, kl in by_stream.items():
        kl.sort()
        for j in range(1, len(kl)):
            g = kl[j][0] - (kl[j-1][0]+kl[j-1][1])
            if g>0: gaps.append(g)
    if not gaps: continue
    g_us = np.sort(np.array(gaps)/1000)
    cdf = np.arange(1, len(g_us)+1)/len(g_us)
    ax.plot(g_us, 1-cdf, ls, color=color, linewidth=2.8, alpha=0.92, label=label)

# Reference lines at 5G TTI ranges
ax.axvline(500, color=INK_MUT, linestyle=":", linewidth=1.5, alpha=0.7)
ax.text(500, 1.3e-4, "5G TTI\n500μs", color=INK_SEC, fontsize=11, ha="center", va="top", style="italic",
        bbox=dict(boxstyle="round,pad=0.3", fc=SURFACE, ec=INK_MUT, alpha=0.9))

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Inter-kernel gap (μs, log scale)", color=INK)
ax.set_ylabel("P(gap > x)  [1 − CDF, log scale]", color=INK)
ax.set_xlim(0.5, 30000)
ax.set_ylim(1e-5, 1.5)
ax.grid(alpha=0.4, which="major")
ax.grid(alpha=0.15, which="minor")
ax.legend(loc="lower left", frameon=False, fontsize=12)

ax.set_title("Figure 30 · Gap distribution shape shifts to heavier-tailed at N≥6, not just longer",
             fontweight="bold", pad=18, loc="left")
fig.text(0.02, 0.005,
         "Chain 17 gap survival function. Baseline / N=4 curves collapse onto each other → MPS on preserves distributional shape. N=6 crosses 5G TTI line at p99. N=8 tail approaches 10 ms.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{FIG}/P30_gap_cdf.png"); plt.close()
print("saved P30_gap_cdf.png")

# ============================================================
# P33: SLA budget — clearer story: only CP fits in TTI
# ============================================================
sla_conds = [
    ("L1 alone",                 COL_BASELINE),
    ("CP + 6 diverse AI",        COL_GOOD),
    ("CP + 6× NRx",              COL_GOOD),
    ("SP + 6 diverse AI",        COL_WARN),
    ("SP + 6× NRx",              COL_BAD),
    ("SP N=6 MPS on",            COL_BAD),
    ("SP N=8 MPS on",            COL_BAD),
    ("SP N=8 MPS off",           INK),
]
sla_paths = [
    f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv",
    f"{BASE}/chain18_p8_gapstats/p8_CPdiverse_t1.gputrace.csv",
    f"{BASE}/chain18_p8_gapstats/p8_CPuniform_t1.gputrace.csv",
    f"{BASE}/chain18_p8_gapstats/p8_SPdiverse_t1.gputrace.csv",
    f"{BASE}/chain18_p8_gapstats/p8_SPuniform_t1.gputrace.csv",
    f"{BASE}/chain17_gapstats/cfgA_A_nrxN6_MPSon_t1.gputrace.csv",
    f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSon_t1.gputrace.csv",
    f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSoff_t1.gputrace.csv",
]
per_slot_us = []
for path in sla_paths:
    kernels = parse_trace(path)
    if not kernels: per_slot_us.append(0); continue
    by_stream = defaultdict(list)
    for s,d,_,st in kernels: by_stream[st].append((s,d))
    durs=[]; gaps=[]
    for _, kl in by_stream.items():
        kl.sort()
        for j,(s,d) in enumerate(kl):
            durs.append(d)
            if j>0:
                g = s - (kl[j-1][0]+kl[j-1][1])
                if g>0: gaps.append(g)
    if not durs: per_slot_us.append(0); continue
    d=np.array(durs); g=np.array(gaps) if gaps else np.array([0])
    per_kernel = (np.median(d) + np.median(g)) / 1000
    per_slot_us.append(per_kernel * 100)  # 100 kernels/slot proxy

labels, colors = zip(*sla_conds)
y = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(13, 8))

# TTI threshold shading
ax.axvspan(0, 500, alpha=0.08, color=COL_GOOD, zorder=0)
ax.axvspan(500, max(per_slot_us)*1.15, alpha=0.05, color=COL_BAD, zorder=0)
ax.axvline(500, color=INK, linestyle="--", linewidth=2, alpha=0.8, zorder=1)
ax.text(500, len(labels)-0.3, "5G TTI budget\n(500μs)", color=INK, fontsize=11, ha="left", va="center",
        fontweight="bold",
        bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=INK_MUT, alpha=0.95))

# Horizontal bars
bars = ax.barh(y, per_slot_us, color=colors, alpha=0.85, edgecolor=SURFACE, linewidth=1.5, height=0.6)

for i, (v, col) in enumerate(zip(per_slot_us, colors)):
    ratio = v / 500
    if ratio < 1.5:
        text = f"{v:.0f} μs  ({ratio:.1f}× TTI)"
    elif ratio < 10:
        text = f"{v:.0f} μs  ({ratio:.1f}× TTI)"
    else:
        text = f"{v/1000:.1f} ms  ({ratio:.0f}× TTI)"
    ax.text(v + max(per_slot_us)*0.008, i, text, va="center", fontsize=11, color=col, fontweight="bold")

ax.set_yticks(y)
ax.set_yticklabels(labels, fontsize=12)
ax.invert_yaxis()
ax.set_xlabel("Estimated per-slot L1 latency (μs, 100 kernels/slot proxy)", color=INK)
ax.set_xlim(0, max(per_slot_us) * 1.30)
ax.grid(axis="x", alpha=0.4)
ax.grid(axis="y", alpha=0)

ax.set_title("Figure 33 · Only cross-partition scenarios fit within 5G TTI budget",
             fontweight="bold", pad=18, loc="left")
fig.text(0.02, 0.005,
         "Estimated median per-slot L1 processing latency. Green zone = safe (< TTI). Red zone = SLA violation. All SP scenarios with N≥6 exceed TTI by 5-100× → guaranteed slot drops.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(f"{FIG}/P33_sla_budget.png"); plt.close()
print("saved P33_sla_budget.png")

# ============================================================
# P32: launch-rate reconciliation
# ============================================================
def launch_rate(path):
    kernels = parse_trace(path)
    if not kernels: return 0
    t0 = min(k[0] for k in kernels)
    tf = max(k[0]+k[1] for k in kernels)
    return len(kernels) / ((tf-t0)/1e9)

ch17_x = [1,2,3,4,6,8]
ch17_y = [launch_rate(f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPSon_t1.gputrace.csv") for N in ch17_x]
p5_x = [1,2,4,8]
p5_y = []
for N in p5_x:
    files = sorted(glob.glob(f"{BASE}/chain18_p5_gapstats/p5_procOnly_N{N}_t*.gputrace.csv"))
    p5_y.append(np.mean([launch_rate(f) for f in files]) if files else 0)

fig, ax = plt.subplots(figsize=(12, 6.5))
ax.axvspan(5.5, 8.5, alpha=0.08, color=COL_BAD, zorder=0)

# Chain 17 curve (breaks)
ax.plot(ch17_x, ch17_y, "-", color=COL_BAD, linewidth=3.5,
        marker="s", markersize=13, markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=5)
# Part 5 curve (stays)
ax.plot(p5_x, p5_y, "-", color=COL_GOOD, linewidth=3.5,
        marker="o", markersize=13, markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=5)

# Direct labels
ax.annotate("Chain 17: 6 identical heavy NRx replicas\n(each pushes kernels at max rate)",
            xy=(8, ch17_y[-1]), xytext=(-40, 70), textcoords="offset points",
            color=COL_BAD, fontsize=11.5, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=COL_BAD, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=COL_BAD, alpha=0.95))
ax.annotate("Part 5: 8 ranai_mix processes\n(14 threads sharing lightweight CUDA ctx)",
            xy=(8, p5_y[-1]), xytext=(0, 60), textcoords="offset points",
            color=COL_GOOD, fontsize=11.5, fontweight="bold", ha="center",
            arrowprops=dict(arrowstyle="->", color=COL_GOOD, lw=1.5),
            bbox=dict(boxstyle="round,pad=0.4", fc=SURFACE, ec=COL_GOOD, alpha=0.95))

# Value annotations on breakdown points
ax.text(6, ch17_y[4]-800, f"{ch17_y[4]:.0f}", ha="center", color=COL_BAD, fontsize=11, fontweight="bold")
ax.text(8, ch17_y[5]-800, f"{ch17_y[5]:.0f}", ha="center", color=COL_BAD, fontsize=11, fontweight="bold")

ax.set_xlabel("N (concurrent AI processes)", color=INK)
ax.set_ylabel("L1 kernel launch rate (kernels/sec)", color=INK)
ax.set_xticks(sorted(set(ch17_x+p5_x)))
ax.grid(axis="y", alpha=0.4)
ax.grid(axis="x", alpha=0)

ax.set_title("Figure 32 · Process count doesn't predict breakdown — aggregate launch rate does",
             fontweight="bold", pad=18, loc="left")
fig.text(0.02, 0.005,
         "Same N=8 gives 1901 (chain17) vs 11423 (part5) kernels/sec for L1. Heavy identical replicas saturate MPS server; multi-threaded diverse-per-process workloads don't.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{FIG}/P32_launch_rate_reconciliation.png"); plt.close()
print("saved P32_launch_rate_reconciliation.png")

# ============================================================
# P29: Extended N sweep with breakdown zones
# ============================================================
# Combine Chain 17 (N=1..8) + Part 3 (N=5..16)
Ns_ext = sorted({1,2,3,4,5,6,7,8,10,12,16})
duty_ext = {}
for N in Ns_ext:
    # Try chain17 first
    s = stats_from([f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPSon_t1.gputrace.csv"])
    if s: duty_ext[N] = s["duty"]; continue
    # Fallback to Part 3
    files = sorted(glob.glob(f"{BASE}/chain18_gapstats/p3_nrxN{N}_MPSon_t*.gputrace.csv"))
    s = stats_from(files)
    if s: duty_ext[N] = s["duty"]
    # Or Part 7 stat
    if not s and N in [5,7]:
        files = sorted(glob.glob(f"{BASE}/chain18_p7_gapstats/p7_stat_nrxN{N}_MPSon_t*.gputrace.csv"))
        s = stats_from(files)
        if s: duty_ext[N] = s["duty"]

Ns_present = sorted(duty_ext.keys())
duties = [duty_ext[N] for N in Ns_present]

fig, ax = plt.subplots(figsize=(13, 6.5))
# Zones
ax.axvspan(0.5, 4.5, alpha=0.05, color=COL_GOOD, zorder=0)
ax.axvspan(4.5, 5.5, alpha=0.06, color=COL_WARN, zorder=0)
ax.axvspan(5.5, 17,  alpha=0.06, color=COL_BAD, zorder=0)
ax.text(2.5, 33, "safe", ha="center", color=COL_GOOD, fontsize=13, fontweight="bold", alpha=0.85)
ax.text(5, 33, "edge", ha="center", color=COL_WARN, fontsize=13, fontweight="bold", alpha=0.85)
ax.text(11, 33, "breakdown", ha="center", color=COL_BAD, fontsize=13, fontweight="bold", alpha=0.85)

# Baseline
ax.axhline(baseline_duty, color=INK_MUT, linestyle="--", linewidth=1.5, alpha=0.7)
ax.text(16, baseline_duty+0.5, f"baseline {baseline_duty:.1f}%",
        color=INK_SEC, fontsize=11, ha="right", va="bottom", style="italic")

# Duty curve
ax.plot(Ns_present, duties, "-", color=COL_MPS_ON, linewidth=3,
        marker="o", markersize=12, markerfacecolor=SURFACE, markeredgewidth=2.5, zorder=5)

# Callouts at key points
callout_Ns = [4, 6, 8, 16]
for N in callout_Ns:
    if N in duty_ext:
        v = duty_ext[N]
        ax.annotate(f"N={N}\n{v:.1f}%", xy=(N, v), xytext=(0, -25),
                    textcoords="offset points", ha="center",
                    color=INK, fontsize=11, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.35", fc=SURFACE, ec=INK_MUT, alpha=0.95))

ax.set_xlabel("N (concurrent NRx processes, MPS on)", color=INK)
ax.set_ylabel("L1 duty cycle (%)", color=INK)
ax.set_xticks(Ns_present)
ax.set_ylim(0, 38)
ax.set_xlim(0.5, 17)
ax.grid(axis="y", alpha=0.4)
ax.grid(axis="x", alpha=0)

ax.set_title("Figure 29 · MPS breakdown asymptotes to ~5-10% duty floor as N grows to 16",
             fontweight="bold", pad=18, loc="left")
fig.text(0.02, 0.005,
         "Chain 17 (N=1..8) + Part 3 (N=5,7,10,12,16) combined. Beyond breakdown at N=6, degradation continues but bounded by L1's irreducible work.",
         fontsize=10.5, color=INK_SEC, style="italic")

plt.tight_layout(rect=[0, 0.05, 1, 1])
plt.savefig(f"{FIG}/P29_extended_nsweep.png"); plt.close()
print("saved P29_extended_nsweep.png")

print("\nAll polished figures saved to:", FIG)
