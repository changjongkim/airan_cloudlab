#!/usr/bin/env python3
"""Chain 18 deeper analysis — mines more from the 2.6GB dataset.

Figures f34-f40:
  f34: Time-series duty cycle over 30s window (breakdown temporal pattern)
  f35: Per-stream kernel activity (does L1 use 1 or many streams?)
  f36: All-trials CDF overlay for Part 7 stat (30 trials)
  f37: Chain 17 workload-signature comparison (nrx/memcpy/embed distribution)
  f38: Per-condition kernel-gap heatmap (all conditions summary)
  f39: NCU-vs-nsys cross reference (per-kernel DRAM vs per-kernel gap after)
  f40: MPS breakdown boundary probability map (N × pct)
"""
import os, csv, glob, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
FIG  = os.path.join(BASE, "figures", "comprehensive")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

def parse_trace(p):
    """Return list of (start_ns, dur_ns, name, stream)."""
    kernels = []
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
            try:
                kernels.append((int(row[idx_s]), int(row[idx_d]), n[:60], row[idx_st]))
            except: pass
    return kernels

# ============================================================
# f34: Time-series duty cycle over 30s window (2s bins)
# For key conditions, plot rolling duty cycle to detect breakdown temporal pattern
# ============================================================
def time_series_duty(kernels, bin_s=2.0):
    """Return (t_center_s, duty_pct) arrays binned every bin_s."""
    if not kernels: return np.array([]), np.array([])
    t0 = min(k[0] for k in kernels)
    tf = max(k[0]+k[1] for k in kernels)
    total_s = (tf - t0)/1e9
    n_bins = max(1, int(total_s/bin_s))
    bins = np.zeros(n_bins)
    for s, d, _, _ in kernels:
        rel_start = (s - t0)/1e9
        rel_end = (s + d - t0)/1e9
        # distribute duration into bins
        b_start = int(rel_start / bin_s)
        b_end = int(rel_end / bin_s)
        if b_start >= n_bins: continue
        b_end = min(b_end, n_bins-1)
        if b_start == b_end:
            bins[b_start] += (rel_end - rel_start)
        else:
            bins[b_start] += (b_start+1)*bin_s - rel_start
            for b in range(b_start+1, b_end):
                bins[b] += bin_s
            bins[b_end] += rel_end - b_end*bin_s
    duty = bins / bin_s * 100  # per-bin fraction * 100
    t = (np.arange(n_bins)+0.5) * bin_s
    return t, duty

fig, axes = plt.subplots(2, 1, figsize=(15, 9), sharex=True)
conds_ts = [
    ("L1 alone", "#3b82f6", f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"),
    ("N=4 MPSon", "#10b981", f"{BASE}/chain17_gapstats/cfgA_A_nrxN4_MPSon_t1.gputrace.csv"),
    ("N=6 MPSon", "#f59e0b", f"{BASE}/chain17_gapstats/cfgA_A_nrxN6_MPSon_t1.gputrace.csv"),
    ("N=8 MPSon", "#f97316", f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSon_t1.gputrace.csv"),
    ("N=8 MPSoff", "#dc2626", f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSoff_t1.gputrace.csv"),
]
for label, color, path in conds_ts:
    if not os.path.exists(path): continue
    kernels = parse_trace(path)
    t, duty = time_series_duty(kernels, bin_s=2.0)
    if len(t)>0:
        axes[0].plot(t, duty, "o-", color=color, label=label, linewidth=1.8, markersize=6, alpha=0.85)

# Second panel: cumulative kernel count
for label, color, path in conds_ts:
    if not os.path.exists(path): continue
    kernels = parse_trace(path)
    if not kernels: continue
    t0 = min(k[0] for k in kernels)
    times = sorted((k[0]-t0)/1e9 for k in kernels)
    cum = np.arange(1, len(times)+1)
    axes[1].plot(times, cum, "-", color=color, label=label, linewidth=1.8, alpha=0.85)

axes[0].set_ylabel("L1 duty cycle per 2s bin (%)")
axes[0].set_title("Temporal duty cycle (rolling 2s bins) — does breakdown ramp or hit immediately?",
                    fontweight="bold")
axes[0].grid(alpha=0.3); axes[0].legend(loc="upper right", fontsize=10)
axes[0].set_ylim(0, 100)
axes[1].set_ylabel("Cumulative L1 kernels launched")
axes[1].set_xlabel("Time within 30s trace (s)")
axes[1].set_title("Cumulative kernel launch progression", fontweight="bold")
axes[1].grid(alpha=0.3); axes[1].legend(loc="upper left", fontsize=10)
plt.suptitle("Figure 34. Temporal analysis — L1 activity within a 30-second trace window\n"
             "Reveals whether MPS breakdown is a startup transient or a steady-state condition",
             fontweight="bold", y=1.005)
plt.tight_layout()
plt.savefig(f"{FIG}/f34_temporal_duty.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f34_temporal_duty.png")

# ============================================================
# f35: Per-stream analysis (how many streams? Do they share load?)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for ax, (label, color, path) in zip(axes, [
    ("L1 alone",  "#3b82f6", f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"),
    ("N=4 MPSon", "#10b981", f"{BASE}/chain17_gapstats/cfgA_A_nrxN4_MPSon_t1.gputrace.csv"),
    ("N=6 MPSon", "#f59e0b", f"{BASE}/chain17_gapstats/cfgA_A_nrxN6_MPSon_t1.gputrace.csv"),
]):
    if not os.path.exists(path): continue
    kernels = parse_trace(path)
    by_stream = defaultdict(list)
    for s, d, n, st in kernels:
        by_stream[st].append((s, d))
    stream_ids = sorted(by_stream.keys(), key=lambda k: -len(by_stream[k]))
    counts = [len(by_stream[s]) for s in stream_ids]
    # bar chart: per-stream kernel count
    ax.bar(range(len(stream_ids)), counts, color=color, alpha=0.85, edgecolor="#111")
    ax.set_xticks(range(len(stream_ids)))
    ax.set_xticklabels([f"str{s}" for s in stream_ids], fontsize=9)
    ax.set_ylabel("Kernel count")
    ax.set_title(f"{label}\n{len(stream_ids)} streams, {sum(counts)} kernels", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)
    for i, c in enumerate(counts):
        ax.text(i, c+50, f"{c}", ha="center", fontsize=9)
plt.suptitle("Figure 35. Per-stream L1 kernel distribution across conditions\n"
             "L1 uses 2 CUDA streams; how balanced are they, does load shift under pressure?",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f35_per_stream.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f35_per_stream.png")

# ============================================================
# f36: All-trials CDF overlay for Part 7 stat (10 trials × 3 Ns = 30 curves)
# ============================================================
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
GAPS7 = f"{BASE}/chain18_p7_gapstats"
for ax, N, color_scheme in zip(axes, [5,6,7], ["Greens","YlOrBr","Reds"]):
    cmap = plt.cm.get_cmap(color_scheme)
    files = sorted(glob.glob(f"{GAPS7}/p7_stat_nrxN{N}_MPSon_t*.gputrace.csv"))
    for i, p in enumerate(files):
        kernels = parse_trace(p)
        by_stream = defaultdict(list)
        for s, d, _, st in kernels:
            by_stream[st].append((s,d))
        gaps = []
        for strm, kl in by_stream.items():
            kl.sort()
            for j in range(1, len(kl)):
                g = kl[j][0] - (kl[j-1][0]+kl[j-1][1])
                if g > 0: gaps.append(g)
        if not gaps: continue
        g_us = np.sort(np.array(gaps)/1000)
        cdf = np.arange(1, len(g_us)+1)/len(g_us)
        col = cmap(0.3 + 0.7*i/len(files))
        ax.plot(g_us, 1-cdf, "-", color=col, linewidth=1.5, alpha=0.75,
                label=f"trial {i+1}" if i<3 or i==len(files)-1 else None)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Gap (μs, log)"); ax.set_ylabel("P(gap > x)")
    ax.set_title(f"N={N} — 10 trials", fontweight="bold")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower left", fontsize=8)
    ax.set_xlim(0.5, 10000)
plt.suptitle("Figure 36. Part 7 statistical robustness — all 10 trials CDF overlay per N\n"
             "How similar are the 10 independent trials? Confirms determinism of breakdown",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f36_all_trials_cdf.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f36_all_trials_cdf.png")

# ============================================================
# f37: Chain 17 workload-signature comparison (Part 3 nrx/memcpy/embed at N=4)
# Show how the L1 gap distribution changes with AI workload TYPE
# ============================================================
fig, ax = plt.subplots(figsize=(13, 6))
p3_types = [("nrx","N=4 NRx","#dc2626"), ("memcpy","N=4 memcpy","#3b82f6"), ("embed","N=4 embed","#10b981")]
for wl, label, color in p3_types:
    files = sorted(glob.glob(f"{BASE}/chain18_gapstats/p3_{wl}N4_MPSon_t*.gputrace.csv"))
    all_gaps = []
    for p in files:
        kernels = parse_trace(p)
        by_stream = defaultdict(list)
        for s, d, _, st in kernels:
            by_stream[st].append((s,d))
        for strm, kl in by_stream.items():
            kl.sort()
            for j in range(1, len(kl)):
                g = kl[j][0] - (kl[j-1][0]+kl[j-1][1])
                if g > 0: all_gaps.append(g)
    if not all_gaps: continue
    g_us = np.sort(np.array(all_gaps)/1000)
    cdf = np.arange(1, len(g_us)+1)/len(g_us)
    ax.plot(g_us, 1-cdf, "-", color=color, linewidth=2.2, label=label, alpha=0.9)
# Add baseline
p = f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"
if os.path.exists(p):
    kernels = parse_trace(p)
    by_stream = defaultdict(list)
    for s, d, _, st in kernels: by_stream[st].append((s,d))
    gaps=[]
    for strm, kl in by_stream.items():
        kl.sort()
        for j in range(1, len(kl)):
            g = kl[j][0] - (kl[j-1][0]+kl[j-1][1])
            if g>0: gaps.append(g)
    if gaps:
        g_us = np.sort(np.array(gaps)/1000)
        cdf = np.arange(1, len(g_us)+1)/len(g_us)
        ax.plot(g_us, 1-cdf, "-", color="#111", linewidth=2, linestyle="--", label="L1 alone (baseline)", alpha=0.7)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("L1 inter-kernel gap (μs, log)")
ax.set_ylabel("P(gap > x)  [1-CDF, log]")
ax.set_title("Figure 37. Same N=4, different AI workload TYPE — do all types stress L1 equally?\n"
             "Reveals which AI kernel signature is most disruptive to L1 sync (all MPS on, Config A)",
             fontweight="bold")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower left", fontsize=11)
ax.set_xlim(0.5, 5000)
plt.tight_layout()
plt.savefig(f"{FIG}/f37_workload_signature_cdf.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f37_workload_signature_cdf.png")

# ============================================================
# f38: Per-condition kernel-gap heatmap (unified summary across ALL analyzed conditions)
# ============================================================
# Collect (condition_label, gap_median, gap_p95, gap_p99, duty)
heat = []

def compute_stats(files):
    all_d=[]; all_g=[]
    for p in files:
        kernels = parse_trace(p)
        by_stream = defaultdict(list)
        for s, d, _, st in kernels: by_stream[st].append((s,d))
        for strm, kl in by_stream.items():
            kl.sort()
            for j, (s,d) in enumerate(kl):
                all_d.append(d)
                if j>0:
                    g = s - (kl[j-1][0]+kl[j-1][1])
                    if g>0: all_g.append(g)
    if not all_d or not all_g: return None
    d=np.array(all_d); g=np.array(all_g)
    return (float(np.median(g)/1000), float(np.percentile(g,95)/1000),
            float(np.percentile(g,99)/1000), float(d.sum()/(d.sum()+g.sum())*100))

# L1 alone
s = compute_stats([f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"])
if s: heat.append(("L1 alone", *s))
# Chain 17 N-sweep
for N in [1,2,3,4,6,8]:
    for m in ["off","on"]:
        f = f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPS{m}_t1.gputrace.csv"
        s = compute_stats([f])
        if s: heat.append((f"ch17 N{N} MPS{m}", *s))
# Part 3 workload types at N=4
for wl in ["nrx","memcpy","embed"]:
    files = sorted(glob.glob(f"{BASE}/chain18_gapstats/p3_{wl}N4_MPSon_t*.gputrace.csv"))
    s = compute_stats(files)
    if s: heat.append((f"p3 {wl} N4 MPSon", *s))
# Part 3 extended N (nrx)
for N in [5,7,10,12,16]:
    files = sorted(glob.glob(f"{BASE}/chain18_gapstats/p3_nrxN{N}_MPSon_t*.gputrace.csv"))
    s = compute_stats(files)
    if s: heat.append((f"p3 nrx N{N} MPSon", *s))
# Part 5
for cfg in [(4,"thr"),(8,"thr"),(16,"thr"),(32,"thr"),(1,"proc"),(2,"proc"),(4,"proc"),(8,"proc")]:
    v, kind = cfg
    files = sorted(glob.glob(f"{BASE}/chain18_p5_gapstats/p5_{kind}Only_{'beam' if kind=='thr' else 'N'}{v}_t*.gputrace.csv"))
    s = compute_stats(files)
    if s: heat.append((f"p5 {kind} {v}", *s))
# Part 7
for N in [5,6,7]:
    files = sorted(glob.glob(f"{BASE}/chain18_p7_gapstats/p7_stat_nrxN{N}_MPSon_t*.gputrace.csv"))
    s = compute_stats(files)
    if s: heat.append((f"p7 stat N{N} MPSon", *s))
# Part 8
for c in ["baseline","CPdiverse","CPuniform","SPdiverse","SPuniform"]:
    files = sorted(glob.glob(f"{BASE}/chain18_p8_gapstats/p8_{c}_t*.gputrace.csv"))
    s = compute_stats(files)
    if s: heat.append((f"p8 {c}", *s))

# Sort by duty cycle
heat.sort(key=lambda x: -x[4])  # duty descending

fig, ax = plt.subplots(figsize=(14, 12))
labels = [h[0] for h in heat]
duty = np.array([h[4] for h in heat])
med  = np.array([h[1] for h in heat])
p95  = np.array([h[2] for h in heat])
p99  = np.array([h[3] for h in heat])

y = np.arange(len(labels))
# Show 3 metrics as different colored bars
w = 0.28
ax.barh(y-w, np.log10(med+1),  w, color="#3b82f6", alpha=0.85, edgecolor="#111", label="log10(gap_med μs)")
ax.barh(y,   np.log10(p95+1),  w, color="#f59e0b", alpha=0.85, edgecolor="#111", label="log10(gap_p95 μs)")
ax.barh(y+w, np.log10(p99+1),  w, color="#dc2626", alpha=0.85, edgecolor="#111", label="log10(gap_p99 μs)")

# Second axis: duty cycle
ax2 = ax.twiny()
ax2.plot(duty, y, "o", color="#059669", markersize=9, zorder=5, label="duty cycle %")
ax2.set_xlim(0, 100)
ax2.set_xlabel("L1 duty cycle (%)", color="#059669", fontweight="bold")
ax2.tick_params(axis="x", colors="#059669")

ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9)
ax.set_xlabel("log10(gap latency μs+1)"); ax.grid(axis="x", alpha=0.3)
ax.legend(loc="lower right", fontsize=9)
ax.set_title("Figure 38. All-condition summary: L1 gap distribution + duty cycle across every analyzed scenario\n"
             "Sorted by duty descending. Green dots = duty cycle (upper x-axis).",
             fontweight="bold")
plt.tight_layout()
plt.savefig(f"{FIG}/f38_all_conditions_summary.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f38_all_conditions_summary.png")

# ============================================================
# f39: NCU-vs-nsys cross reference — do slow-in-NCU kernels also have big gaps after them?
# ============================================================
# For each L1 kernel type in Part 2 NCU (MPSoff nrx_multi4), get mean DRAM %
# For same kernels in chain17 N=4 MPSoff, get mean gap-after
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

ncu = parse_ncu(f"{BASE}/chain18_p2_ncu/p2_ncu_nrx_multi4_MPSoff.ncu.csv")
ncu_dram = defaultdict(list); ncu_dur = defaultdict(list)
for k in ncu:
    name = k.get("_name",""); dram = k.get("dram__throughput.avg.pct_of_peak_sustained_elapsed", None)
    dur = k.get("gpu__time_active.sum", None)
    if name and dram is not None: ncu_dram[name].append(dram)
    if name and dur is not None:  ncu_dur[name].append(dur/1000)  # us

# Gap-after per kernel from chain17 N=4 MPSoff trace
kernels_ch = parse_trace(f"{BASE}/chain17_gapstats/cfgA_A_nrxN4_MPSoff_t1.gputrace.csv")
by_stream = defaultdict(list)
for s, d, n, st in kernels_ch: by_stream[st].append((s,d,n))
gap_after = defaultdict(list)
for strm, kl in by_stream.items():
    kl.sort()
    for j in range(len(kl)-1):
        g = kl[j+1][0] - (kl[j][0]+kl[j][1])
        if g > 0: gap_after[kl[j][2]].append(g/1000)  # us

# Cross-reference
xy = []
for name in ncu_dur:
    if name in gap_after and ncu_dur[name] and gap_after[name]:
        xy.append((name, np.median(ncu_dur[name]), np.median(gap_after[name]), np.median(ncu_dram.get(name,[0]))))
if xy:
    names_x, durs, gaps_after, drams = zip(*xy)
    fig, ax = plt.subplots(figsize=(12, 7))
    sc = ax.scatter(durs, gaps_after, c=drams, s=120, cmap="viridis", edgecolor="#111", linewidth=1)
    for n, x, y, _ in xy:
        short = n.split("<")[0].split("::")[-1][:20]
        ax.annotate(short, (x, y), fontsize=8, alpha=0.8, xytext=(3,3), textcoords="offset points")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Kernel duration (μs, from NCU on Full GPU)")
    ax.set_ylabel("Median gap AFTER this kernel type (μs, from chain17 N=4 MPSoff)")
    ax.set_title("Figure 39. Per-kernel-type: does a slow kernel also have a long gap after it?\n"
                 "Color = NCU DRAM %. Cluster patterns reveal which kernels drive the sync cost.",
                 fontweight="bold")
    cbar = plt.colorbar(sc); cbar.set_label("DRAM throughput (%)")
    ax.grid(alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(f"{FIG}/f39_ncu_vs_nsys_correlation.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved f39_ncu_vs_nsys_correlation.png")

# ============================================================
# f40: Part 4 partial (100/80 pct) — what CAN we see from the truncated sweep?
# ============================================================
p4_data = {}
for pct in [100, 80]:
    for wl in ["nrx4","ranai","memcpy4","embed4"]:
        files = sorted(glob.glob(f"{BASE}/chain18_gapstats/p4_{wl}_pct{pct}_t*.gputrace.csv"))
        s = compute_stats(files)
        if s: p4_data[(pct, wl)] = s
# Chain 17 Part B had pct 100/70/50/30 for nrx_multi4
# Compare Part 4 (pct 100/80) with those points
fig, ax = plt.subplots(figsize=(12, 6))
markers = {"nrx4":"o","ranai":"s","memcpy4":"^","embed4":"D"}
palette = {"nrx4":"#dc2626","ranai":"#f59e0b","memcpy4":"#3b82f6","embed4":"#10b981"}
for wl in ["nrx4","ranai","memcpy4","embed4"]:
    x=[]; y=[]
    for pct in sorted({p for (p,_) in p4_data.keys()}):
        if (pct,wl) in p4_data:
            x.append(pct); y.append(p4_data[(pct,wl)][3])  # duty
    if x:
        ax.plot(x, y, "-", marker=markers[wl], color=palette[wl], label=wl,
                linewidth=2.5, markersize=12)
ax.set_xlabel("CUDA_MPS_ACTIVE_THREAD_PERCENTAGE")
ax.set_ylabel("L1 duty cycle (%)")
ax.set_title("Figure 40. Part 4 partial fine MPS thread% sweep (pct=100, 80 only)\n"
             "Truncated by 2h budget but captures the top end of the sensitivity",
             fontweight="bold")
ax.grid(alpha=0.3); ax.legend(fontsize=11)
plt.tight_layout()
plt.savefig(f"{FIG}/f40_p4_partial_pct.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f40_p4_partial_pct.png")

# Dump
with open(f"{BASE}/deeper_analysis.json","w") as fp:
    json.dump({
        "heat_summary": [{"cond":h[0], "gap_med":h[1], "gap_p95":h[2], "gap_p99":h[3], "duty":h[4]} for h in heat],
        "p4_partial": {f"pct{p}_{w}": {"gap_med":v[0],"gap_p95":v[1],"gap_p99":v[2],"duty":v[3]} for (p,w),v in p4_data.items()},
    }, fp, indent=2, default=float)
print("Wrote deeper_analysis.json")
print(f"\nTotal analyzed conditions: {len(heat)}")
