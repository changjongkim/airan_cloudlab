#!/usr/bin/env python3
"""Chain 18 unified analysis — Parts 5/7/8 + Part 2b (NCU MPSon).

Extends analyze_kernel_gaps.py with:
  - Part 5: multi-thread vs multi-process controlled → f25
  - Part 7 stat: 10-trial statistical at N=5,6,7 breakdown zone → f26
  - Part 8: realistic AI-RAN stack (CP/SP × diverse/uniform + baseline) → f24
  - Part 2b: NCU DRAM/SM under MPS on → f27 (comparison with Part 2 MPSoff)
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
    """Return dict {stream: [(start_ns, dur_ns)]} of kernel rows (excluding memcpy)."""
    with open(p) as f:
        rdr = csv.reader(f); hdr=None
        for row in rdr:
            if row and row[0]=="Start (ns)": hdr=row; break
        if not hdr: return {}
        idx_s=hdr.index("Start (ns)"); idx_d=hdr.index("Duration (ns)")
        idx_g=hdr.index("GrdX"); idx_n=hdr.index("Name"); idx_st=hdr.index("Strm")
        by_stream = defaultdict(list)
        for row in rdr:
            if len(row)<=idx_n or not row[idx_g].strip(): continue
            if 'memcpy' in row[idx_n].lower() or 'memset' in row[idx_n].lower(): continue
            try: by_stream[row[idx_st]].append((int(row[idx_s]), int(row[idx_d])))
            except: pass
    return by_stream

def gap_stats_from_files(files):
    """Aggregate stats from a list of gputrace CSVs."""
    all_dur=[]; all_gap=[]
    for p in files:
        bs = parse_trace(p)
        for strm, kl in bs.items():
            kl.sort()
            for i,(s,d) in enumerate(kl):
                all_dur.append(d)
                if i>0:
                    g = s - (kl[i-1][0]+kl[i-1][1])
                    if g>0: all_gap.append(g)
    if not all_dur: return None
    dur=np.array(all_dur); gap=np.array(all_gap)
    return dict(n=len(dur), dur_med=float(np.median(dur)), dur_p95=float(np.percentile(dur,95)),
                gap_med=float(np.median(gap)), gap_p95=float(np.percentile(gap,95)),
                gap_p99=float(np.percentile(gap,99)),
                duty=float(dur.sum()/(dur.sum()+gap.sum())),
                slot_time_us=float((np.median(dur)+np.median(gap))/1000))

# ============================================================
# Part 8 — Realistic AI-RAN stack (5 scenarios × 3 trials)
# ============================================================
p8_conds = ["baseline","CPdiverse","CPuniform","SPdiverse","SPuniform"]
p8_labels = {"baseline":"L1 alone",
             "CPdiverse":"CP + diverse (Qwen+Whisper+BERT+NRx+CSI+Beam)",
             "CPuniform":"CP + 6× NRx",
             "SPdiverse":"SP + diverse (6 different)",
             "SPuniform":"SP + 6× NRx"}
p8 = {}
GAPS = f"{BASE}/chain18_p8_gapstats"
for c in p8_conds:
    files = sorted(glob.glob(f"{GAPS}/p8_{c}_t*.gputrace.csv"))
    s = gap_stats_from_files(files)
    if s: p8[c] = s
    print(f"P8 {c:<12}: kernels={s['n']}, dur_med={s['dur_med']/1000:.2f}us, gap_med={s['gap_med']/1000:.2f}us, duty={s['duty']*100:.2f}%")

# ---- f24: Part 8 comparison ----
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
conds_present = [c for c in p8_conds if c in p8]
labels = [p8_labels[c] for c in conds_present]
palette = ["#3b82f6","#10b981","#059669","#f59e0b","#dc2626"]
pos = np.arange(len(conds_present))

# Duty cycle
duty = [p8[c]["duty"]*100 for c in conds_present]
bars = axes[0].bar(pos, duty, color=palette, alpha=0.85, edgecolor="#111", lw=1.2)
for b, v in zip(bars, duty):
    axes[0].text(b.get_x()+b.get_width()/2, v+1, f"{v:.1f}%", ha="center", fontsize=9)
axes[0].set_xticks(pos); axes[0].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
axes[0].set_ylabel("L1 duty cycle (%)")
axes[0].set_title("L1 GPU duty cycle", fontweight="bold"); axes[0].grid(axis="y", alpha=0.3)
if "baseline" in p8:
    axes[0].axhline(p8["baseline"]["duty"]*100, color="#111", linestyle="--", alpha=0.5, label="baseline")

# Gap p95 (SLA-relevant)
gp95 = [p8[c]["gap_p95"]/1000 for c in conds_present]
bars = axes[1].bar(pos, gp95, color=palette, alpha=0.85, edgecolor="#111", lw=1.2)
for b, v in zip(bars, gp95):
    axes[1].text(b.get_x()+b.get_width()/2, v+15, f"{v:.0f}μs", ha="center", fontsize=9)
axes[1].set_xticks(pos); axes[1].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
axes[1].set_ylabel("L1 inter-kernel gap p95 (μs)")
axes[1].set_title("Tail latency (p95)", fontweight="bold"); axes[1].grid(axis="y", alpha=0.3)

# Effective slot time (kernel dur + gap median)
slot = [p8[c]["slot_time_us"] for c in conds_present]
bars = axes[2].bar(pos, slot, color=palette, alpha=0.85, edgecolor="#111", lw=1.2)
for b, v in zip(bars, slot):
    axes[2].text(b.get_x()+b.get_width()/2, v+2, f"{v:.1f}μs", ha="center", fontsize=9)
axes[2].set_xticks(pos); axes[2].set_xticklabels(labels, rotation=25, ha="right", fontsize=9)
axes[2].set_ylabel("Per-kernel budget: dur_med + gap_med (μs)")
axes[2].set_title("Effective L1 kernel budget", fontweight="bold"); axes[2].grid(axis="y", alpha=0.3)
axes[2].axhline(500, color="#dc2626", linestyle="--", alpha=0.7, label="5G TTI 500μs")
axes[2].legend()

plt.suptitle("Figure 24. Chain 18 Part 8 — Realistic AI-RAN diverse workload stack (Config A)\n"
             "Cross-partition preserves baseline under DIVERSE deployment; same-partition breaks",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f24_p8_realistic_stack.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f24_p8_realistic_stack.png")

# ============================================================
# Part 5 — Multi-thread vs multi-process controlled
# ============================================================
p5 = {}
GAPS5 = f"{BASE}/chain18_p5_gapstats"
# thrOnly beam ∈ {4,8,16,32}
for BEAM in [4,8,16,32]:
    files = sorted(glob.glob(f"{GAPS5}/p5_thrOnly_beam{BEAM}_t*.gputrace.csv"))
    s = gap_stats_from_files(files)
    if s: p5[f"thr_{BEAM+6}"] = {**s, "total_threads": BEAM+6, "cuda_ctx": 1}
# procOnly N ∈ {1,2,4,8}
for N in [1,2,4,8]:
    files = sorted(glob.glob(f"{GAPS5}/p5_procOnly_N{N}_t*.gputrace.csv"))
    s = gap_stats_from_files(files)
    if s: p5[f"proc_{N}"] = {**s, "total_threads": N*14, "cuda_ctx": N}

# ---- f25: Part 5 ----
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
thr_pts = sorted([(v["total_threads"], v["duty"]*100, v["gap_p95"]/1000)
                   for k,v in p5.items() if k.startswith("thr_")])
proc_pts = sorted([(v["total_threads"], v["duty"]*100, v["gap_p95"]/1000)
                    for k,v in p5.items() if k.startswith("proc_")])
if thr_pts:
    x, dy, gp = zip(*thr_pts)
    ax1.plot(x, dy, "o-", color="#3b82f6", label="Multi-thread (1 CUDA ctx)", linewidth=2.5, markersize=10)
    ax2.plot(x, gp, "o-", color="#3b82f6", label="Multi-thread (1 CUDA ctx)", linewidth=2.5, markersize=10)
if proc_pts:
    x, dy, gp = zip(*proc_pts)
    ax1.plot(x, dy, "s-", color="#dc2626", label="Multi-process (N CUDA ctxs)", linewidth=2.5, markersize=10)
    ax2.plot(x, gp, "s-", color="#dc2626", label="Multi-process (N CUDA ctxs)", linewidth=2.5, markersize=10)
for ax, ylabel, title in [(ax1, "L1 duty cycle (%)", "L1 duty cycle"),
                           (ax2, "L1 gap p95 (μs)", "L1 gap tail (p95)")]:
    ax.set_xlabel("Total concurrent AI threads across (thread) or processes")
    ax.set_ylabel(ylabel); ax.set_title(title, fontweight="bold")
    ax.grid(alpha=0.3); ax.legend()
plt.suptitle("Figure 25. Chain 18 Part 5 — Multi-thread vs multi-process controlled\n"
             "Same total AI work: 1 process (threads) vs N processes (contexts)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f25_p5_thread_vs_process.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f25_p5_thread_vs_process.png")

# ============================================================
# Part 7 statistical — 10 trials at N=5,6,7 (breakdown zone robustness)
# ============================================================
p7 = {}
GAPS7 = f"{BASE}/chain18_p7_gapstats"
for N in [5,6,7]:
    # Group per-trial to get inter-trial variability
    files = sorted(glob.glob(f"{GAPS7}/p7_stat_nrxN{N}_MPSon_t*.gputrace.csv"))
    per_trial = []
    for p in files:
        bs = parse_trace(p)
        durs=[]; gaps=[]
        for strm, kl in bs.items():
            kl.sort()
            for i,(s,d) in enumerate(kl):
                durs.append(d)
                if i>0:
                    g = s - (kl[i-1][0]+kl[i-1][1])
                    if g>0: gaps.append(g)
        if durs and gaps:
            per_trial.append(dict(
                duty=sum(durs)/(sum(durs)+sum(gaps))*100,
                gap_p95=float(np.percentile(gaps,95))/1000,
                gap_p99=float(np.percentile(gaps,99))/1000,
                dur_med=float(np.median(durs))/1000,
                gap_med=float(np.median(gaps))/1000,
            ))
    p7[N] = per_trial
    if per_trial:
        duty_arr = np.array([t["duty"] for t in per_trial])
        gp95_arr = np.array([t["gap_p95"] for t in per_trial])
        print(f"P7 N={N} (n={len(per_trial)}): duty={duty_arr.mean():.1f}±{duty_arr.std():.1f}%  gap_p95={gp95_arr.mean():.0f}±{gp95_arr.std():.0f}μs")

# ---- f26: Part 7 stat boxplots ----
fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))
Ns = [5,6,7]
duty_data = [[t["duty"] for t in p7[N]] if p7.get(N) else [] for N in Ns]
gp95_data = [[t["gap_p95"] for t in p7[N]] if p7.get(N) else [] for N in Ns]

palette = ["#10b981","#f59e0b","#dc2626"]
bp1 = axes[0].boxplot(duty_data, positions=range(len(Ns)), widths=0.6, patch_artist=True,
                        medianprops=dict(color="#111",lw=1.5))
bp2 = axes[1].boxplot(gp95_data, positions=range(len(Ns)), widths=0.6, patch_artist=True,
                        medianprops=dict(color="#111",lw=1.5))
for bp, colors in [(bp1, palette), (bp2, palette)]:
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c); patch.set_alpha(0.7)
# scatter individual trials
for i, N in enumerate(Ns):
    if p7.get(N):
        axes[0].scatter([i]*len(duty_data[i]), duty_data[i], color="#111", s=25, alpha=0.6, zorder=3)
        axes[1].scatter([i]*len(gp95_data[i]), gp95_data[i], color="#111", s=25, alpha=0.6, zorder=3)

axes[0].set_xticks(range(len(Ns))); axes[0].set_xticklabels([f"N={N}" for N in Ns])
axes[0].set_ylabel("L1 duty cycle (%)"); axes[0].set_title("Duty cycle across 10 trials", fontweight="bold")
axes[0].grid(axis="y", alpha=0.3)
axes[1].set_xticks(range(len(Ns))); axes[1].set_xticklabels([f"N={N}" for N in Ns])
axes[1].set_ylabel("gap p95 (μs)"); axes[1].set_title("Gap p95 across 10 trials", fontweight="bold")
axes[1].grid(axis="y", alpha=0.3)
plt.suptitle("Figure 26. Chain 18 Part 7 — 10-trial statistical at MPSon breakdown zone (N=5,6,7)\n"
             "Distribution of L1 duty cycle and tail gap — is the breakdown deterministic?",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f26_p7_statistical.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f26_p7_statistical.png")

# ============================================================
# Part 2b — NCU MPS on (using --mps client)
# ============================================================
p2b = {}
NCU2B = f"{BASE}/chain18_p2b_ncu_mps"

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
            try: kernels[kid][mn] = float(d.get("Metric Value","").replace(",",""))
            except: pass
    return list(kernels.values())

METS = {
    "dram_bw":  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_act":   "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram_bytes": "dram__bytes.sum",
    "l2_read":  "lts__t_sectors_op_read.sum",
    "l2_write": "lts__t_sectors_op_write.sum",
}
CONDS2B = ["idle","nrx","memcpy_loop","embed_lookup","ranai_mix","nrx_multi4"]
for wl in CONDS2B:
    p = f"{NCU2B}/p2b_ncu_{wl}_MPSon.ncu.csv"
    if not os.path.exists(p) or os.path.getsize(p) < 500: continue
    kernels = parse_ncu(p)
    if not kernels: continue
    s = {}
    for k, m in METS.items():
        vals = [x[m] for x in kernels if m in x]
        if vals:
            s[k] = dict(mean=float(np.mean(vals)), p95=float(np.percentile(vals,95)),
                        max=float(np.max(vals)), values=vals)
    p2b[wl] = s
    print(f"P2b {wl:<15}: kernels={len(kernels)}, DRAM_mean={s.get('dram_bw',{}).get('mean',0):.2f}%, SM={s.get('sm_act',{}).get('mean',0):.2f}%")

# ---- f27: Part 2b vs Part 2 (MPS on vs off) ----
if p2b:
    # Also reload Part 2 MPSoff for comparison
    P2NCU = f"{BASE}/chain18_p2_ncu"
    p2 = {}
    for wl in CONDS2B:
        p = f"{P2NCU}/p2_ncu_{wl}_MPSoff.ncu.csv"
        if not os.path.exists(p) or os.path.getsize(p) < 500: continue
        kernels = parse_ncu(p)
        if not kernels: continue
        s = {}
        for k, m in METS.items():
            vals = [x[m] for x in kernels if m in x]
            if vals:
                s[k] = dict(mean=float(np.mean(vals)), p95=float(np.percentile(vals,95)),
                            max=float(np.max(vals)), values=vals)
        p2[wl] = s
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
    conds = [wl for wl in CONDS2B if wl in p2b and wl in p2]
    pos = np.arange(len(conds))
    w = 0.35
    p2_vals  = [p2[wl]["dram_bw"]["mean"] for wl in conds]
    p2b_vals = [p2b[wl]["dram_bw"]["mean"] for wl in conds]
    ax1.bar(pos-w/2, p2_vals,  w, label="MPS off (Part 2)",  color="#dc2626", alpha=0.85, edgecolor="#111")
    ax1.bar(pos+w/2, p2b_vals, w, label="MPS on  (Part 2b)", color="#10b981", alpha=0.85, edgecolor="#111")
    ax1.set_xticks(pos); ax1.set_xticklabels(conds, rotation=25, ha="right")
    ax1.set_ylabel("L1 kernel DRAM throughput (% of peak)")
    ax1.set_title("DRAM utilization: MPS off vs on", fontweight="bold")
    ax1.grid(axis="y", alpha=0.3); ax1.legend()

    p2_sm  = [p2[wl]["sm_act"]["mean"] for wl in conds]
    p2b_sm = [p2b[wl]["sm_act"]["mean"] for wl in conds]
    ax2.bar(pos-w/2, p2_sm,  w, label="MPS off (Part 2)",  color="#dc2626", alpha=0.85, edgecolor="#111")
    ax2.bar(pos+w/2, p2b_sm, w, label="MPS on  (Part 2b)", color="#10b981", alpha=0.85, edgecolor="#111")
    ax2.set_xticks(pos); ax2.set_xticklabels(conds, rotation=25, ha="right")
    ax2.set_ylabel("L1 kernel SM active (%)")
    ax2.set_title("SM utilization: MPS off vs on", fontweight="bold")
    ax2.grid(axis="y", alpha=0.3); ax2.legend()

    plt.suptitle("Figure 27. Chain 18 Part 2b — NCU DRAM/SM under MPS on (--mps client)\n"
                 "Direct comparison: does MPS on change per-kernel memory pattern?",
                 fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(f"{FIG}/f27_p2b_ncu_mpson.png", dpi=140, bbox_inches='tight')
    plt.close()
    print("saved f27_p2b_ncu_mpson.png")

# ============================================================
# Dump combined stats
# ============================================================
summary = {
    "part8": {c:{k:v for k,v in s.items() if k!="values"} for c,s in p8.items()},
    "part5": {c:{k:v for k,v in s.items() if k!="values"} for c,s in p5.items()},
    "part7_stat": p7,
    "part2b": {c:{m:{kk:vv for kk,vv in val.items() if kk!="values"} for m,val in s.items()} for c,s in p2b.items()},
}
with open(f"{BASE}/chain18_all_stats.json","w") as fp:
    json.dump(summary, fp, indent=2, default=float)
print("Wrote chain18_all_stats.json")
