#!/usr/bin/env python3
"""Chain 18 DEEP analysis — kernel-level, extended N, workload intensity, SLA.

Produces figures f28-f33:
  f28: Per-kernel-name duration comparison (which cuPHY kernels affected)
  f29: Extended N-sweep (Chain 17 N=1..8 + Part 3 N=5..16) — asymptote
  f30: Gap CDF log-log for all key conditions
  f31: Workload type comparison (nrx vs memcpy vs embed) at matched N
  f32: Part 5 vs Chain 17 kernel-launch-rate reconciliation
  f33: SLA budget vs 5G TTI (500us) across ALL conditions
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

def parse_trace(p, want_names=False):
    """Return {stream: [(start_ns, dur_ns, name_short)]}."""
    with open(p) as f:
        rdr = csv.reader(f); hdr=None
        for row in rdr:
            if row and row[0]=="Start (ns)": hdr=row; break
        if not hdr: return {} if not want_names else ({}, [])
        idx_s=hdr.index("Start (ns)"); idx_d=hdr.index("Duration (ns)")
        idx_g=hdr.index("GrdX"); idx_n=hdr.index("Name"); idx_st=hdr.index("Strm")
        by_stream = defaultdict(list); names = []
        for row in rdr:
            if len(row)<=idx_n or not row[idx_g].strip(): continue
            n = row[idx_n]
            if 'memcpy' in n.lower() or 'memset' in n.lower(): continue
            try:
                by_stream[row[idx_st]].append((int(row[idx_s]), int(row[idx_d]), n[:60]))
                names.append(n[:60])
            except: pass
    return (by_stream, names) if want_names else by_stream

def gap_arrays(by_stream):
    """Return (dur_arr, gap_arr, per_kernel_gap_dict)."""
    durs=[]; gaps=[]; per_kern_gap = defaultdict(list); per_kern_dur = defaultdict(list)
    for strm, kl in by_stream.items():
        kl.sort()
        for i,(s,d,n) in enumerate(kl):
            durs.append(d)
            per_kern_dur[n].append(d)
            if i>0:
                g = s - (kl[i-1][0]+kl[i-1][1])
                if g>0:
                    gaps.append(g)
                    per_kern_gap[kl[i-1][2]].append(g)  # gap after kernel[i-1]
    return np.array(durs), np.array(gaps), per_kern_dur, per_kern_gap

# ============================================================
# f28: Per-kernel-name duration analysis
# Compare each L1 kernel type across (baseline, CP-diverse, SP-diverse, SP-uniform)
# ============================================================
p8_conds = ["baseline","CPdiverse","CPuniform","SPdiverse","SPuniform"]
kern_dur_by_cond = defaultdict(lambda: defaultdict(list))
for c in p8_conds:
    for p in sorted(glob.glob(f"{BASE}/chain18_p8_gapstats/p8_{c}_t*.gputrace.csv")):
        bs = parse_trace(p)
        _, _, pkd, _ = gap_arrays(bs)
        for k, v in pkd.items():
            kern_dur_by_cond[c][k].extend(v)

# Identify top 8 most-common kernel names
all_names = defaultdict(int)
for c in p8_conds:
    for k, v in kern_dur_by_cond[c].items():
        all_names[k] += len(v)
top8 = [k for k,_ in sorted(all_names.items(), key=lambda x: -x[1])[:8]]

fig, ax = plt.subplots(figsize=(16, 7))
palette = ["#3b82f6","#10b981","#059669","#f59e0b","#dc2626"]
x = np.arange(len(top8))
w = 0.17
for i, c in enumerate(p8_conds):
    vals = []
    for k in top8:
        arr = kern_dur_by_cond[c].get(k, [])
        vals.append(np.median(arr)/1000 if arr else 0)
    ax.bar(x + (i-2)*w, vals, w, label=c, color=palette[i], alpha=0.85, edgecolor="#111", lw=0.5)
ax.set_xticks(x)
ax.set_xticklabels([k[:35]+".." for k in top8], rotation=25, ha="right", fontsize=9)
ax.set_ylabel("Kernel duration median (μs)")
ax.set_title("Figure 28. Per-cuPHY-kernel duration median across Part 8 scenarios\n"
             "Which specific L1 kernels slow down under diverse vs uniform AI pressure?",
             fontweight="bold")
ax.legend(loc="upper right", fontsize=10); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(f"{FIG}/f28_per_kernel_duration.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f28_per_kernel_duration.png")

# Also compute the ratio (SP-uniform vs baseline) per kernel — which kernels are most affected?
print("\n=== Per-kernel SP-uniform / baseline duration ratio ===")
ratios = []
for k in top8:
    b = np.median(kern_dur_by_cond["baseline"].get(k, [1]))
    s = np.median(kern_dur_by_cond["SPuniform"].get(k, [1]))
    ratios.append((k[:40], s/b, b/1000, s/1000))
for k, r, b, s in sorted(ratios, key=lambda x: -x[1]):
    print(f"  {r:6.2f}x  baseline={b:6.2f}us -> SP-uniform={s:6.2f}us  [{k}]")

# ============================================================
# f29: Extended N-sweep combined (Chain 17 N=1..8 + Part 3 N=5..16)
# ============================================================
n_data = {}  # (N, MPS) -> stats dict
# Chain 17 (12 files, N=1,2,3,4,6,8)
for N in [1,2,3,4,6,8]:
    for m in ["off","on"]:
        p = f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPS{m}_t1.gputrace.csv"
        if not os.path.exists(p): continue
        bs = parse_trace(p)
        d, g, _, _ = gap_arrays(bs)
        if len(d)>0 and len(g)>0:
            n_data[(N,m,"ch17")] = dict(duty=d.sum()/(d.sum()+g.sum())*100,
                                          gap_p95=np.percentile(g,95)/1000,
                                          gap_med=np.median(g)/1000,
                                          dur_med=np.median(d)/1000,
                                          n=len(d))
# Part 3 (N=5,7,10,12,16, 3 trials each)
for N in [5,7,10,12,16]:
    for m in ["off","on"]:
        files = sorted(glob.glob(f"{BASE}/chain18_gapstats/p3_nrxN{N}_MPS{m}_t*.gputrace.csv"))
        if not files: continue
        alld=[]; allg=[]
        for p in files:
            bs = parse_trace(p)
            d, g, _, _ = gap_arrays(bs)
            alld.extend(d.tolist()); allg.extend(g.tolist())
        if alld and allg:
            d=np.array(alld); g=np.array(allg)
            n_data[(N,m,"p3")] = dict(duty=d.sum()/(d.sum()+g.sum())*100,
                                       gap_p95=np.percentile(g,95)/1000,
                                       gap_med=np.median(g)/1000,
                                       dur_med=np.median(d)/1000,
                                       n=len(d))
# Merge Chain 17 + Part 3 into single N series
all_ns = sorted({N for (N,_,_) in n_data.keys()})
fig, axes = plt.subplots(1, 3, figsize=(20, 6))
for m, color in [("off","#dc2626"),("on","#10b981")]:
    ys_duty=[]; ys_gap=[]; ys_dur=[]; xs=[]
    for N in all_ns:
        # prefer chain17 for N=1..8, part3 for N>8
        rec = n_data.get((N,m,"ch17")) or n_data.get((N,m,"p3"))
        if rec:
            xs.append(N); ys_duty.append(rec["duty"]); ys_gap.append(rec["gap_p95"]); ys_dur.append(rec["dur_med"])
    axes[0].plot(xs, ys_duty, "o-", color=color, label=f"MPS {m}", linewidth=2.5, markersize=9)
    axes[1].plot(xs, ys_gap,  "o-", color=color, label=f"MPS {m}", linewidth=2.5, markersize=9)
    axes[2].plot(xs, ys_dur,  "o-", color=color, label=f"MPS {m}", linewidth=2.5, markersize=9)
axes[0].axvspan(6, 8.5, alpha=0.15, color="#eab308", label="Chain 17 breakdown zone")
axes[0].axvspan(10, 16.5, alpha=0.1, color="#f97316", label="Extended breakdown zone")
axes[0].set_ylabel("Duty cycle (%)"); axes[0].set_title("L1 duty cycle asymptote", fontweight="bold")
axes[1].set_ylabel("gap p95 (μs)"); axes[1].set_yscale("log"); axes[1].set_title("Gap tail (p95) — log scale", fontweight="bold")
axes[2].set_ylabel("kernel dur median (μs)"); axes[2].set_title("Kernel duration inflation", fontweight="bold")
for ax in axes:
    ax.set_xlabel("N (concurrent NRx processes)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="best", fontsize=9)
plt.suptitle("Figure 29. Extended N-sweep (Chain 17 N=1..8 + Part 3 N=5..16) on Config A\n"
             "Does MPSon breakdown asymptote or continue degrading?",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f29_extended_nsweep.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f29_extended_nsweep.png")

# ============================================================
# f30: Gap CDF log-log for key conditions
# ============================================================
key_conds = [
    ("L1 alone", "#3b82f6", f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv"),
    ("N=1 MPSon", "#059669", f"{BASE}/chain17_gapstats/cfgA_A_nrxN1_MPSon_t1.gputrace.csv"),
    ("N=4 MPSon", "#10b981", f"{BASE}/chain17_gapstats/cfgA_A_nrxN4_MPSon_t1.gputrace.csv"),
    ("N=6 MPSon", "#f59e0b", f"{BASE}/chain17_gapstats/cfgA_A_nrxN6_MPSon_t1.gputrace.csv"),
    ("N=8 MPSon", "#f97316", f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSon_t1.gputrace.csv"),
    ("N=1 MPSoff", "#a3a3a3", f"{BASE}/chain17_gapstats/cfgA_A_nrxN1_MPSoff_t1.gputrace.csv"),
    ("N=8 MPSoff", "#dc2626", f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSoff_t1.gputrace.csv"),
]
fig, ax = plt.subplots(figsize=(12, 7))
for label, color, path in key_conds:
    if not os.path.exists(path): continue
    bs = parse_trace(path)
    _, g, _, _ = gap_arrays(bs)
    if len(g)==0: continue
    g_us = np.sort(g/1000)
    cdf = np.arange(1, len(g_us)+1) / len(g_us)
    ax.plot(g_us, 1-cdf, label=label, color=color, linewidth=2, alpha=0.9)
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Inter-kernel gap (μs, log scale)")
ax.set_ylabel("P(gap > x)  [1-CDF, log scale]")
ax.set_title("Figure 30. Gap survival function (1-CDF) log-log across N-sweep\n"
             "Tail heaviness: MPS on flattens tail; MPS off has heavy tail at all N",
             fontweight="bold")
ax.grid(alpha=0.3, which="both"); ax.legend(loc="lower left", fontsize=10)
ax.set_xlim(0.5, 30000)
plt.tight_layout()
plt.savefig(f"{FIG}/f30_gap_cdf_loglog.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f30_gap_cdf_loglog.png")

# ============================================================
# f31: Multi-workload comparison at matched N (Part 3)
# ============================================================
wl_data = {}
for wl in ["nrx","memcpy","embed"]:
    for N in [1,2,4,6,8]:
        for m in ["on"]:  # focus MPS on
            files = sorted(glob.glob(f"{BASE}/chain18_gapstats/p3_{wl}N{N}_MPS{m}_t*.gputrace.csv"))
            if not files: continue
            alld=[]; allg=[]
            for p in files:
                bs = parse_trace(p)
                d, g, _, _ = gap_arrays(bs)
                alld.extend(d.tolist()); allg.extend(g.tolist())
            if alld and allg:
                d=np.array(alld); g=np.array(allg)
                wl_data[(wl,N,m)] = dict(duty=d.sum()/(d.sum()+g.sum())*100,
                                          gap_p95=np.percentile(g,95)/1000)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
Ns_wl = [1,2,4,6,8]
palette = {"nrx":"#dc2626","memcpy":"#3b82f6","embed":"#10b981"}
markers = {"nrx":"o","memcpy":"s","embed":"^"}
for wl in ["nrx","memcpy","embed"]:
    duty = [wl_data.get((wl,N,"on"),{}).get("duty",0) for N in Ns_wl]
    gp95 = [wl_data.get((wl,N,"on"),{}).get("gap_p95",0) for N in Ns_wl]
    ax1.plot(Ns_wl, duty, "-", marker=markers[wl], color=palette[wl],
             label=wl, linewidth=2.5, markersize=10)
    ax2.plot(Ns_wl, gp95, "-", marker=markers[wl], color=palette[wl],
             label=wl, linewidth=2.5, markersize=10)
ax1.set_xlabel("N (concurrent processes)"); ax1.set_ylabel("L1 duty cycle (%)")
ax1.set_title("L1 duty by AI workload type (MPS on)", fontweight="bold"); ax1.grid(alpha=0.3); ax1.legend()
ax2.set_xlabel("N (concurrent processes)"); ax2.set_ylabel("L1 gap p95 (μs)")
ax2.set_yscale("log")
ax2.set_title("L1 gap p95 by AI workload type (MPS on)", fontweight="bold"); ax2.grid(alpha=0.3, which="both"); ax2.legend()
plt.suptitle("Figure 31. Workload-type dependency of L1 sync (Part 3, MPS on)\n"
             "nrx (compute-heavy) vs memcpy (bandwidth-only) vs embed (light kernel)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f31_workload_type_comparison.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f31_workload_type_comparison.png")

# ============================================================
# f32: Part 5 vs Chain 17 kernel-launch-rate reconciliation
# Question: why did Part 5 proc_8 (ranai_mix) NOT break like Chain 17 nrx_multi8?
# Answer: measure per-process kernel launches per second
# ============================================================
def total_kernels_per_sec(path):
    """Approximate: kernels total / trace_wall_time_seconds."""
    bs = parse_trace(path)
    all_starts=[]; all_ends=[]
    for strm, kl in bs.items():
        for s,d,n in kl:
            all_starts.append(s); all_ends.append(s+d)
    if not all_starts: return 0
    wall_s = (max(all_ends) - min(all_starts))/1e9
    return len(all_starts)/max(wall_s,1e-9)

# Aggregate for L1 kernels on the profile side (which is what our nsys traces show)
p5_rate = {}
for BEAM in [4,8,16,32]:
    files = sorted(glob.glob(f"{BASE}/chain18_p5_gapstats/p5_thrOnly_beam{BEAM}_t*.gputrace.csv"))
    if files: p5_rate[f"thr_{BEAM+6}"] = np.mean([total_kernels_per_sec(f) for f in files])
for N in [1,2,4,8]:
    files = sorted(glob.glob(f"{BASE}/chain18_p5_gapstats/p5_procOnly_N{N}_t*.gputrace.csv"))
    if files: p5_rate[f"proc_{N}"] = np.mean([total_kernels_per_sec(f) for f in files])
ch17_rate = {}
for N in [1,2,3,4,6,8]:
    p = f"{BASE}/chain17_gapstats/cfgA_A_nrxN{N}_MPSon_t1.gputrace.csv"
    if os.path.exists(p): ch17_rate[f"ch17_N{N}"] = total_kernels_per_sec(p)

print("\n=== L1 kernel launch rate (kernels/sec) ===")
for k, v in {**p5_rate, **ch17_rate}.items():
    print(f"  {k:<15} {v:.0f}")

fig, ax = plt.subplots(figsize=(12, 6))
# Chain 17 series
ch17_x = [1,2,3,4,6,8]; ch17_y = [ch17_rate.get(f"ch17_N{N}",0) for N in ch17_x]
ax.plot(ch17_x, ch17_y, "o-", color="#dc2626", label="Chain 17: N× identical NRx procs", linewidth=2.5, markersize=10)
# Part 5 proc series
p5_x = [1,2,4,8]; p5_y = [p5_rate.get(f"proc_{N}",0) for N in p5_x]
ax.plot(p5_x, p5_y, "s-", color="#10b981", label="Part 5: N× ranai_mix (14 threads each)", linewidth=2.5, markersize=10)
ax.set_xlabel("N (concurrent AI processes)")
ax.set_ylabel("L1 kernel launch rate (kernels/sec) — proxy for effective launch pressure")
ax.set_title("Figure 32. L1 launch rate: identical NRx replicas vs ranai_mix processes\n"
             "Why Chain 17 breaks at N=6 but Part 5 proc_8 doesn't: launch-rate ceiling not reached",
             fontweight="bold")
ax.grid(alpha=0.3); ax.legend()
plt.tight_layout()
plt.savefig(f"{FIG}/f32_launch_rate_reconciliation.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f32_launch_rate_reconciliation.png")

# ============================================================
# f33: SLA budget analysis — 5G TTI 500us across all conditions
# ============================================================
# Compute effective per-slot latency estimate for each condition
sla_conds = []
# baseline
for path, label in [
    (f"{BASE}/chain17_gapstats/L1alone_cfgA_SP0_baseline_t1.gputrace.csv", "L1 alone"),
    (f"{BASE}/chain18_p8_gapstats/p8_CPdiverse_t1.gputrace.csv", "CP + 6 diverse AI"),
    (f"{BASE}/chain18_p8_gapstats/p8_CPuniform_t1.gputrace.csv", "CP + 6× NRx"),
    (f"{BASE}/chain18_p8_gapstats/p8_SPdiverse_t1.gputrace.csv", "SP + 6 diverse AI"),
    (f"{BASE}/chain18_p8_gapstats/p8_SPuniform_t1.gputrace.csv", "SP + 6× NRx"),
    (f"{BASE}/chain17_gapstats/cfgA_A_nrxN6_MPSon_t1.gputrace.csv", "SP N=6 MPSon"),
    (f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSon_t1.gputrace.csv", "SP N=8 MPSon"),
    (f"{BASE}/chain17_gapstats/cfgA_A_nrxN8_MPSoff_t1.gputrace.csv", "SP N=8 MPSoff"),
]:
    if not os.path.exists(path): continue
    bs = parse_trace(path)
    d, g, _, _ = gap_arrays(bs)
    if len(d)==0: continue
    # A "slot" in cuPHY is approximated as 100 kernels (cuphy PUSCH pipeline ~ tens of kernels per slot)
    per_slot = 100  # kernels per slot approximation
    # median per-kernel time = dur + gap
    per_kernel_us = (np.median(d) + np.median(g))/1000
    per_slot_us_est = per_kernel_us * per_slot
    p95_per_kernel = (np.percentile(d,95) + np.percentile(g,95))/1000
    p95_per_slot   = p95_per_kernel * per_slot
    sla_conds.append((label, per_slot_us_est, p95_per_slot))

labels, med_slot, p95_slot = zip(*sla_conds)
x = np.arange(len(labels))
fig, ax = plt.subplots(figsize=(15, 6.5))
w = 0.4
ax.bar(x-w/2, med_slot, w, color="#3b82f6", alpha=0.85, label="median per-slot", edgecolor="#111")
ax.bar(x+w/2, p95_slot, w, color="#dc2626", alpha=0.85, label="p95 per-slot", edgecolor="#111")
ax.axhline(500, color="#111", linestyle="--", linewidth=2, alpha=0.7, label="5G TTI (500μs)")
ax.axhline(1000, color="#f59e0b", linestyle=":", linewidth=1.5, alpha=0.7, label="15kHz TTI (1000μs)")
for i, (v1, v2) in enumerate(zip(med_slot, p95_slot)):
    ax.text(i-w/2, v1+30, f"{v1:.0f}", ha="center", fontsize=8, rotation=0)
    ax.text(i+w/2, v2+30, f"{v2:.0f}", ha="center", fontsize=8, rotation=0)
ax.set_xticks(x); ax.set_xticklabels(labels, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Estimated per-slot L1 latency (μs, 100 kernels/slot proxy)")
ax.set_title("Figure 33. Estimated 5G L1 per-slot latency vs TTI budget\n"
             "Median and p95 across all deployment scenarios — where does SLA break?",
             fontweight="bold")
ax.grid(axis="y", alpha=0.3); ax.legend(loc="upper left", fontsize=10)
ax.set_ylim(0, max(p95_slot)*1.15)
plt.tight_layout()
plt.savefig(f"{FIG}/f33_sla_budget.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f33_sla_budget.png")

# Save aggregated deep stats
deep = {
    "per_kernel_dur_ratios": [{"kernel":k,"ratio":r,"baseline_us":b,"sp_uniform_us":s} for k,r,b,s in ratios],
    "n_sweep_extended": {f"N{N}_MPS{m}": {k:float(v) for k,v in rec.items()} for (N,m,src), rec in n_data.items()},
    "workload_type": {f"{w}_N{n}_MPSon": {k:float(v) for k,v in rec.items()} for (w,n,m), rec in wl_data.items()},
    "p5_launch_rate": p5_rate,
    "chain17_launch_rate": ch17_rate,
    "sla_analysis": [{"label":l,"median_slot_us":m,"p95_slot_us":p} for l,m,p in sla_conds],
}
with open(f"{BASE}/deep_analysis.json","w") as fp:
    json.dump(deep, fp, indent=2, default=float)
print("\nWrote deep_analysis.json")
