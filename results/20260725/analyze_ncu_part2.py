#!/usr/bin/env python3
"""Chain 18 Part 2 — NCU DRAM/SM analysis of L1 kernels on Full GPU.

Extracts per-kernel DRAM throughput & SM utilization when L1 runs
alone (baseline) vs. with each interfering workload (MPS off).
Directly answers: "Is HBM bandwidth the sync driver?"

MPSon runs failed because NCU needs --mps client — those are being
redone by Part 2b.

Output:
  - ncu_stats.json
  - f13_ncu_dram_by_workload.png
  - f14_ncu_sm_by_workload.png
"""
import os, csv, json, glob
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
NCU  = os.path.join(BASE, "chain18_p2_ncu")
FIG  = os.path.join(BASE, "figures", "comprehensive")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

WORKLOADS_ORDER = ["idle", "nrx", "memcpy_loop", "embed_lookup", "ranai_mix", "nrx_multi4"]
WORKLOADS_LABEL = {
    "idle":         "L1 alone",
    "nrx":          "+ NRx (1 proc)",
    "memcpy_loop":  "+ memcpy",
    "embed_lookup": "+ embed lookup",
    "ranai_mix":    "+ RAN-AI mix (14 thr)",
    "nrx_multi4":   "+ 4× NRx procs",
}
KEY_METRICS = {
    "dram_bw":  "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm_act":   "smsp__cycles_active.avg.pct_of_peak_sustained_elapsed",
    "dram_bytes": "dram__bytes.sum",
    "l2_sectors": "l2_tex__t_sectors.sum",
    "gpu_time":   "gpu__time_active.sum",
}

def parse_ncu(path):
    """Return list of {kernel, metric->value} dicts."""
    rows = []
    with open(path) as f:
        rdr = csv.reader(f)
        hdr = None
        cur_kernel = None
        cur = {}
        for row in rdr:
            if not row or row[0].startswith("==PROF==") or row[0].startswith("==ERROR=="): continue
            if row[0] == "ID": hdr = row; continue
            if hdr is None: continue
            d = dict(zip(hdr, row))
            kid = (d.get("ID",""), d.get("Kernel Name","")[:80])
            if kid != cur_kernel:
                if cur:
                    rows.append(cur)
                cur = {"id": kid[0], "kernel": kid[1]}
                cur_kernel = kid
            mn = d.get("Metric Name","")
            mv = d.get("Metric Value","").replace(",","")
            try: cur[mn] = float(mv)
            except: pass
        if cur: rows.append(cur)
    return rows

stats = {}
for wl in WORKLOADS_ORDER:
    p = f"{NCU}/p2_ncu_{wl}_MPSoff.ncu.csv"
    if not os.path.exists(p) or os.path.getsize(p) < 500: continue
    kernels = parse_ncu(p)
    if not kernels: continue
    s = {}
    for key, mname in KEY_METRICS.items():
        vals = [k[mname] for k in kernels if mname in k]
        if vals:
            arr = np.array(vals)
            s[key] = {"mean": float(arr.mean()), "median": float(np.median(arr)),
                       "p95": float(np.percentile(arr, 95)), "max": float(arr.max()),
                       "n": len(arr), "values": vals}
    stats[wl] = s
    print(f"{wl}: {len(kernels)} kernels, DRAM_BW mean={s.get('dram_bw',{}).get('mean',0):.1f}%, SM mean={s.get('sm_act',{}).get('mean',0):.1f}%")

with open(f"{BASE}/ncu_stats.json","w") as fp:
    json.dump({k:{m:{kk:vv for kk,vv in val.items() if kk!='values'} for m,val in s.items()} for k,s in stats.items()},
              fp, indent=2)

# --------------------------------------------------------------
# Figure 13 — DRAM BW % by workload (per-kernel dist)
# --------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
present = [wl for wl in WORKLOADS_ORDER if wl in stats and "dram_bw" in stats[wl]]
labels = [WORKLOADS_LABEL[wl] for wl in present]
positions = np.arange(len(present))

# Box plot of DRAM BW across L1 kernels for each condition
dram_data = [stats[wl]["dram_bw"]["values"] for wl in present]
bp = ax1.boxplot(dram_data, positions=positions, widths=0.6, patch_artist=True,
                 medianprops=dict(color="#111", lw=1.5))
palette = ["#3b82f6","#10b981","#f59e0b","#f97316","#8b5cf6","#dc2626"]
for patch, c in zip(bp["boxes"], palette[:len(present)]):
    patch.set_facecolor(c); patch.set_alpha(0.7)
ax1.set_xticks(positions); ax1.set_xticklabels(labels, rotation=25, ha="right")
ax1.set_ylabel("DRAM throughput (% of peak)")
ax1.set_title("Per-kernel DRAM bandwidth utilization\n(30 L1 kernels each)", fontweight="bold")
ax1.grid(axis="y", alpha=0.3); ax1.set_ylim(0, 105)

# Bar of mean SM utilization
sm_means = [stats[wl]["sm_act"]["mean"] for wl in present]
bars = ax2.bar(positions, sm_means, color=palette[:len(present)], alpha=0.85, edgecolor="#111", lw=1.2)
for bar, val in zip(bars, sm_means):
    ax2.text(bar.get_x()+bar.get_width()/2, val+2, f"{val:.1f}%", ha="center", fontsize=10)
ax2.set_xticks(positions); ax2.set_xticklabels(labels, rotation=25, ha="right")
ax2.set_ylabel("SM active cycles (% of peak)")
ax2.set_title("Mean SM utilization of L1 kernels", fontweight="bold")
ax2.grid(axis="y", alpha=0.3); ax2.set_ylim(0, 100)

plt.suptitle("Figure 13. Chain 18 Part 2 — NCU per-kernel DRAM & SM utilization on Full GPU (MPS off)\n"
             "L1 kernels profiled alone vs. with interfering AI workloads",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f13_ncu_dram_by_workload.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f13_ncu_dram_by_workload.png")

# --------------------------------------------------------------
# Figure 14 — DRAM bytes/kernel and L2 sectors
# --------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))
dram_bytes = [np.mean(stats[wl]["dram_bytes"]["values"])/1e6 if "dram_bytes" in stats[wl] else 0 for wl in present]
l2_sect    = [np.mean(stats[wl]["l2_sectors"]["values"])/1e6 if "l2_sectors" in stats[wl] else 0 for wl in present]

bars = ax1.bar(positions, dram_bytes, color=palette[:len(present)], alpha=0.85, edgecolor="#111")
for bar, val in zip(bars, dram_bytes):
    ax1.text(bar.get_x()+bar.get_width()/2, val, f"{val:.2f}MB", ha="center", va="bottom", fontsize=9)
ax1.set_xticks(positions); ax1.set_xticklabels(labels, rotation=25, ha="right")
ax1.set_ylabel("Mean DRAM bytes / kernel (MB)")
ax1.set_title("HBM traffic per L1 kernel", fontweight="bold")
ax1.grid(axis="y", alpha=0.3)

bars = ax2.bar(positions, l2_sect, color=palette[:len(present)], alpha=0.85, edgecolor="#111")
for bar, val in zip(bars, l2_sect):
    ax2.text(bar.get_x()+bar.get_width()/2, val, f"{val:.1f}M", ha="center", va="bottom", fontsize=9)
ax2.set_xticks(positions); ax2.set_xticklabels(labels, rotation=25, ha="right")
ax2.set_ylabel("Mean L2 sectors accessed / kernel (M)")
ax2.set_title("L2 cache traffic per L1 kernel", fontweight="bold")
ax2.grid(axis="y", alpha=0.3)

plt.suptitle("Figure 14. Chain 18 Part 2 — NCU memory traffic per L1 kernel on Full GPU\n"
             "DRAM bytes vs. L2 sectors per kernel (MPSoff, 30 kernels each)",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f14_ncu_traffic_by_workload.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f14_ncu_traffic_by_workload.png")
