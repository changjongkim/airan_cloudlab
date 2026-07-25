#!/usr/bin/env python3
"""Chain 17 N-sweep — L1 kernel-gap analysis.

For each nsys cuda_gpu_trace CSV, compute the inter-kernel gap distribution
(time from end of kernel i to start of kernel i+1). This is where cudaFree
implicit sync, driver launch overhead, and cross-context serialization
hide — none of which NCU can see.

Compare gaps across N ∈ {1,2,3,4,6,8} × MPS off/on to isolate the
per-process contention cost.
"""
import os, csv, json
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260725"
GAPS = os.path.join(BASE, "chain17_gapstats")
FIG  = os.path.join(BASE, "figures", "comprehensive")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"],
    "font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

Ns = [1, 2, 3, 4, 6, 8]
MPS_MODES = ["off", "on"]

def parse_trace_csv(path):
    """Returns list of (start_ns, dur_ns, name) for KERNELS only (not memcpy)."""
    kernels = []
    with open(path) as f:
        # find header
        rdr = csv.reader(f)
        hdr = None
        for row in rdr:
            if row and row[0] == "Start (ns)":
                hdr = row; break
        if hdr is None: return []
        idx_start = hdr.index("Start (ns)")
        idx_dur   = hdr.index("Duration (ns)")
        idx_grd   = hdr.index("GrdX")
        idx_name  = hdr.index("Name")
        idx_stream= hdr.index("Strm")
        for row in rdr:
            if len(row) <= idx_name: continue
            grd = row[idx_grd]
            name = row[idx_name]
            # Kernel rows have GrdX filled; memcpy rows leave it empty
            if not grd.strip(): continue
            if "memcpy" in name.lower() or "memset" in name.lower(): continue
            try:
                s = int(row[idx_start])
                d = int(row[idx_dur])
                kernels.append((s, d, name, row[idx_stream]))
            except: pass
    return kernels

def gap_distribution(kernels):
    """Return per-stream inter-kernel gaps (ns). Gaps computed within same stream
       so parallel-stream kernels don't confuse each other."""
    by_stream = defaultdict(list)
    for s, d, n, strm in kernels:
        by_stream[strm].append((s, d))
    gaps = []
    for strm, kl in by_stream.items():
        kl.sort()
        for i in range(1, len(kl)):
            gap = kl[i][0] - (kl[i-1][0] + kl[i-1][1])
            if gap > 0: gaps.append(gap)
    return np.array(gaps)

# --- collect all N-sweep gap data ---
gap_data = {}
kern_stats = {}
for N in Ns:
    for m in MPS_MODES:
        path = f"{GAPS}/cfgA_A_nrxN{N}_MPS{m}_t1.gputrace.csv"
        if not os.path.exists(path): continue
        kernels = parse_trace_csv(path)
        gaps = gap_distribution(kernels)
        durs = np.array([d for _,d,_,_ in kernels])
        gap_data[(N, m)] = gaps
        kern_stats[(N, m)] = dict(
            n_kernels=len(kernels),
            dur_mean=float(durs.mean()),
            dur_median=float(np.median(durs)),
            gap_mean=float(gaps.mean()),
            gap_median=float(np.median(gaps)),
            gap_p95=float(np.percentile(gaps, 95)),
            gap_p99=float(np.percentile(gaps, 99)),
            gap_max=float(gaps.max()),
            duty_cycle=float(durs.sum() / (durs.sum() + gaps.sum())),
        )
        print(f"N={N} MPS={m}: kernels={len(kernels):<6} dur_med={np.median(durs)/1000:6.2f}us  "
              f"gap_med={np.median(gaps)/1000:6.2f}us  gap_p95={np.percentile(gaps,95)/1000:7.2f}us  "
              f"gap_p99={np.percentile(gaps,99)/1000:8.2f}us  duty={durs.sum()/(durs.sum()+gaps.sum()):.4f}")

with open(f"{BASE}/kernel_gap_stats.json","w") as fp:
    json.dump({f"N{n}_MPS{m}": v for (n,m), v in kern_stats.items()}, fp, indent=2)

# ---- Figure 21: gap median/p95/p99 vs N, MPS off vs on ----
fig, axes = plt.subplots(1, 2, figsize=(15, 6))
for m, color in [("off","#dc2626"),("on","#10b981")]:
    med   = [kern_stats[(N,m)]["gap_median"]/1000 if (N,m) in kern_stats else 0 for N in Ns]
    p95   = [kern_stats[(N,m)]["gap_p95"]/1000    if (N,m) in kern_stats else 0 for N in Ns]
    p99   = [kern_stats[(N,m)]["gap_p99"]/1000    if (N,m) in kern_stats else 0 for N in Ns]
    axes[0].plot(Ns, med, "o-",  color=color, label=f"MPS {m} median", linewidth=2, markersize=8)
    axes[0].plot(Ns, p95, "s--", color=color, label=f"MPS {m} p95",    linewidth=2, markersize=8, alpha=0.85)
    axes[1].plot(Ns, p99, "^-",  color=color, label=f"MPS {m} p99",    linewidth=2, markersize=8)
axes[0].set_yscale("log")
axes[1].set_yscale("log")
for ax in axes:
    ax.set_xlabel("N (concurrent NRx processes)")
    ax.set_ylabel("Inter-kernel gap (μs)")
    ax.grid(alpha=0.3, which="both"); ax.legend(loc="upper left", fontsize=9)
axes[0].set_title("Median & p95 gap", fontweight="bold")
axes[1].set_title("p99 gap (tail latency)", fontweight="bold")
plt.suptitle("Figure 21. L1 inter-kernel gap distribution vs N (Chain 17 N-sweep)\n"
             "The bottleneck NCU can't see: idle time BETWEEN kernels grows with concurrent processes",
             fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/f21_kernel_gap_vs_N.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f21_kernel_gap_vs_N.png")

# ---- Figure 22: full gap histogram overlay for key conditions ----
fig, axes = plt.subplots(2, 3, figsize=(18, 9))
key_conds = [(1,"on"),(4,"on"),(8,"on"),(1,"off"),(4,"off"),(8,"off")]
for ax, (N, m) in zip(axes.flat, key_conds):
    if (N,m) not in gap_data: continue
    g = gap_data[(N,m)]/1000  # us
    g_clip = g[g < 1000]  # < 1ms for main body
    ax.hist(g_clip, bins=100, color=("#10b981" if m=="on" else "#dc2626"), alpha=0.75, edgecolor="#111")
    med = np.median(g); p95 = np.percentile(g,95); p99 = np.percentile(g,99)
    ax.axvline(med/1, color="#111", linestyle="--", label=f"median {med/1000:.1f}μs")
    ax.axvline(p95/1, color="#f59e0b", linestyle="--", label=f"p95 {p95/1000:.1f}μs")
    ax.axvline(p99/1, color="#dc2626", linestyle="--", label=f"p99 {p99/1000:.1f}μs")
    ax.set_xlabel("Gap (μs)"); ax.set_ylabel("Count")
    ax.set_title(f"N={N}, MPS {m} — {kern_stats[(N,m)]['n_kernels']} kernels", fontweight="bold")
    ax.legend(fontsize=9); ax.set_xlim(0, min(g.max()/1000, 1000))
plt.suptitle("Figure 22. Inter-kernel gap histogram (bins ≤ 1 ms) — MPS vs no-MPS at N=1,4,8\n"
             "Distributional evidence of driver-level sync serialization",
             fontweight="bold", y=1.005)
plt.tight_layout()
plt.savefig(f"{FIG}/f22_gap_histograms.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f22_gap_histograms.png")

# ---- Figure 23: duty cycle (fraction of wall spent in kernels) ----
fig, ax = plt.subplots(figsize=(11, 6))
for m, color in [("off","#dc2626"),("on","#10b981")]:
    duty = [100*kern_stats[(N,m)]["duty_cycle"] if (N,m) in kern_stats else 0 for N in Ns]
    ax.plot(Ns, duty, "o-", color=color, label=f"MPS {m}", linewidth=2.5, markersize=10)
    for x, y in zip(Ns, duty):
        ax.text(x, y+1.5, f"{y:.1f}%", ha="center", fontsize=9, color=color)
ax.set_xlabel("N (concurrent NRx processes)")
ax.set_ylabel("L1 GPU duty cycle: dur / (dur + gap) (%)")
ax.set_title("Figure 23. L1 GPU duty cycle vs N — how much of wall time is actual kernel execution\n"
             "MPS on preserves duty cycle; no-MPS degrades linearly with N",
             fontweight="bold")
ax.grid(alpha=0.3); ax.legend(loc="upper right"); ax.set_ylim(0, 100)
plt.tight_layout()
plt.savefig(f"{FIG}/f23_l1_duty_cycle.png", dpi=140, bbox_inches='tight')
plt.close()
print("saved f23_l1_duty_cycle.png")

# ---- Summary table ----
print()
print("=== SUMMARY (dur = kernel duration, gap = time between kernels) ===")
print(f"{'N':<4}{'MPS':<6}{'kernels':<10}{'dur_med(us)':<14}{'gap_med(us)':<14}{'gap_p95(us)':<14}{'gap_p99(us)':<14}{'duty(%)':<10}")
for N in Ns:
    for m in MPS_MODES:
        if (N,m) not in kern_stats: continue
        k = kern_stats[(N,m)]
        print(f"{N:<4}{m:<6}{k['n_kernels']:<10}{k['dur_median']/1000:<14.2f}{k['gap_median']/1000:<14.2f}{k['gap_p95']/1000:<14.2f}{k['gap_p99']/1000:<14.2f}{100*k['duty_cycle']:<10.2f}")
