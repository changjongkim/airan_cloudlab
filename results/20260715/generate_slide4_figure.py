#!/usr/bin/env python3
"""Slide 4 synthesis figure — two independent escape paths from cudaFree sync."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715"
FIG = os.path.join(BASE, "figures")

with open(os.path.join(BASE, "chain12_summary.json")) as f:
    d = json.load(f)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo", "AppleGothic", "DejaVu Sans"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.unicode_minus": False,
})

fig, (axL, axR) = plt.subplots(1, 2, figsize=(14, 6))

# =========================================================================
# LEFT — Spatial escape (TS vs MPS at ~39K launches, baseL1)
# =========================================================================
conditions = ["TS\nalone", "TS\n+ NRx", "MPS100\nalone", "MPS100\n+ NRx"]
totals_L = [
    d["baseL1_TS_alone"]["total_host_ms"],
    d["baseL1_TS_nrx"]["total_host_ms"],
    d["baseL1_MPS100_alone"]["total_host_ms"],
    d["baseL1_MPS100_nrx"]["total_host_ms"],
]
colors_L = ["#94a3b8", "#dc2626", "#94a3b8", "#10b981"]
xL = np.arange(len(conditions))
bars = axL.bar(xL, totals_L, color=colors_L, edgecolor="#111", linewidth=0.8)

for i, t in enumerate(totals_L):
    axL.text(i, t + 700, f"{t:,.0f} ms", ha="center", fontsize=11, fontweight="bold")

# Annotate the sync explosion and MPS rescue
axL.annotate("", xy=(1, 25000), xytext=(0, 25000),
             arrowprops=dict(arrowstyle="->", color="#dc2626", lw=2))
axL.text(0.5, 26000, "9.4× sync 폭발", ha="center", color="#dc2626",
         fontsize=11, fontweight="bold")

axL.annotate("", xy=(3, 5000), xytext=(1, 22000),
             arrowprops=dict(arrowstyle="->", color="#10b981", lw=2,
                             connectionstyle="arc3,rad=-0.2"))
axL.text(2.2, 15000, "MPS로\nsync 사라짐",
         ha="center", color="#10b981", fontsize=11, fontweight="bold")

axL.set_xticks(xL); axL.set_xticklabels(conditions, fontsize=10)
axL.set_ylabel("Total host CUDA time (ms)")
axL.set_title("① Spatial escape — MPS multiplex\n(same baseL1 workload, ~39K launches)",
              fontsize=12, fontweight="bold", pad=10)
axL.set_ylim(0, 31000)
axL.grid(axis="y", alpha=0.3)

# =========================================================================
# RIGHT — Architectural escape (high vs low launch under TS+NRx)
# =========================================================================
wls = ["baseL1\n(39,140)", "baseL1_arena\n(39,364)", "multiBase\n(10,100)", "multiMega\n(6)", "persistMega\n(2)"]
totals_R = [
    d["baseL1_TS_nrx"]["total_host_ms"],
    d["baseL1_arena_TS_nrx"]["total_host_ms"],
    d["multiBase_TS_nrx"]["total_host_ms"],
    d["multiMega_TS_nrx"]["total_host_ms"],
    d["persistMega_TS_nrx"]["total_host_ms"],
]
colors_R = ["#dc2626", "#eab308", "#3b82f6", "#10b981", "#10b981"]
xR = np.arange(len(wls))
axR.bar(xR, totals_R, color=colors_R, edgecolor="#111", linewidth=0.8)

for i, t in enumerate(totals_R):
    axR.text(i, t*1.4 + 5, f"{t:,.0f} ms", ha="center", fontsize=10, fontweight="bold")

# Annotate the launch collapse
axR.annotate("", xy=(3, 200), xytext=(0, 28000),
             arrowprops=dict(arrowstyle="->", color="#10b981", lw=2,
                             connectionstyle="arc3,rad=-0.3"))
axR.text(1.8, 5000, "launch 39K → 6\nsync 소멸\n(703× speedup)",
         ha="center", color="#10b981", fontsize=11, fontweight="bold",
         bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#10b981"))

axR.set_xticks(xR); axR.set_xticklabels(wls, fontsize=9)
axR.set_ylabel("Total host CUDA time (ms, log)")
axR.set_yscale("log")
axR.set_ylim(10, 80000)
axR.set_title("② Architectural escape — launch collapse\n(all under TS + NRx coloc, sync-triggering condition)",
              fontsize=12, fontweight="bold", pad=10)
axR.grid(axis="y", alpha=0.3, which="both")

fig.suptitle("Figure 13. cudaFree sync escape paths\n"
             "두 개의 독립적 경로 — spatial multiplex (MPS/MIG) 또는 architectural rewrite (megakernel)",
             fontsize=13, fontweight="bold", y=1.02)

plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig13_escape_paths.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig13_escape_paths.png")
