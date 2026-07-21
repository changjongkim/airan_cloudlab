#!/usr/bin/env python3
"""Generate PNG figures for 20260708 report."""
import json
import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260708"
FIG_DIR = os.path.join(BASE, "figures")
os.makedirs(FIG_DIR, exist_ok=True)

with open(os.path.join(BASE, "analysis", "cudaFree_all_conditions.json")) as f:
    data = json.load(f)

def ms(k):
    return data[k]["cudaFree_total_ms_mean"]

def std(k):
    return data[k]["cudaFree_total_ms_std"]

plt.rcParams.update({
    "font.family": ["DejaVu Sans"],
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# =============================================================================
# Figure 1: 4-mode L1 cudaFree comparison (bar chart with error bars)
# =============================================================================
fig, ax = plt.subplots(figsize=(12, 6))

conditions = [
    ("MIG 4g\nalone\n(no MPS)",     ms("round3_mig_mps/MIG4g_alone_noMPS"), std("round3_mig_mps/MIG4g_alone_noMPS"), "#10b981"),
    ("MIG 4g + NRx\nsame-part\n(no MPS)", ms("round3_mig_mps/MIG4g_nrx_noMPS"), std("round3_mig_mps/MIG4g_nrx_noMPS"), "#dc2626"),
    ("MIG 4g\nalone\n(with MPS)",   ms("round3_mig_mps/MIG4g_alone_MPS"), std("round3_mig_mps/MIG4g_alone_MPS"), "#10b981"),
    ("MIG 4g + NRx\nsame-part\n(with MPS)", ms("round3_mig_mps/MIG4g_nrx_MPS"), std("round3_mig_mps/MIG4g_nrx_MPS"), "#f59e0b"),
    ("Full GPU\nalone\n(TS)",       ms("mps_ts_v2/TS_alone"), std("mps_ts_v2/TS_alone"), "#10b981"),
    ("Full GPU + NRx\n(TS)",        ms("mps_ts_v2/TS_coloc"), std("mps_ts_v2/TS_coloc"), "#dc2626"),
    ("Full GPU\nalone\n(MPS)",      ms("mps_only/MPS_alone"), std("mps_only/MPS_alone"), "#10b981"),
    ("Full GPU + NRx\n(MPS)",       ms("mps_only/MPS_coloc"), std("mps_only/MPS_coloc"), "#f59e0b"),
    ("Full GPU +\nHBM stress\n(MPS)", ms("mps_hbm/MPS_hbm"), std("mps_hbm/MPS_hbm"), "#7f1d1d"),
]
labels = [c[0] for c in conditions]
values = [c[1] for c in conditions]
errors = [c[2] for c in conditions]
colors = [c[3] for c in conditions]

xs = np.arange(len(conditions))
bars = ax.bar(xs, values, yerr=errors, capsize=4, color=colors, edgecolor="black", linewidth=0.5, alpha=0.9)

for bar, v in zip(bars, values):
    ax.text(bar.get_x() + bar.get_width()/2, v + 500,
            f"{v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("cudaFree total time (ms, 30s window)", fontsize=12)
ax.set_title("Figure 1. L1 cudaFree penalty across 4 GPU sharing modes\n(cells=20, NRx / HBM stress AI workload)",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylim(0, 26000)
ax.grid(axis="y", alpha=0.3)

# Legend
safe = mpatches.Patch(color="#10b981", label="alone (baseline)")
severe = mpatches.Patch(color="#dc2626", label="coloc — temporal (no MPS) → sync path")
good = mpatches.Patch(color="#f59e0b", label="coloc — spatial (MPS) → sync bypassed")
worst = mpatches.Patch(color="#7f1d1d", label="MPS + HBM stress → different mechanism (bandwidth)")
ax.legend(handles=[safe, severe, good, worst], loc="upper left", fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig1_4mode_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_4mode_comparison.png")

# =============================================================================
# Figure 2: Round 3 fair comparison (MIG 4g × MPS on/off)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))

r3_labels = ["alone", "L1 + NRx (coloc)"]
noMPS = [ms("round3_mig_mps/MIG4g_alone_noMPS"), ms("round3_mig_mps/MIG4g_nrx_noMPS")]
withMPS = [ms("round3_mig_mps/MIG4g_alone_MPS"), ms("round3_mig_mps/MIG4g_nrx_MPS")]
noMPS_std = [std("round3_mig_mps/MIG4g_alone_noMPS"), std("round3_mig_mps/MIG4g_nrx_noMPS")]
withMPS_std = [std("round3_mig_mps/MIG4g_alone_MPS"), std("round3_mig_mps/MIG4g_nrx_MPS")]

x = np.arange(len(r3_labels))
w = 0.35
bars1 = ax.bar(x - w/2, noMPS, w, yerr=noMPS_std, capsize=4, label="MIG 4g (no MPS) — temporal", color="#dc2626", alpha=0.9)
bars2 = ax.bar(x + w/2, withMPS, w, yerr=withMPS_std, capsize=4, label="MIG 4g + MPS — spatial", color="#f59e0b", alpha=0.9)

for bar_group in [bars1, bars2]:
    for bar in bar_group:
        v = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2, v + 400,
                f"{v:,.0f}", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_ylabel("cudaFree total time (ms, 30s window)", fontsize=12)
ax.set_title("Figure 2. MIG 4g × MPS on/off at same 42 SMs",
             fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(r3_labels, fontsize=11)
ax.set_ylim(0, 22000)
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig2_round3_fair_comparison.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_round3_fair_comparison.png")

# =============================================================================
# Figure 3: Per-call cudaFree distribution (bimodal)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5))

# From all_conditions.json - fetch percentiles
dist_conditions = [
    ("TS alone",         "mps_ts_v2/TS_alone",         "#10b981"),
    ("TS + NRx",         "mps_ts_v2/TS_coloc",         "#dc2626"),
    ("MPS alone",        "mps_only/MPS_alone",         "#10b981"),
    ("MPS + NRx",        "mps_only/MPS_coloc",         "#f59e0b"),
    ("MIG 4g + NRx (no MPS)", "round3_mig_mps/MIG4g_nrx_noMPS", "#dc2626"),
    ("MIG 4g + NRx + MPS",     "round3_mig_mps/MIG4g_nrx_MPS",   "#f59e0b"),
    ("MPS + HBM stress",  "mps_hbm/MPS_hbm",            "#7f1d1d"),
]

y_pos = np.arange(len(dist_conditions))
fast_pct  = [data[k]["pct_fast_lt_1ms"]  for _, k, _ in dist_conditions]
slow_pct  = [data[k]["pct_slow_1_10ms"]  for _, k, _ in dist_conditions]
cat_pct   = [data[k]["pct_cat_gt_10ms"]  for _, k, _ in dist_conditions]

ax.barh(y_pos, fast_pct, color="#10b981", label="Fast <1ms (no sync)")
ax.barh(y_pos, slow_pct, left=fast_pct, color="#f59e0b", label="Slow 1-10ms (cross-process sync)")
ax.barh(y_pos, cat_pct, left=[f+s for f, s in zip(fast_pct, slow_pct)], color="#7f1d1d",
        label="Catastrophic >10ms (HBM contention)")

# labels on right
for i, (name, k, _) in enumerate(dist_conditions):
    p50 = data[k]["cudaFree_p50_us"]
    n = int(data[k]["cudaFree_calls_mean"])
    ax.text(101, i, f"p50={p50:,.0f}µs · n={n:,}", va="center", fontsize=9, color="gray")

ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in dist_conditions], fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, 155)
ax.set_xlabel("% of cudaFree calls in each latency bin", fontsize=11)
ax.set_title("Figure 3. Per-call cudaFree distribution — three mechanisms\n(bimodal signature reveals cross-process sync)",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=9)
ax.grid(axis="x", alpha=0.3)
ax.set_xticks([0, 25, 50, 75, 100])

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig3_percall_distribution.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_percall_distribution.png")

# =============================================================================
# Figure 4: Host CUDA time breakdown (sync migration test)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5.5))

sync_conditions = [
    ("MIG 4g alone",           "round3_mig_mps/MIG4g_alone_noMPS"),
    ("MIG 4g + NRx (no MPS)",  "round3_mig_mps/MIG4g_nrx_noMPS"),
    ("MIG 4g + NRx + MPS",     "round3_mig_mps/MIG4g_nrx_MPS"),
    ("TS + NRx (Full GPU)",    "mps_ts_v2/TS_coloc"),
    ("MPS + NRx (Full GPU)",   "mps_only/MPS_coloc"),
]

y_pos = np.arange(len(sync_conditions))
cudaFree = [data[k]["cudaFree_total_ms_mean"] for _, k in sync_conditions]
memcpy   = [data[k]["memcpyAsync_total_ms_mean"] for _, k in sync_conditions]
malloc   = [data[k]["malloc_total_ms_mean"] for _, k in sync_conditions]
total    = [data[k]["total_host_runtime_ms_mean"] for _, k in sync_conditions]
other    = [max(0, t - cf - mc - ma) for t, cf, mc, ma in zip(total, cudaFree, memcpy, malloc)]

ax.barh(y_pos, cudaFree, color="#dc2626", label="cudaFree")
ax.barh(y_pos, memcpy, left=cudaFree, color="#f59e0b", label="cudaMemcpyAsync")
ax.barh(y_pos, malloc, left=[a+b for a, b in zip(cudaFree, memcpy)], color="#3b82f6", label="cudaMalloc")
ax.barh(y_pos, other, left=[a+b+c for a, b, c in zip(cudaFree, memcpy, malloc)], color="#94a3b8", label="other")

for i, t in enumerate(total):
    ax.text(t + 500, i, f"total {t:,.0f}ms", va="center", fontsize=9)

ax.set_yticks(y_pos)
ax.set_yticklabels([c[0] for c in sync_conditions], fontsize=10)
ax.invert_yaxis()
ax.set_xlabel("Host CUDA API time (ms, 30s window)", fontsize=11)
ax.set_title("Figure 4. Host CUDA time breakdown — proof that MPS eliminates (not migrates) sync\n(Chain 8 async shim moved sync to memcpy; MPS makes it disappear)",
             fontsize=13, fontweight="bold", pad=15)
ax.legend(loc="lower right", fontsize=9)
ax.set_xlim(0, 32000)
ax.grid(axis="x", alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(FIG_DIR, "fig4_host_cuda_breakdown.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_host_cuda_breakdown.png")

print("\nAll figures saved to:", FIG_DIR)
