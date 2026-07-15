#!/usr/bin/env python3
"""Chain 11 figures — megakernel validation across 3 modes (TS/MPS100/MPS30)."""
import json, os, numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(BASE, "chain11_summary.json")) as f:
    d = json.load(f)

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

# =============================================================================
# Figure 6: All 3 workloads × 3 modes × alone/nrx — total host CUDA time
# =============================================================================
fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=False)
workloads = [("baseL1", "real cuPHY L1"), ("persistBase", "synthetic per-iter"), ("persistMega", "synthetic megakernel")]
modes = ["TS", "MPS100", "MPS30"]
mode_labels = ["TS\n(temporal)", "MPS 100%\n(spatial)", "MPS 30%\n(soft partition)"]
colors_alone = "#10b981"
colors_nrx   = "#dc2626"

for ax, (wl_key, wl_label) in zip(axes, workloads):
    alone = [d[f"{wl_key}_{m}_alone"]["total_host_ms"] for m in modes]
    nrx   = [d[f"{wl_key}_{m}_nrx"]["total_host_ms"]   for m in modes]
    x = np.arange(len(modes))
    w = 0.35
    ax.bar(x - w/2, alone, w, color=colors_alone, label="alone (L1 only)")
    ax.bar(x + w/2, nrx,   w, color=colors_nrx,   label="+ NRx coloc")
    for i, (a, b) in enumerate(zip(alone, nrx)):
        ax.text(i - w/2, a + 200, f"{a:.0f}", ha="center", fontsize=9, fontweight="bold")
        ax.text(i + w/2, b + 200, f"{b:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.set_xticks(x); ax.set_xticklabels(mode_labels, fontsize=9)
    ax.set_title(wl_label, fontsize=12, fontweight="bold")
    ax.set_ylabel("Total host CUDA time (ms)")
    ax.grid(axis="y", alpha=0.3)
    top = max(max(alone), max(nrx)) * 1.15
    ax.set_ylim(0, top)

axes[0].legend(loc="upper left", fontsize=9)
fig.suptitle("Figure 6. Chain 11 — Megakernel validation across execution modes (30s NSYS window)",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig6_chain11_3workloads_3modes.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig6_chain11_3workloads_3modes.png")

# =============================================================================
# Figure 7: Direct persistBase vs persistMega side-by-side (TS mode) — the money shot
# =============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
labels = ["alone", "+NRx coloc"]
base = [d["persistBase_TS_alone"]["total_host_ms"], d["persistBase_TS_nrx"]["total_host_ms"]]
mega = [d["persistMega_TS_alone"]["total_host_ms"], d["persistMega_TS_nrx"]["total_host_ms"]]
launch_base = [d["persistBase_TS_alone"]["launchKernel_n"], d["persistBase_TS_nrx"]["launchKernel_n"]]
launch_mega = [d["persistMega_TS_alone"]["launchKernel_n"], d["persistMega_TS_nrx"]["launchKernel_n"]]

x = np.arange(len(labels))
w = 0.35
b1 = ax1.bar(x - w/2, base, w, color="#dc2626", label="persistBase (1000 launches)")
b2 = ax1.bar(x + w/2, mega, w, color="#10b981", label="persistMega (2 launches)")
for i, (a, m) in enumerate(zip(base, mega)):
    ax1.text(i - w/2, a + 50, f"{a:.0f} ms", ha="center", fontsize=10, fontweight="bold")
    ax1.text(i + w/2, m + 50, f"{m:.0f} ms", ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(labels, fontsize=11)
ax1.set_ylabel("Total host CUDA time (ms)")
ax1.set_title("Host CUDA time (TS mode)\nsync path only triggered by per-iter API calls",
              fontsize=12, fontweight="bold")
ax1.legend(loc="upper left", fontsize=10)
ax1.grid(axis="y", alpha=0.3)

# Right: launch count vs total time
ratio = [b / m for b, m in zip(base, mega)]
ax2.bar(labels, ratio, color=["#94a3b8", "#dc2626"], alpha=0.85)
for i, r in enumerate(ratio):
    ax2.text(i, r + 1, f"{r:.1f}×", ha="center", fontsize=14, fontweight="bold")
ax2.set_ylabel("Host time reduction ratio (baseline / megakernel)")
ax2.set_title("Megakernel reduction factor\n(same GPU work, 500× fewer API calls)",
              fontsize=12, fontweight="bold")
ax2.grid(axis="y", alpha=0.3)
ax2.set_ylim(0, max(ratio) * 1.2)

fig.suptitle("Figure 7. Megakernel vs baseline — TS mode, same synthetic workload",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig7_megakernel_reduction.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig7_megakernel_reduction.png")

# =============================================================================
# Figure 8: Kernel launch count vs sync path — mechanism visualization
# Uses cuLaunchKernel counts extracted separately (persistX use cupy RawKernel → driver API):
#   baseL1 (real cuPHY):   ~9,616 (both runtime+driver LaunchKernel entries)
#   persistBase:           ~1,010 (via cuLaunchKernel driver API)
#   persistMega:           2       (single launch + warmup)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5))
wl_names = ["baseL1\n(real cuPHY,\n~9,600 kernel launches)",
            "persistBase\n(synthetic,\n1,010 launches)",
            "persistMega\n(synthetic,\n2 launches)"]
launch_counts = [9616, 1010, 2]
host_time = [
    d["baseL1_TS_nrx"]["total_host_ms"],
    d["persistBase_TS_nrx"]["total_host_ms"],
    d["persistMega_TS_nrx"]["total_host_ms"],
]

x = np.arange(len(wl_names))
w = 0.4
b1 = ax.bar(x - w/2, launch_counts, w, color="#3b82f6", label="Kernel launches (log)")
ax.set_ylabel("Kernel launch count (log)", color="#3b82f6")
ax.set_yscale("log")
ax.tick_params(axis='y', labelcolor="#3b82f6")
ax.set_ylim(1, 20000)

ax2 = ax.twinx()
b2 = ax2.bar(x + w/2, host_time, w, color="#dc2626", label="Host CUDA time (log)")
ax2.set_ylabel("Total host CUDA time (ms, log)", color="#dc2626")
ax2.set_yscale("log")
ax2.tick_params(axis='y', labelcolor="#dc2626")
ax2.set_ylim(10, 40000)

ax.set_xticks(x); ax.set_xticklabels(wl_names, fontsize=10)

for i, (lc, ht) in enumerate(zip(launch_counts, host_time)):
    ax.text(i - w/2, lc * 1.3, f"{int(lc):,}", ha="center", fontsize=10, color="#3b82f6", fontweight="bold")
    ax2.text(i + w/2, ht * 1.3, f"{ht:,.0f} ms", ha="center", fontsize=10, color="#dc2626", fontweight="bold")

ax.set_title("Figure 8. Launch-count vs sync-penalty (TS + NRx coloc)\n"
             "Host time scales with API-launch rate; GPU compute stays ~40 ms in all three cases",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig8_launches_vs_sync.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig8_launches_vs_sync.png")

# =============================================================================
# Figure 9: MPS 30% — soft SM budget robustness
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5))
wl_short = ["baseL1", "persistBase", "persistMega"]
mps100_nrx = [d[f"{w}_MPS100_nrx"]["total_host_ms"] for w in wl_short]
mps30_nrx  = [d[f"{w}_MPS30_nrx"]["total_host_ms"]  for w in wl_short]

x = np.arange(len(wl_short))
w = 0.35
ax.bar(x - w/2, mps100_nrx, w, color="#10b981", label="MPS 100% (full SM budget)")
ax.bar(x + w/2, mps30_nrx,  w, color="#059669", label="MPS 30% (soft SM cap)")
for i, (a, b) in enumerate(zip(mps100_nrx, mps30_nrx)):
    ax.text(i - w/2, a + 50, f"{a:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax.text(i + w/2, b + 50, f"{b:.0f}", ha="center", fontsize=10, fontweight="bold")
ax.set_xticks(x); ax.set_xticklabels(wl_short, fontsize=11)
ax.set_ylabel("Total host CUDA time (ms) — under NRx coloc")
ax.set_title("Figure 9. MPS 30% soft SM cap — spatial-multiplex benefit is preserved\n"
             "even when L1 is limited to 30% of GPU SMs (compute-bound co-tenant)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig9_mps30_robustness.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig9_mps30_robustness.png")

print("\nDone.")
