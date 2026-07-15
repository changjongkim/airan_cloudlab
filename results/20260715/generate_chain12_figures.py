#!/usr/bin/env python3
"""Chain 12 figures — approaches A/B/C combined validation."""
import json, os, numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(BASE, "chain12_summary.json")) as f:
    d = json.load(f)

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

# =============================================================================
# Figure 10: Chain 12 approach comparison — TS + NRx coloc
# =============================================================================
fig, ax = plt.subplots(figsize=(13, 6))

wls = ["baseL1", "baseL1_arena", "multiBase", "multiMega", "persistMega"]
wl_labels = [
    "baseL1\n(real cuPHY,\n39K launches)",
    "baseL1+arena\n(Approach C,\nreal cuPHY + shim)",
    "multiBase\n(synth 6-stage,\n10K launches)",
    "multiMega\n(Approach B,\n6 launches)",
    "persistMega\n(elementwise,\n2 launches)",
]
cudaFree = [d[f"{w}_TS_nrx"]["cudaFree_ms"] for w in wls]
memcpy   = [d[f"{w}_TS_nrx"]["memcpyAsync_ms"] for w in wls]
malloc   = [d[f"{w}_TS_nrx"]["cudaMalloc_ms"] for w in wls]
total    = [d[f"{w}_TS_nrx"]["total_host_ms"] for w in wls]
other    = [max(0, t - cf - mc - ma) for t, cf, mc, ma in zip(total, cudaFree, memcpy, malloc)]

x = np.arange(len(wls))
ax.bar(x, cudaFree, color="#dc2626", label="cudaFree")
ax.bar(x, memcpy, bottom=cudaFree, color="#f59e0b", label="cudaMemcpyAsync")
ax.bar(x, malloc, bottom=[a+b for a,b in zip(cudaFree, memcpy)], color="#3b82f6", label="cudaMalloc")
ax.bar(x, other, bottom=[a+b+c for a,b,c in zip(cudaFree, memcpy, malloc)], color="#94a3b8", label="other")

for i, t in enumerate(total):
    ax.text(i, t + 400, f"{t:.0f} ms", ha="center", fontsize=10, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(wl_labels, fontsize=9)
ax.set_ylabel("Total host CUDA time (ms) — TS + NRx coloc")
ax.set_title("Figure 10. Chain 12 — Approach A/B/C validation on TS + NRx coloc\n"
             "Real cuPHY + arena shim (C): sync migrates. Synth megakernel (B): sync eliminated. Same pattern as Chain 9/11.",
             fontsize=12, fontweight="bold", pad=12)
ax.legend(loc="upper right", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(total)*1.15)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig10_chain12_approaches.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig10_chain12_approaches.png")

# =============================================================================
# Figure 11: baseL1 vs baseL1_arena — real cuPHY sync migration
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
conditions = ["TS alone", "TS + NRx", "MPS100 alone", "MPS100 + NRx"]
base_cf = [d["baseL1_TS_alone"]["cudaFree_ms"], d["baseL1_TS_nrx"]["cudaFree_ms"],
           d["baseL1_MPS100_alone"]["cudaFree_ms"], d["baseL1_MPS100_nrx"]["cudaFree_ms"]]
base_mc = [d["baseL1_TS_alone"]["memcpyAsync_ms"], d["baseL1_TS_nrx"]["memcpyAsync_ms"],
           d["baseL1_MPS100_alone"]["memcpyAsync_ms"], d["baseL1_MPS100_nrx"]["memcpyAsync_ms"]]
arn_cf  = [d["baseL1_arena_TS_alone"]["cudaFree_ms"], d["baseL1_arena_TS_nrx"]["cudaFree_ms"],
           d["baseL1_arena_MPS100_alone"]["cudaFree_ms"], d["baseL1_arena_MPS100_nrx"]["cudaFree_ms"]]
arn_mc  = [d["baseL1_arena_TS_alone"]["memcpyAsync_ms"], d["baseL1_arena_TS_nrx"]["memcpyAsync_ms"],
           d["baseL1_arena_MPS100_alone"]["memcpyAsync_ms"], d["baseL1_arena_MPS100_nrx"]["memcpyAsync_ms"]]
base_tot = [d[f"baseL1_{m}_{c}"]["total_host_ms"] for m,c in [("TS","alone"),("TS","nrx"),("MPS100","alone"),("MPS100","nrx")]]
arn_tot  = [d[f"baseL1_arena_{m}_{c}"]["total_host_ms"] for m,c in [("TS","alone"),("TS","nrx"),("MPS100","alone"),("MPS100","nrx")]]

x = np.arange(len(conditions))
w = 0.35
b1 = ax.bar(x - w/2, base_cf, w, color="#dc2626", label="cudaFree (baseL1)")
ax.bar(x - w/2, base_mc, w, bottom=base_cf, color="#f59e0b", label="memcpyAsync (baseL1)")
b2 = ax.bar(x + w/2, arn_cf, w, color="#dc2626", alpha=0.4)
ax.bar(x + w/2, arn_mc, w, bottom=arn_cf, color="#f59e0b", alpha=0.9)

for i, (bt, at) in enumerate(zip(base_tot, arn_tot)):
    ax.text(i - w/2, bt + 400, f"{bt:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, at + 400, f"{at:.0f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(conditions, fontsize=11)
ax.set_ylabel("cudaFree + memcpyAsync time (ms)")
ax.set_title("Figure 11. Approach C on REAL cuPHY (baseL1 vs baseL1+arena_shim)\n"
             "sync migrates from cudaFree → memcpyAsync (TS+NRx: cf 18,513 → 0, memcpy 7,374 → 26,082, TOTAL 27,458 → 27,557)",
             fontsize=11, fontweight="bold", pad=12)
ax.legend(loc="upper left", fontsize=9)
ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig11_baseL1_vs_arena.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig11_baseL1_vs_arena.png")

# =============================================================================
# Figure 12: Launch count vs total host time (log-log)
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 6))
wl_data = [
    ("baseL1", "#dc2626"),
    ("baseL1_arena", "#eab308"),
    ("multiBase", "#3b82f6"),
    ("multiMega", "#10b981"),
    ("persistMega", "#059669"),
]
for wl, color in wl_data:
    launches_alone = d[f"{wl}_TS_alone"]["launches_n"]
    launches_nrx   = d[f"{wl}_TS_nrx"]["launches_n"]
    host_alone     = d[f"{wl}_TS_alone"]["total_host_ms"]
    host_nrx       = d[f"{wl}_TS_nrx"]["total_host_ms"]
    ax.scatter([launches_alone], [host_alone], s=120, color=color, marker="o",
               label=f"{wl} (alone)", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.scatter([launches_nrx], [host_nrx], s=180, color=color, marker="^",
               label=f"{wl} (+NRx)", alpha=0.85, edgecolor="black", linewidth=1)
    ax.annotate(wl, (launches_nrx, host_nrx), textcoords="offset points",
                xytext=(8, 3), fontsize=8, color=color, fontweight="bold")

ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("Kernel launches per 30 s window (log)")
ax.set_ylabel("Total host CUDA time (ms, log)")
ax.set_title("Figure 12. Launch-count vs host-time correlation (TS mode)\n"
             "Circles = alone; Triangles = + NRx coloc. NRx penalty grows with launch rate.",
             fontsize=12, fontweight="bold")
ax.grid(which="both", alpha=0.3)
ax.set_xlim(1, 100000)
ax.set_ylim(10, 40000)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig12_launches_vs_host_time.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig12_launches_vs_host_time.png")

print("\nDone.")
