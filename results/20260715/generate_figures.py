#!/usr/bin/env python3
"""Generate figures for 20260715 report (Chain 9 + Chain 10 combined)."""
import json, os, numpy as np
import matplotlib.pyplot as plt

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260715"
FIG = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

with open(os.path.join(BASE, "chain9_summary.json")) as f:
    c9 = json.load(f)
with open(os.path.join(BASE, "chain10_summary.json")) as f:
    c10 = json.load(f)

plt.rcParams.update({
    "font.family": ["DejaVu Sans"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False,
})

# =============================================================================
# Figure 1: Chain 9 — API-level shims all migrate the sync (TS mode)
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 6))
shims = ["baseline", "cudaFreeAsync", "cudaMemPool", "defer", "arena"]
labels = ["baseline\n(no shim)", "cudaFreeAsync\n(Ch8 A)", "cudaMemPool\n(Ch8 B)", "defer\n(free=no-op)", "arena\n(persistent pool)"]
cudaFree_alone = [c9[f"{s}_TS_c20_alone"]["cudaFree_ms"] for s in shims]
cudaFree_nrx   = [c9[f"{s}_TS_c20_nrx"]["cudaFree_ms"] for s in shims]
memcpy_nrx     = [c9[f"{s}_TS_c20_nrx"]["memcpyAsync_ms"] for s in shims]
total_nrx      = [c9[f"{s}_TS_c20_nrx"]["total_host_ms"] for s in shims]

x = np.arange(len(shims))
w = 0.25
ax.bar(x - w, cudaFree_nrx, w, color="#dc2626", label="cudaFree total")
ax.bar(x,     memcpy_nrx,   w, color="#f59e0b", label="cudaMemcpyAsync total")
ax.bar(x + w, total_nrx,    w, color="#334155", label="TOTAL host CUDA time")

for i, v in enumerate(total_nrx):
    ax.text(i + w, v + 400, f"{v:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Time (ms, 30s NSYS window)")
ax.set_title("Figure 1. Chain 9 — API-level shims: sync migrates but is not eliminated\n"
             "(all shims tested under TS + NRx coloc; total host wait ≈ 27 s regardless)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 32000)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig1_chain9_shims.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig1_chain9_shims.png")

# =============================================================================
# Figure 2: MPS mode — shims work because sync isn't there anyway
# =============================================================================
fig, ax = plt.subplots(figsize=(10, 5))
ts_total = [c9[f"{s}_TS_c20_nrx"]["total_host_ms"] for s in shims]
mps_total = [c9[f"{s}_MPS_c20_nrx"]["total_host_ms"] for s in shims]

x = np.arange(len(shims))
w = 0.35
ax.bar(x - w/2, ts_total, w, color="#dc2626", label="TS (temporal) — sync present")
ax.bar(x + w/2, mps_total, w, color="#10b981", label="MPS (spatial) — sync eliminated")

for i, (a, b) in enumerate(zip(ts_total, mps_total)):
    ax.text(i - w/2, a + 400, f"{a:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, b + 400, f"{b:,.0f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel("Total host CUDA time (ms)")
ax.set_title("Figure 2. Same shims under MPS: penalty gone regardless of shim\n"
             "(execution model dominates over API-level fixes)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left", fontsize=10)
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, 32000)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig2_shims_mps_vs_ts.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig2_shims_mps_vs_ts.png")

# =============================================================================
# Figure 3: MPS thread% — soft SM partition retains spatial benefit
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
pcts = ["100", "30", "50", "70"]
labels_p = ["MPS 100%\n(all SMs)", "MPS 30%", "MPS 50%", "MPS 70%"]
cf_alone = [c10["baseL1_MPS_alone"]["cudaFree_ms"]] + [c10[f"mpsP{p}_alone"]["cudaFree_ms"] for p in pcts[1:]]
cf_nrx   = [c10["baseL1_MPS_nrx"]["cudaFree_ms"]]   + [c10[f"mpsP{p}_nrx"]["cudaFree_ms"] for p in pcts[1:]]

x = np.arange(len(pcts))
w = 0.35
ax.bar(x - w/2, cf_alone, w, color="#10b981", label="alone (L1 only)")
ax.bar(x + w/2, cf_nrx,   w, color="#f59e0b", label="+ NRx coloc")

for i, (a, b) in enumerate(zip(cf_alone, cf_nrx)):
    ax.text(i - w/2, a + 30, f"{a:,.0f}", ha="center", fontsize=9, fontweight="bold")
    ax.text(i + w/2, b + 30, f"{b:,.0f}", ha="center", fontsize=9, fontweight="bold")

ax.set_xticks(x); ax.set_xticklabels(labels_p)
ax.set_ylabel("cudaFree total (ms)")
ax.set_title("Figure 3. MPS thread-percentage sweep — soft SM partition preserves spatial benefit\n"
             "(L1 with 30% SM budget still avoids temporal sync)",
             fontsize=12, fontweight="bold")
ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig3_mps_thread_pct.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig3_mps_thread_pct.png")

# =============================================================================
# Figure 4: Persistent megakernel — architectural fix vs API-level
# =============================================================================
fig, ax = plt.subplots(figsize=(9, 5))
groups = ["persistBase\nTS alone", "persistBase\nTS + NRx", "persistMega\nTS alone", "persistMega\nTS + NRx"]
totals = [
    c10["persistBase_TS_alone"]["total_host_ms"],
    c10["persistBase_TS_nrx"]["total_host_ms"],
    c10["persistMega_TS_alone"]["total_host_ms"],
    c10["persistMega_TS_nrx"]["total_host_ms"],
]
colors = ["#94a3b8", "#dc2626", "#10b981", "#065f46"]
bars = ax.bar(groups, totals, color=colors)
for bar, t in zip(bars, totals):
    ax.text(bar.get_x()+bar.get_width()/2, t+40, f"{t:.0f} ms", ha="center", fontsize=10, fontweight="bold")

ax.set_ylabel("Total host CUDA time (ms)")
ax.set_title("Figure 4. Persistent megakernel — architectural change eliminates the sync path\n"
             "(baseline = per-iter alloc/launch/free ; megakernel = ONE launch, zero per-iter API calls)",
             fontsize=12, fontweight="bold")
ax.grid(axis="y", alpha=0.3)
ax.set_ylim(0, max(totals) * 1.15)
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig4_persistent_kernel.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig4_persistent_kernel.png")

# =============================================================================
# Figure 5: Layer summary — which approach layer actually works
# =============================================================================
fig, ax = plt.subplots(figsize=(11, 5))
layers = [
    "CUDA runtime API\n(Chain 9 shims)",
    "Application layer\n(CUDA Graph capture)",
    "Execution model\n(MPS spatial)",
    "Execution model\n(MPS 30% thread)",
    "Architectural\n(persistent megakernel)",
]
results = [
    ("Fails — sync migrates to memcpy", "#dc2626"),
    ("Blocked — pyaerial internal D2H", "#dc2626"),
    ("Works — 1.02× penalty", "#10b981"),
    ("Works — even with 30% SM", "#10b981"),
    ("Works — sync path never triggered", "#065f46"),
]
y = np.arange(len(layers))
ax.barh(y, [1]*len(layers), color=[r[1] for r in results], alpha=0.85)
for i, (label, (text, _)) in enumerate(zip(layers, results)):
    ax.text(0.02, i, text, va="center", fontsize=11, color="white", fontweight="bold")
ax.set_yticks(y); ax.set_yticklabels(layers, fontsize=10)
ax.invert_yaxis()
ax.set_xlim(0, 1); ax.set_xticks([])
ax.set_title("Figure 5. Approach-layer summary — only execution-model or architectural changes eliminate the sync",
             fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(os.path.join(FIG, "fig5_layer_summary.png"), dpi=150, bbox_inches="tight")
plt.close()
print("Saved fig5_layer_summary.png")

print("\nAll figures →", FIG)
