#!/usr/bin/env python3
"""Chain 16 figures — realistic RAN AI multi-instance mix, HBM contention story."""
import json, os
import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"
FIG  = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
})

with open(f"{BASE}/chain16_summary.json") as f: d = json.load(f)

wls = ["ranai_mix","ranai_mix_heavy","nrx_multi4"]
wl_labels = ["ranai_mix\n(14 threads, 1 proc)", "ranai_mix_heavy\n(28 threads, 1 proc)", "nrx_multi4\n(4 separate containers)"]

# ============================================================
# Figure 1 — L1 mean + p99 latency by workload (Config A)
# ============================================================
base_mean = d.get("cfgA_SP0_baseline",{}).get("l1_mean_ms",42)
base_p99  = d.get("cfgA_SP0_baseline",{}).get("l1_p99_ms",44)

off_mean = [d.get(f"cfgA_SP_{w}_MPSoff",{}).get("l1_mean_ms",0) for w in wls]
on_mean  = [d.get(f"cfgA_SP_{w}_MPSon", {}).get("l1_mean_ms",0) for w in wls]
off_p99  = [d.get(f"cfgA_SP_{w}_MPSoff",{}).get("l1_p99_ms",0) for w in wls]
on_p99   = [d.get(f"cfgA_SP_{w}_MPSon", {}).get("l1_p99_ms",0) for w in wls]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

x = np.arange(len(wls)); w = 0.35
ax1.bar(x-w/2, off_mean, w, color="#dc2626", label="MPS off (temporal)")
ax1.bar(x+w/2, on_mean,  w, color="#10b981", label="MPS on (spatial)")
for i,(o,n) in enumerate(zip(off_mean,on_mean)):
    ax1.text(i-w/2, o+3, f"{o:.1f}", ha="center", fontsize=9, fontweight="bold")
    ax1.text(i+w/2, n+3, f"{n:.1f}", ha="center", fontsize=9, fontweight="bold")
ax1.axhline(base_mean, color="#111", ls=":", alpha=0.6, label=f"baseline {base_mean:.1f}ms")
ax1.set_xticks(x); ax1.set_xticklabels(wl_labels, fontsize=9)
ax1.set_ylabel("L1 iteration mean latency (ms)")
ax1.set_title("L1 mean latency — same-partition realistic RAN AI mix", fontweight="bold")
ax1.legend(loc="upper left"); ax1.grid(axis="y", alpha=0.3)

ax2.bar(x-w/2, off_p99, w, color="#dc2626", label="MPS off (temporal)")
ax2.bar(x+w/2, on_p99,  w, color="#10b981", label="MPS on (spatial)")
for i,(o,n) in enumerate(zip(off_p99,on_p99)):
    ax2.text(i-w/2, o*1.05, f"{o:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax2.text(i+w/2, n*1.05, f"{n:.1f}", ha="center", fontsize=9, fontweight="bold")
ax2.axhline(base_p99, color="#111", ls=":", alpha=0.6, label=f"baseline p99 {base_p99:.1f}ms")
ax2.set_xticks(x); ax2.set_xticklabels(wl_labels, fontsize=9)
ax2.set_yscale("log")
ax2.set_ylabel("L1 iteration p99 latency (ms, log)")
ax2.set_title("L1 p99 tail latency — HBM contention signature", fontweight="bold")
ax2.legend(loc="upper left"); ax2.grid(axis="y", alpha=0.3, which="both")

plt.suptitle("Figure 1. Chain 16 Config A (MIG 4g) — multi-thread vs multi-process co-tenancy",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/ch16_A_l1_latency.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch16_A_l1_latency.png")

# ============================================================
# Figure 2 — cudaFree comparison across 3 configs (nrx_multi4)
# ============================================================
cfgs = ["A (MIG 4g+3g)", "B (Full GPU)", "C (3g+2g+2g)"]
cfg_keys = ["cfgA","cfgB","cfgC"]
off_cf = [d.get(f"{c}_SP_nrx_multi4_MPSoff",{}).get("cudaFree_ms",0) for c in cfg_keys]
on_cf  = [d.get(f"{c}_SP_nrx_multi4_MPSon", {}).get("cudaFree_ms",0) for c in cfg_keys]
off_p99 = [d.get(f"{c}_SP_nrx_multi4_MPSoff",{}).get("l1_p99_ms",0) for c in cfg_keys]
on_p99  = [d.get(f"{c}_SP_nrx_multi4_MPSon", {}).get("l1_p99_ms",0) for c in cfg_keys]
base_p99_all = [d.get(f"{c}_SP0_baseline",{}).get("l1_p99_ms",0) for c in cfg_keys]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(3); w = 0.35
ax1.bar(x-w/2, off_cf, w, color="#dc2626", label="MPS off")
ax1.bar(x+w/2, on_cf,  w, color="#10b981", label="MPS on")
for i,(o,n) in enumerate(zip(off_cf,on_cf)):
    ax1.text(i-w/2, o+300, f"{o:.0f}", ha="center", fontsize=10, fontweight="bold")
    ax1.text(i+w/2, n+300, f"{n:.0f}", ha="center", fontsize=10, fontweight="bold")
ax1.set_xticks(x); ax1.set_xticklabels(cfgs)
ax1.set_ylabel("L1 cudaFree total (ms)")
ax1.set_title("nrx_multi4 — cudaFree (sync signature)", fontweight="bold")
ax1.legend(loc="upper right"); ax1.grid(axis="y", alpha=0.3)

ax2.bar(x-w/2, off_p99, w, color="#dc2626", label="MPS off")
ax2.bar(x+w/2, on_p99,  w, color="#10b981", label="MPS on")
for i,(bp,o,n) in enumerate(zip(base_p99_all, off_p99, on_p99)):
    ax2.text(i-w/2, o*1.05, f"{o:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax2.text(i+w/2, n*1.05, f"{n:.0f}", ha="center", fontsize=9, fontweight="bold")
    ax2.plot([i-0.5, i+0.5], [bp, bp], 'k:', alpha=0.6, linewidth=1.5)
ax2.set_xticks(x); ax2.set_xticklabels(cfgs)
ax2.set_yscale("log")
ax2.set_ylabel("L1 p99 latency (ms, log)")
ax2.set_title("nrx_multi4 — L1 p99 tail (HBM residual on MPS on)", fontweight="bold")
ax2.legend(loc="upper right"); ax2.grid(axis="y", alpha=0.3, which="both")

plt.suptitle("Figure 2. Chain 16 nrx_multi4 (4 separate NRx containers) — MPS partial recovery + HBM residual",
             fontsize=13, fontweight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{FIG}/ch16_nrx_multi4_configs.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch16_nrx_multi4_configs.png")

# ============================================================
# Figure 3 — Summary: MPS 완벽 회복 vs 부분 회복
# ============================================================
fig, ax = plt.subplots(figsize=(12, 6))

# recovery ratio = 1 - (l1_mps_on / l1_baseline)  (0 = 완벽 회복, negative = 나빠짐)
# actually use direct ratio: mps_on / baseline (1.0 = perfect, >1 = residual)
categories = ["ranai_mix\nMPSoff", "ranai_mix\nMPSon", "ranai_mix_heavy\nMPSoff", "ranai_mix_heavy\nMPSon",
              "nrx_multi4\nMPSoff", "nrx_multi4\nMPSon"]
ratios = [d.get(f"cfgA_SP_{w}_MPS{m}",{}).get("l1_p99_ms",0)/base_p99
          for w in wls for m in ["off","on"]]
colors = ["#dc2626","#10b981","#dc2626","#10b981","#dc2626","#10b981"]

xs = np.arange(len(categories))
ax.bar(xs, ratios, color=colors, edgecolor="#111")
for x,r in zip(xs, ratios):
    ax.text(x, r*1.05, f"{r:.1f}×", ha="center", fontweight="bold")
ax.axhline(1.0, color="#059669", ls="--", label="Perfect (baseline p99)")
ax.axhline(2.0, color="#eab308", ls=":", alpha=0.7, label="Warning (2× baseline)")
ax.set_xticks(xs); ax.set_xticklabels(categories, fontsize=9)
ax.set_ylabel("L1 p99 / baseline p99")
ax.set_yscale("log")
ax.set_title("Figure 3. MPS 회복 정도 — single-process (완벽) vs multi-process (부분)\n"
             "nrx_multi4는 MPS on해도 p99 2.6× 잔여 → HBM contention residual",
             fontweight="bold")
ax.legend(loc="upper left"); ax.grid(axis="y", alpha=0.3, which="both")
plt.tight_layout()
plt.savefig(f"{FIG}/ch16_recovery_summary.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch16_recovery_summary.png")

print("\nAll ch16 figures saved.")
