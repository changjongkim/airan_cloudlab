#!/usr/bin/env python3
"""Chain 17 figures — N-process sweep + MPS thread% cap + NCU."""
import json, os, csv
import matplotlib.pyplot as plt
import numpy as np

BASE = "/Users/changjongkim/New_research/cloudlab_results/results/20260724"
FIG  = os.path.join(BASE, "figures")
os.makedirs(FIG, exist_ok=True)

plt.rcParams.update({
    "font.family": ["Apple SD Gothic Neo","AppleGothic","DejaVu Sans"], "font.size": 11,
    "axes.spines.top": False, "axes.spines.right": False, "axes.unicode_minus": False,
})

with open(f"{BASE}/chain17_summary.json") as f: d = json.load(f)

# ============================================================
# Figure 1 — N-process sweep (Part A) — MPS breakdown at N>=6
# ============================================================
Ns = [1, 2, 3, 4, 6, 8]
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5))

for ax, metric, ylabel, title in [
    (ax1, "cudaFree_ms",  "L1 cudaFree total (ms)",       "cudaFree — sync signature"),
    (ax2, "l1_p99_ms",    "L1 p99 latency (ms, log)",     "L1 p99 tail latency"),
]:
    off_vals = [d.get(f"cfgA_A_nrxN{N}_MPSoff",{}).get(metric,0) for N in Ns]
    on_vals  = [d.get(f"cfgA_A_nrxN{N}_MPSon", {}).get(metric,0) for N in Ns]
    ax.plot(Ns, off_vals, "o-", color="#dc2626", label="MPS off (temporal)", linewidth=2, markersize=9)
    ax.plot(Ns, on_vals,  "s-", color="#10b981", label="MPS on (spatial)",  linewidth=2, markersize=9)
    for x,(o,n) in enumerate(zip(off_vals, on_vals)):
        if o > 0: ax.annotate(f"{o:.0f}", (Ns[x], o), textcoords="offset points", xytext=(0,10), ha="center", fontsize=9, color="#dc2626", fontweight="bold")
        if n > 0: ax.annotate(f"{n:.0f}", (Ns[x], n), textcoords="offset points", xytext=(0,-15), ha="center", fontsize=9, color="#10b981", fontweight="bold")
    if metric == "l1_p99_ms":
        ax.axhline(40, color="#111", ls=":", alpha=0.6, label="baseline ~40ms")
        ax.set_yscale("log")
    ax.axvspan(6, 8.3, alpha=0.15, color="#eab308", label="MPS breakdown zone")
    ax.set_xlabel("Number of concurrent NRx processes"); ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold"); ax.legend(loc="upper left"); ax.grid(alpha=0.3)

plt.suptitle("Figure 1. Chain 17 Part A (Config A) — N-process sweep: MPS breakdown at N≥6\n"
             "MPS on scales to N=4 (~80ms p99); N=6 catastrophic (332ms); N=8: MPS on WORSE than off",
             fontsize=13, fontweight="bold", y=1.05)
plt.tight_layout()
plt.savefig(f"{FIG}/ch17_A_Nsweep.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch17_A_Nsweep.png")

# ============================================================
# Figure 2 — MPS thread% cap (Part B) — sweet spot at pct=70
# ============================================================
pcts = [100, 70, 50, 30]
wls_b = ["nrx", "chanpred", "memcpy_loop", "embed_lookup", "ranai_mix", "ranai_mix_heavy", "nrx_multi4"]

fig, ax = plt.subplots(figsize=(13, 6))
for wl in wls_b:
    p99s = [d.get(f"cfgA_B_{wl}_pct{p}",{}).get("l1_p99_ms",0) for p in pcts]
    if all(v == 0 for v in p99s): continue
    ax.plot(pcts, p99s, "o-", label=wl, linewidth=2, markersize=8)

ax.axhline(43, color="#111", ls=":", alpha=0.5, label="baseline p99 ~43ms")
ax.set_xlabel("AI CUDA_MPS_ACTIVE_THREAD_PERCENTAGE (%)")
ax.set_ylabel("L1 p99 latency (ms)")
ax.set_title("Figure 2. Chain 17 Part B (Config A) — MPS thread% cap effect\n"
             "For multi-process (nrx_multi4): sweet spot ~pct=70 (L1 p99: 96 → 56 ms)",
             fontsize=12, fontweight="bold", pad=10)
ax.legend(loc="upper left", fontsize=9); ax.grid(alpha=0.3)
ax.invert_xaxis()  # 100 → 30 reads more naturally left-to-right
plt.tight_layout()
plt.savefig(f"{FIG}/ch17_B_thread_cap.png", dpi=150, bbox_inches='tight'); plt.close()
print("saved ch17_B_thread_cap.png")

# ============================================================
# Figure 3 — Part C NCU: DRAM utilization comparison (if available)
# ============================================================
ncu_dir = f"{BASE}/chain17_ncu"
if os.path.isdir(ncu_dir):
    dram_data = {}
    for csv_file in sorted(os.listdir(ncu_dir)):
        if not csv_file.endswith(".ncu.csv"): continue
        label = csv_file.replace(".ncu.csv","")
        path = os.path.join(ncu_dir, csv_file)
        try:
            drams = []
            with open(path, errors='ignore') as f:
                r = csv.DictReader(f)
                for row in r:
                    for k, v in row.items():
                        if k and "dram__throughput" in k and "pct_of_peak" in k:
                            try: drams.append(float(v))
                            except: pass
            if drams: dram_data[label] = {"mean": np.mean(drams), "p99": np.percentile(drams, 99), "n": len(drams)}
        except Exception as e:
            print(f"skip {csv_file}: {e}")

    if dram_data:
        labels = sorted(dram_data.keys())
        means = [dram_data[k]["mean"] for k in labels]
        fig, ax = plt.subplots(figsize=(14, 5))
        colors = ["#dc2626" if "MPSoff" in l else "#10b981" for l in labels]
        ax.bar(range(len(labels)), means, color=colors, edgecolor="#111")
        ax.set_xticks(range(len(labels))); ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Mean DRAM throughput (% of peak)")
        ax.set_title("Figure 3. Chain 17 Part C — L1 kernel DRAM utilization (NCU)\n"
                     "Direct measurement of HBM bandwidth pressure during co-tenant workload",
                     fontweight="bold")
        ax.axhline(50, color="#eab308", ls=":", alpha=0.6, label="50% peak")
        ax.grid(axis="y", alpha=0.3); ax.legend()
        plt.tight_layout()
        plt.savefig(f"{FIG}/ch17_C_dram_util.png", dpi=150, bbox_inches='tight'); plt.close()
        print(f"saved ch17_C_dram_util.png ({len(labels)} conditions)")
    else:
        print("no DRAM data found in NCU CSVs")

print("\nAll Chain 17 figures saved.")
